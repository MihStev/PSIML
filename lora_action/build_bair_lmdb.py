#!/usr/bin/env python
"""
Step 2 of the BAIR data pipeline (runs in the pytorch-minwm venv, GPU).

Reads the sharded raw npz files produced by extract_bair_windows.py
(images: (N,30,64,64,3) uint8, actions: (N,30,4) float32), VAE-encodes each
30-frame clip with the Wan2.1 VAE, and writes an LMDB compatible with the
existing CameraLatentLMDBDataset (wan_utils/dataset.py) so we can reuse the
existing trainer/dataloader plumbing without modification.

Camera fields (intrinsics/poses) are NOT meaningful for BAIR (no camera in
the dataset) -- filled with a fixed identity placeholder, same convention
already used in the mock training test (see CLAUDE.md, "Mock trening test").
The real conditioning signal is the new `actions` field (raw, unencoded,
full (30,4) per sample -- NOT collapsed to one vector, so the eventual
per-chunk action-injection design, still open, has the full episode to work
with), read by a small CameraLatentLMDBDataset subclass, not built here.

Mirrors the VAE wrapper + LMDB schema from
Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py, adapted for
BAIR (64x64/30 frames instead of 480x832/77, single GPU, no distributed
sharding needed given the small scale).

Usage:
    USER=mls10 LOGNAME=mls10 HOME=/home/mls10 \
    /home/mls10/venvs/pytorch-minwm/bin/python build_bair_lmdb.py \
        --in_dir /tmp/bair_raw/train --out_dir /tmp/bair_lmdb/train \
        --batch_size 64
"""
import argparse
import glob
import os
import time

import lmdb
import numpy as np
import torch

PLACEHOLDER_PROMPT = "mock bair action-conditioned clip"  # matches poc_action_injection.py

# per-sample byte estimate for LMDB map_size: latents(8,16,8,8 f16) + actions(30,4 f32)
# + intrinsics(4 f32) + poses(8,7 f32) + prompt text + key overhead
PER_SAMPLE_BYTES = (8 * 16 * 8 * 8 * 2) + (30 * 4 * 4) + (4 * 4) + (8 * 7 * 4) + 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="dir with shard_*.npz from extract_bair_windows.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vae_path", default="/data/ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=-1, help="cap number of sequences (debug)")
    return p.parse_args()


class WanVAE:
    """Same wrapper as build_worldplaygen_lmdb.py."""

    def __init__(self, vae_path, device):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Wan21"))
        from wan.modules.vae import _video_vae

        self.device = device
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
                 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
                 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.model = _video_vae(pretrained_path=vae_path, z_dim=16).eval().requires_grad_(False).to(device)

    @torch.no_grad()
    def encode(self, pixel):
        scale = [self.mean.to(self.device), 1.0 / self.std.to(self.device)]
        z = self.model.encode(pixel.to(self.device), scale).float()
        z = z.permute(0, 2, 1, 3, 4)  # (B,16,F,H,W) -> (B,F,16,H,W)
        return z.half()


def main():
    args = parse_args()
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(args.in_dir, "shard_*.npz")))
    assert shard_paths, f"no shard_*.npz found in {args.in_dir}"
    print(f"[build_lmdb] {len(shard_paths)} input shards in {args.in_dir}")

    vae = WanVAE(args.vae_path, device)

    map_size = int(45000 * PER_SAMPLE_BYTES * 1.5) + 200_000_000  # generous upper bound
    env = lmdb.open(args.out_dir, map_size=map_size, subdir=True)

    # fixed identity camera placeholder (BAIR has no camera; kept only for
    # schema compatibility with CameraLatentLMDBDataset)
    dummy_intrinsics = np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)  # fx,fy,cx,cy (normalized)

    count = 0
    lat_shape = intr_shape = poses_shape = None
    t0 = time.time()

    for shard_path in shard_paths:
        d = np.load(shard_path)
        images, actions = d["images"], d["actions"]  # (n,30,64,64,3) uint8, (n,30,4) f32
        n = images.shape[0]
        if args.limit > 0:
            n = min(n, args.limit - count)
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch = images[start:end]  # (bs,30,64,64,3)
            bs = batch.shape[0]
            pixel = torch.from_numpy(batch).float().permute(0, 4, 1, 2, 3)  # (bs,3,30,64,64)
            pixel = (pixel / 255.0 - 0.5) * 2.0
            latents = vae.encode(pixel).cpu().numpy()  # (bs, F_lat, 16, 8, 8)
            del pixel
            n_lat = latents.shape[1]
            poses = np.zeros((n_lat, 7), dtype=np.float32)
            poses[:, 6] = 1.0  # identity quaternion [0,0,0,1]

            with env.begin(write=True) as txn:
                for j in range(bs):
                    gi = count + j
                    txn.put(f"latents_{gi}_data".encode(), latents[j].tobytes())
                    txn.put(f"prompts_{gi}_data".encode(), PLACEHOLDER_PROMPT.encode())
                    txn.put(f"intrinsics_{gi}_data".encode(), dummy_intrinsics.tobytes())
                    txn.put(f"poses_{gi}_data".encode(), poses.tobytes())
                    txn.put(f"actions_{gi}_data".encode(), actions[start + j].astype(np.float32).tobytes())

            if lat_shape is None:
                lat_shape, intr_shape, poses_shape = latents.shape[1:], dummy_intrinsics.shape, poses.shape
                print(f"[build_lmdb] first batch: latent={latents.shape} actions={actions[start:end].shape}")

            count += bs
            elapsed = time.time() - t0
            print(f"[build_lmdb] {count} samples ({count/elapsed:.1f}/s)", end="\r")
        if args.limit > 0 and count >= args.limit:
            break

    with env.begin(write=True) as txn:
        txn.put(b"latents_shape", f"{count} {' '.join(map(str, lat_shape))}".encode())
        txn.put(b"prompts_shape", f"{count}".encode())
        txn.put(b"intrinsics_shape", f"{count} {' '.join(map(str, intr_shape))}".encode())
        txn.put(b"poses_shape", f"{count} {' '.join(map(str, poses_shape))}".encode())
        txn.put(b"actions_shape", f"{count} 30 4".encode())
    env.sync()
    env.close()
    print(f"\n[build_lmdb] DONE. {count} samples -> {args.out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
