#!/usr/bin/env python
"""
Step 1 of the BAIR data pipeline (runs in the tf-bair venv, CPU only).

Reads the BAIR TFRecord dataset and dumps raw image_main + action arrays
(full 30-frame sequences, unencoded) into sharded .npz files under a scratch
directory. This is intentionally the ONLY thing this script does — no VAE
encoding here, that happens in the pytorch-minwm venv (build_bair_lmdb.py),
per the two-venv split documented in CLAUDE.md (TF venv cuts windows ->
PyTorch venv VAE-encodes -> light Dataset).

Usage:
    /home/mls10/venvs/tf-bair/bin/python extract_bair_windows.py \
        --split train --out_dir /tmp/bair_raw/train --shard_size 2000
"""
import argparse
import os
import time

import numpy as np
import tensorflow_datasets as tfds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data/bair_robot_pushing_small_tfrecord/2.0.0")
    p.add_argument("--split", required=True, choices=["train", "test"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--shard_size", type=int, default=2000)
    p.add_argument("--limit", type=int, default=-1, help="cap number of sequences (debug)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    builder = tfds.builder_from_directory(args.data_dir)
    n_total = builder.info.splits[args.split].num_examples
    n = n_total if args.limit < 0 else min(args.limit, n_total)
    print(f"[extract] split={args.split} total={n_total} using={n} shard_size={args.shard_size}")

    ds = builder.as_dataset(split=args.split).take(n)

    images_buf, actions_buf = [], []
    shard_idx = 0
    written = 0
    t0 = time.time()

    def flush():
        nonlocal images_buf, actions_buf, shard_idx
        if not images_buf:
            return
        images = np.stack(images_buf, axis=0)   # (n, 30, 64, 64, 3) uint8
        actions = np.stack(actions_buf, axis=0)  # (n, 30, 4) float32
        out_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.npz")
        np.savez(out_path, images=images, actions=actions)
        print(f"[extract] wrote {out_path} ({len(images_buf)} samples, "
              f"{os.path.getsize(out_path)/1e6:.1f} MB)")
        images_buf, actions_buf = [], []
        shard_idx += 1

    for ex in ds.as_numpy_iterator():
        images_buf.append(ex["image_main"])   # (30, 64, 64, 3) uint8
        actions_buf.append(ex["action"])      # (30, 4) float32
        written += 1
        if len(images_buf) >= args.shard_size:
            flush()
        if written % 5000 == 0:
            elapsed = time.time() - t0
            print(f"[extract] {written}/{n} ({written/elapsed:.1f}/s)")

    flush()
    print(f"[extract] DONE. {written} sequences -> {shard_idx} shards in {args.out_dir} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
