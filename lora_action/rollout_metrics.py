#!/usr/bin/env python
"""
Metrics as a function of FREE-ROLLOUT DEPTH.

This answers both of Danilo's points in one experiment:

  1. "FID is temporally blind, add FVD."  FID pools per-frame Inception features,
     so it cannot see whether motion is coherent -- which is the only thing our
     project actually claims. We add an FVD-style Frechet distance over
     spatio-temporal features.
  2. "Is everything teacher-forced? Do we have free rollouts?"  Yes to the first,
     and this script measures the second properly: the model consumes its OWN
     output block after block, and we measure how quality and control decay with
     depth.

Comparison design. Real BAIR episodes are only 30 frames, so beyond depth 1 there
is no paired ground truth and PSNR/SSIM are undefined. Instead, at every depth we
take the LAST 16 decoded frames -- the most recently generated block -- and compare
that window against random 16-frame windows from real held-out clips. So the
question at each depth is "does the most recent output still look real?", asked
identically at every depth.

FVD caveat, stated up front: canonical FVD uses a specific Kinetics I3D checkpoint
that is not installable here, so we use torchvision's S3D (also Kinetics-400, the
architectural successor). Our numbers are therefore NOT comparable to published
FVD; they are valid only as a RELATIVE measure between our own conditions. The
same is true of our FID at this sample count, and of feeding 64x64 frames to a
network expecting 224x224.

Usage:
    python rollout_metrics.py --max_depth 6 --n_scenes 32
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USER", "mls10")
os.environ.setdefault("LOGNAME", "mls10")
os.environ.setdefault("HOME", "/home/mls10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/mls10/minWM-dawidzard/Wan21")
sys.path.insert(0, "/home/mls10/minWM-dawidzard/shared")
sys.path.insert(0, "/home/mls10/minWM-dawidzard/lora_action")
os.chdir("/home/mls10/minWM-dawidzard")

import lmdb
import numpy as np
import torch
import torch.nn.functional as Fn
from omegaconf import OmegaConf
from scipy import linalg

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402

D = 0.07
WINDOW = 16   # decoded frames per generated block


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--n_scenes", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--out_json", default="/home/mls10/logs/rollout_metrics.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def frechet(f1, f2):
    """Frechet distance between two sets of feature vectors (numpy, N x D)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def arm_centroid_x(frames):
    f = frames.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mask = (r > g + 25) & (r > b + 25) & (r < 170)
    xs = np.arange(f.shape[2])
    out = np.full(f.shape[0], np.nan, dtype=np.float32)
    for i in range(f.shape[0]):
        m = mask[i]
        if m.sum() >= 20:
            out[i] = (xs[None, :] * m).sum() / m.sum()
    return out


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion  # noqa: E402
    print("=== loading model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load("/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt", map_location="cpu")
    gen_sd = base.get("generator_ema", base.get("generator"))
    try:
        model.generator.load_state_dict(gen_sd)
    except RuntimeError:
        model.generator.load_state_dict(
            {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}, strict=False)
    for m in (model.generator, model.text_encoder, model.vae):
        m.to(device=device, dtype=torch.bfloat16)

    from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
    a_mean, a_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    print("=== loading feature extractors ===", flush=True)
    from torchvision.models.video import s3d, S3D_Weights
    s3d_net = s3d(weights=S3D_Weights.KINETICS400_V1).eval().to(device)
    S3D_MEAN = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    S3D_STD = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def video_features(frames_uint8):
        """(N, T, H, W, 3) uint8 -> (N, 1024) S3D features."""
        x = torch.tensor(frames_uint8.copy()).float().permute(0, 4, 1, 2, 3).to(device) / 255.0
        x = Fn.interpolate(x.flatten(0, 1) if False else x.reshape(-1, 3, x.shape[3], x.shape[4]),
                           size=(224, 224), mode="bilinear", align_corners=False)
        x = x.reshape(frames_uint8.shape[0], frames_uint8.shape[1], 3, 224, 224).permute(0, 2, 1, 3, 4)
        x = (x - S3D_MEAN) / S3D_STD
        feats = []
        for i in range(0, x.shape[0], 4):
            f = s3d_net.features(x[i:i + 4])
            feats.append(Fn.adaptive_avg_pool3d(f, 1).flatten(1).float().cpu().numpy())
        return np.concatenate(feats, 0)

    cond = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    F = lat_shape[1]
    n_ctx = config.num_frame_per_block
    n_total = min(args.n_scenes, lat_shape[0])
    model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    def embed(B, direction):
        apl = np.zeros((B, F, 16), dtype=np.float32)
        dx, dy = (0.0, 0.0) if direction is None else ((-D, 0.0) if direction == "right" else (+D, 0.0))
        for i in range(1, F):
            apl[:, i] = np.tile([dx, dy, 0.5, 0.25], 4)
        a = torch.tensor(apl, device=device)
        an = (a - a_mean) / a_std
        an[:, 0, :] = 0.0
        return action_encoder(an.to(torch.bfloat16))

    def rollout(context, direction, vm, ks, depth):
        """Free rollout: each block's context is the PREVIOUS GENERATED block."""
        B = context.shape[0]
        ae = embed(B, direction)
        blocks = []
        for _ in range(depth):
            window = torch.cat([context, torch.randn_like(context)], dim=1)
            clean = torch.cat([context, context], dim=1)
            for i, t_val in enumerate(schedule):
                ts = torch.zeros((B, F), device=device, dtype=torch.bfloat16)
                ts[:, n_ctx:] = t_val.item()
                window[:, :n_ctx] = context
                _, x0 = model.generator(noisy_image_or_video=window, conditional_dict=cond, timestep=ts,
                                        clean_x=clean, aug_t=None, viewmats=vm, Ks=ks, action_embed=ae)
                x0 = x0.float().clamp(-6, 6)
                if i == len(schedule) - 1:
                    window[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
                else:
                    sn = float(model.scheduler.sigmas[i + 1])
                    window[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
            context = window[:, n_ctx:].clone()
            blocks.append(context.clone())
        return blocks

    # ---- real reference windows ----
    print(f"=== building real reference ({n_total} held-out clips) ===", flush=True)
    real_wins = []
    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:])
                        for i in range(start, end)])
        r = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
        real_wins.append(decode(r)[:, -WINDOW:])
    real_wins = np.concatenate(real_wins, 0)
    real_feat = video_features(real_wins)
    fid_m.reset()
    fid_m.update(torch.tensor(real_wins.copy()).permute(0, 1, 4, 2, 3).flatten(0, 1).to(device).to(torch.uint8),
                 real=True)
    print(f"    real windows: {real_wins.shape}", flush=True)

    results = {}
    gen_by_depth = {d: [] for d in range(1, args.max_depth + 1)}
    div_by_depth = {d: [] for d in range(1, args.max_depth + 1)}
    dir_ok = {d: [0, 0] for d in range(1, args.max_depth + 1)}

    t0 = time.time()
    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        B = end - start
        lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:])
                        for i in range(start, end)])
        real = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
        ctx0 = real[:, :n_ctx].clone()
        vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, F, 1, 1)
        ks = torch.tensor([[.5, 0, .5], [0, .5, .5], [0, 0, 1]], device=device,
                          dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(B, F, 1, 1)

        blocks_r = rollout(ctx0, "right", vm, ks, args.max_depth)
        blocks_l = rollout(ctx0, "left", vm, ks, args.max_depth)
        for d in range(1, args.max_depth + 1):
            fr = decode(blocks_r[d - 1])[:, -WINDOW:]
            fl = decode(blocks_l[d - 1])[:, -WINDOW:]
            gen_by_depth[d].append(fr)
            div_by_depth[d].append(float(np.abs(fr.astype(np.float32) - fl.astype(np.float32)).mean()))
            for b in range(B):
                xr, xl = arm_centroid_x(fr[b]), arm_centroid_x(fl[b])
                if np.isnan(xr[0]) or np.isnan(xr[-1]) or np.isnan(xl[0]) or np.isnan(xl[-1]):
                    continue
                dir_ok[d][1] += 1
                dir_ok[d][0] += int((xr[-1] - xr[0]) > (xl[-1] - xl[0]))
        print(f"    scenes {start}-{end} done ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== computing metrics per depth ===", flush=True)
    for d in range(1, args.max_depth + 1):
        gen = np.concatenate(gen_by_depth[d], 0)
        gf = video_features(gen)
        fvd = frechet(real_feat, gf)
        fid_m.reset()
        fid_m.update(torch.tensor(real_wins.copy()).permute(0, 1, 4, 2, 3).flatten(0, 1).to(device).to(torch.uint8), real=True)
        fid_m.update(torch.tensor(gen.copy()).permute(0, 1, 4, 2, 3).flatten(0, 1).to(device).to(torch.uint8), real=False)
        fid = float(fid_m.compute())
        ok, n = dir_ok[d]
        results[d] = {"fvd_s3d": fvd, "fid": fid, "divergence": float(np.mean(div_by_depth[d])),
                      "dir_rel": ok / n if n else None, "n_dir": n,
                      "frames": int(1 + 4 * n_ctx * d)}
        with open(args.out_json, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"  depth {d}: FVD*={fvd:8.1f}  FID={fid:6.2f}  div={results[d]['divergence']:6.2f}  "
              f"dir_rel={results[d]['dir_rel']}", flush=True)

    print("\n=== FREE-ROLLOUT DEGRADATION (depth 1 = the teacher-forced-context regime) ===", flush=True)
    print(f"{'depth':>6} {'frames':>7} {'FVD*':>9} {'FID':>7} {'div':>7} {'dir_rel':>8}", flush=True)
    for d in sorted(results):
        r = results[d]
        print(f"{d:>6} {r['frames']:>7} {r['fvd_s3d']:>9.1f} {r['fid']:>7.2f} "
              f"{r['divergence']:>7.2f} {r['dir_rel']:>8.3f}", flush=True)
    print("\n  FVD* = Frechet distance over S3D/Kinetics-400 features, NOT the canonical I3D FVD.", flush=True)
    print("  Relative comparison between depths only; absolute values are not literature-comparable.", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
