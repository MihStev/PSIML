#!/usr/bin/env python
"""
Monte Carlo free-rollout evaluation: N independent (scene, random action program)
trials, batched, depths 1..max_depth. Unlike rollout_metrics.py (which compares one
FIXED direction, e.g. "right", shared across a whole batch of scenes), here EVERY
scene gets its OWN independently-drawn action per block. This removes the confound
of testing controllability under only one hand-picked command -- the two ad-hoc demo
clips we generated earlier (up-up-down-down vs right x6) already hinted this might
matter (50% vs 33% landed), so this tests it properly, at scale.

Metrics, and why each is scoped the way it is:

  - Direction accuracy: per scene, per depth, binary "did the arm move as ITS OWN
    commanded direction says" (reusing the corrected landed-check from
    generate_sequence.py -- untracked/arm-not-found counts as a FAILURE, not a
    dropped sample, per the survivorship-bias bug found and fixed earlier the same
    day). With n=256 independent trials per depth this is a real empirical
    distribution, not a small-sample estimate -- a simple normal-approx CI is
    reported alongside the point estimate.

  - FVD*/FID: computed once per depth over the POOLED freshest-generated window
    across all 256 trials (this is a distribution distance, not a per-sample
    quantity -- same S3D/Kinetics-400 caveat as rollout_metrics.py, NOT the
    canonical FVD; same 64x64-upsampled-to-224x224 caveat as our FID elsewhere).

  - PSNR/SSIM: ONLY at depth 1. BAIR episodes are 30 raw frames, which is exactly
    enough for one context block + one real target block -- there is no stored
    ground truth for a "real block 2" or "real block 3", so PSNR/SSIM are
    mathematically undefined beyond depth 1, not just uncomputed. Reported here
    with real n=256 pairs, tighter than the 64-scene evaluate.py table.

Usage:
    python rollout_metrics_mc.py --n_trials 256 --max_depth 3 --seed 0
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
WINDOW = 16
ACTIONS = ["up", "down", "left", "right"]
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0), "still": (0.0, 0.0)}
EXPECTED = {"up": (1, -1), "down": (1, +1), "right": (0, +1), "left": (0, -1)}  # (axis, sign)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--n_trials", type=int, default=256)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--out_json", default="/home/mls10/logs/rollout_metrics_mc.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--dmd_schedule", action="store_true")
    return p.parse_args()


def frechet(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def arm_xy(frame):
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = (r > g + 25) & (r > b + 25) & (r < 170)
    if m.sum() < 20:
        return np.nan, np.nan
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t_start = time.time()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion  # noqa: E402
    print("=== loading model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load(args.base_checkpoint, map_location="cpu")
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
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    psnr_m = PeakSignalNoiseRatio(data_range=255.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)

    def fid_update_chunked(metric, frames_uint8, real, chunk=64):
        """frames_uint8: (N, T, H, W, 3) numpy uint8. Feeds Inception in small
        chunks -- a single (n_scenes*16)-image forward pass OOMs once n_scenes
        gets into the hundreds (confirmed: 4096 images needed 10.8GiB in one shot)."""
        flat = frames_uint8.reshape(-1, *frames_uint8.shape[2:])  # (N*T, H, W, 3)
        for i in range(0, flat.shape[0], chunk):
            t = torch.tensor(flat[i:i + chunk].copy()).permute(0, 3, 1, 2).to(device).to(torch.uint8)
            metric.update(t, real=real)

    def video_features(frames_uint8):
        x = torch.tensor(frames_uint8.copy()).float().permute(0, 4, 1, 2, 3).to(device) / 255.0
        x = Fn.interpolate(x.reshape(-1, 3, x.shape[3], x.shape[4]), size=(224, 224),
                            mode="bilinear", align_corners=False)
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
    act_shape = get_array_shape_from_lmdb(env, "actions")
    F_LAT = lat_shape[1]
    n_ctx = config.num_frame_per_block
    n_total = lat_shape[0]
    print(f"=== test LMDB has {n_total} scenes; requested {args.n_trials} trials ===", flush=True)
    if args.dmd_schedule:
        model.scheduler.set_timesteps(1000)
        _full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        _sch = _full[[1000 - i for i in (1000, 750, 500, 250)]]
        model.scheduler.sigmas = (_sch / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = _sch
        print(f"=== DMD 4-step schedule: {[round(float(t),1) for t in _sch]} ===", flush=True)
    else:
        model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    def embed_batch(prev_dirs, cur_dirs):
        B = len(cur_dirs)
        apl = np.zeros((B, F_LAT, 16), dtype=np.float32)
        for b in range(B):
            pdx, pdy = DIRS[prev_dirs[b]]
            for i in range(1, n_ctx):
                apl[b, i] = np.tile([pdx, pdy, 0.5, 0.25], 4)
            cdx, cdy = DIRS[cur_dirs[b]]
            for i in range(n_ctx, F_LAT):
                apl[b, i] = np.tile([cdx, cdy, 0.5, 0.25], 4)
        a = torch.tensor(apl, device=device)
        an = (a - a_mean) / a_std
        an[:, 0, :] = 0.0
        return action_encoder(an.to(torch.bfloat16))

    def embed_real(raw_actions_batch):
        """raw_actions_batch: (B, 30, 4) numpy. Same alignment as bair_dataset.py's
        BairActionLatentDataset._align_to_latent_frames -- latent i>=1 gets
        raw_actions[4*(i-1):4*i] flattened (16 dims), latent 0 stays zero. This is
        the SCENE'S OWN recorded action, used only for the PSNR/SSIM comparison
        (which requires the generation to be conditioned on whatever action
        actually produced the real target frames -- a random command would be
        compared against an unrelated future, which is meaningless)."""
        B = raw_actions_batch.shape[0]
        apl = np.zeros((B, F_LAT, 16), dtype=np.float32)
        for i in range(1, F_LAT):
            start = 4 * (i - 1)
            chunk = raw_actions_batch[:, start:start + 4, :].reshape(B, -1)
            apl[:, i, :chunk.shape[1]] = chunk
        a = torch.tensor(apl, device=device)
        an = (a - a_mean) / a_std
        an[:, 0, :] = 0.0
        return action_encoder(an.to(torch.bfloat16))

    def run_block(context, ae, vm, ks):
        """One denoising rollout of the next block, given a context and an
        already-computed action embedding. Factored out so the random-command
        pass and the real-action PSNR/SSIM pass share identical sampling code."""
        B = context.shape[0]
        window = torch.cat([context, torch.randn_like(context)], dim=1)
        clean = torch.cat([context, context], dim=1)
        for i, t_val in enumerate(schedule):
            ts = torch.zeros((B, F_LAT), device=device, dtype=torch.bfloat16)
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
        return window[:, n_ctx:].clone()

    # ---- draw the Monte Carlo trials up front: which scene, which action program ----
    trial_scenes = rng.choice(n_total, size=args.n_trials, replace=(args.n_trials > n_total))
    trial_programs = [[ACTIONS[i] for i in rng.integers(0, 4, size=args.max_depth)]
                       for _ in range(args.n_trials)]

    # ---- real reference pool for FVD*/FID: ALL scenes' real target-block windows ----
    print(f"=== building real reference ({n_total} scenes) ===", flush=True)
    real_wins = []
    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:])
                        for i in range(start, end)])
        r = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
        real_wins.append(decode(r)[:, -WINDOW:])
    real_wins = np.concatenate(real_wins, 0)
    real_feat = video_features(real_wins)
    print(f"    real windows: {real_wins.shape}", flush=True)

    gen_by_depth = {d: [] for d in range(1, args.max_depth + 1)}
    dir_hit = {d: 0 for d in range(1, args.max_depth + 1)}
    dir_total = {d: 0 for d in range(1, args.max_depth + 1)}

    vm_cache, ks_cache = {}, {}

    def get_vm_ks(B):
        if B not in vm_cache:
            vm_cache[B] = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, F_LAT, 1, 1)
            ks_cache[B] = torch.tensor([[.5, 0, .5], [0, .5, .5], [0, 0, 1]], device=device,
                                        dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(B, F_LAT, 1, 1)
        return vm_cache[B], ks_cache[B]

    t0 = time.time()
    for start in range(0, args.n_trials, args.batch_size):
        end = min(start + args.batch_size, args.n_trials)
        B = end - start
        idx = trial_scenes[start:end]
        programs = trial_programs[start:end]

        lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, int(i), shape=lat_shape[1:])
                        for i in idx])
        real = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
        raw_actions = np.stack([retrieve_row_from_lmdb(env, "actions", np.float32, int(i), shape=act_shape[1:])
                                for i in idx])   # (B, 30, 4), this scene's TRUE recorded actions
        context = real[:, :n_ctx].clone()
        real_target = real[:, n_ctx:].clone()   # depth-1 ground truth, for PSNR/SSIM only
        vm, ks = get_vm_ks(B)

        prev_dirs = ["still"] * B
        for depth_i in range(args.max_depth):
            d = depth_i + 1
            cur_dirs = [programs[b][depth_i] for b in range(B)]
            ae = embed_batch(prev_dirs, cur_dirs)
            new_block = run_block(context, ae, vm, ks)
            frames = decode(new_block)   # (B, 16, 64, 64, 3)
            gen_by_depth[d].append(frames)

            for b in range(B):
                axis, sign = EXPECTED[cur_dirs[b]]
                p0 = arm_xy(frames[b, 0])
                p1 = arm_xy(frames[b, -1])
                dir_total[d] += 1
                if np.isnan(p0).any() or np.isnan(p1).any():
                    continue  # untracked counts as failure (denominator already incremented)
                delta = (p1[0] - p0[0], p1[1] - p0[1])
                if (delta[axis] * sign) > 0:
                    dir_hit[d] += 1

            if d == 1:
                # SEPARATE pass, conditioned on the scene's OWN recorded action --
                # comparing the random-command block against real_target would be
                # scoring against an unrelated future (see the smoke-test bug this
                # fixes). Same starting context, independent noise draw.
                ae_real = embed_real(raw_actions)
                block_real = run_block(context, ae_real, vm, ks)
                gen_real = decode(block_real)
                gt = decode(real_target)
                g_t = torch.tensor(gen_real.copy()).permute(0, 1, 4, 2, 3).flatten(0, 1).to(device).to(torch.uint8)
                r_t = torch.tensor(gt.copy()).permute(0, 1, 4, 2, 3).flatten(0, 1).to(device).to(torch.uint8)
                psnr_m.update(g_t.float(), r_t.float())
                ssim_m.update(g_t.float(), r_t.float())

            context = new_block
            prev_dirs = cur_dirs

        print(f"    trials {start}-{end} done ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== computing FVD*/FID per depth ===", flush=True)
    results = {}
    for d in range(1, args.max_depth + 1):
        gen = np.concatenate(gen_by_depth[d], 0)
        gf = video_features(gen)
        fvd = frechet(real_feat, gf)
        fid_m.reset()
        fid_update_chunked(fid_m, real_wins, real=True)
        fid_update_chunked(fid_m, gen, real=False)
        fid = float(fid_m.compute())
        lo, hi = wilson_ci(dir_hit[d], dir_total[d])
        results[d] = {
            "fvd_s3d": fvd, "fid": fid,
            "dir_acc": dir_hit[d] / dir_total[d], "dir_hit": dir_hit[d], "dir_total": dir_total[d],
            "dir_ci95": [lo, hi],
        }
        if d == 1:
            results[d]["psnr"] = float(psnr_m.compute())
            results[d]["ssim"] = float(ssim_m.compute())
        with open(args.out_json, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"  depth {d}: FVD*={fvd:8.1f}  FID={fid:6.2f}  dir_acc={results[d]['dir_acc']:.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f})" + (f"  PSNR={results[d].get('psnr'):.2f} SSIM={results[d].get('ssim'):.4f}" if d == 1 else ""),
              flush=True)

    print("\n=== MONTE CARLO FREE-ROLLOUT SUMMARY ===", flush=True)
    print(f"{'depth':>6} {'n':>5} {'FVD*':>9} {'FID':>7} {'dir_acc':>9} {'95% CI':>16} {'PSNR':>7} {'SSIM':>7}", flush=True)
    for d in sorted(results):
        r = results[d]
        psnr_s = f"{r['psnr']:.2f}" if "psnr" in r else "n/a"
        ssim_s = f"{r['ssim']:.4f}" if "ssim" in r else "n/a"
        print(f"{d:>6} {r['dir_total']:>5} {r['fvd_s3d']:>9.1f} {r['fid']:>7.2f} {r['dir_acc']:>9.3f} "
              f"[{r['dir_ci95'][0]:.3f},{r['dir_ci95'][1]:.3f}] {psnr_s:>7} {ssim_s:>7}", flush=True)
    print(f"\n  PSNR/SSIM reported ONLY at depth 1 -- BAIR episodes (30 raw frames) have no", flush=True)
    print(f"  stored ground truth beyond one generated block, so they are undefined past depth 1.", flush=True)
    print(f"  FVD* = S3D/Kinetics-400 Frechet distance, relative measure only, not canonical FVD.", flush=True)
    print(f"\n  total wall time: {time.time()-t_start:.0f}s", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
