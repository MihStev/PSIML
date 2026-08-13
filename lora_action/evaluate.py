#!/usr/bin/env python
"""
Full evaluation suite, on the HELD-OUT TEST SPLIT (never trained on).

Two modes, because they answer different questions and PSNR/SSIM are only
meaningful in the first one:

  MODE A -- real action from the episode. The ground-truth continuation is the
    correct answer, so we can measure reconstruction fidelity:
        1. PSNR      (generated vs GT, generated frames only)
        2. SSIM      (same)
        3. FID       (distribution-level realism; needs many frames, so treat
                      it as a RELATIVE measure between checkpoints, not as an
                      absolute number)

  MODE B -- the same scene and the same noise, but the displacement dim forced
    to +/-0.07 (inside the real +/-0.07 range). There is no ground truth here
    by construction -- the model is supposed to produce something different:
        4. action-swap divergence  (L1 between the two generated futures;
                                    the context part must stay ~0)
        5. direction accuracy      (does the gripper actually move the way the
                                    action says? uses the red-pixel centroid
                                    tracker that verified the sign convention
                                    against raw BAIR pixels)

Sign convention (verified empirically, see CLAUDE.md): dim0 < 0 moves the
gripper RIGHT in the image, dim0 > 0 moves it LEFT.

Runs over several checkpoints in ONE model load, so we get a
"controllability vs training steps" curve -- which also empirically tests the
mentor's claim that ~2500 iterations is the minimum for control to appear.

Usage:
    python evaluate.py --checkpoints /home/mls10/checkpoints/bair_lora_big/step_*.pt
    python evaluate.py --n_scenes 64 --skip_fid
"""
import argparse
import glob
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
from omegaconf import OmegaConf

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402

DISP_MAX = 0.07          # hard range of the real BAIR displacement dims
SANITY_MAX_SIGMA = 6.0   # refuse out-of-distribution actions (this was bug #1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+",
                    default=sorted(glob.glob("/home/mls10/checkpoints/bair_lora_big/step_*.pt"),
                                   key=lambda x: int(x.split("_")[-1].split(".")[0])))
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test", help="HELD-OUT split")
    p.add_argument("--n_scenes", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--out_json", default="/home/mls10/logs/eval_results.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_checkpoint", default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--dmd_schedule", action="store_true",
                    help="use the distilled model's own 4-step schedule (config denoising_step_list\n                          [1000,750,500,250], warped through the 1000-step sigma schedule) instead\n                          of set_timesteps(--n_steps); required for a fair few-step DMD test")
    return p.parse_args()


def arm_centroid_x(frames_uint8):
    """Horizontal centroid of the gripper (dark red / maroon) per frame.
    Same detector used to verify the sign convention against raw BAIR pixels.
    frames: (F, H, W, 3) uint8 -> (F,) float, NaN where the arm isn't found."""
    f = frames_uint8.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mask = (r > g + 25) & (r > b + 25) & (r < 170)
    H, W = f.shape[1], f.shape[2]
    xs = np.arange(W)[None, None, :]
    out = np.full(f.shape[0], np.nan, dtype=np.float32)
    for i in range(f.shape[0]):
        m = mask[i]
        if m.sum() >= 20:
            out[i] = (xs[0] * m).sum() / m.sum()
    return out


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    print(f"=== Checkpoints to evaluate: {len(args.checkpoints)} ===", flush=True)
    for c in args.checkpoints:
        print(f"    {os.path.basename(c)}", flush=True)

    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)
    NUM_FRAME_PER_BLOCK = config.num_frame_per_block

    from model import CameraCausalDiffusion  # noqa: E402

    print("=== Loading base model (ONCE; checkpoints are swapped in place) ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    print(f"=== base checkpoint: {args.base_checkpoint} ===", flush=True)
    base = torch.load(args.base_checkpoint, map_location="cpu")
    gen_sd = base.get("generator_ema", base.get("generator"))
    try:
        model.generator.load_state_dict(gen_sd)
    except RuntimeError:
        model.generator.load_state_dict(
            {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()},
            strict=False)
    model.generator.to(device=device, dtype=torch.bfloat16)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)
    model.vae.to(device=device, dtype=torch.bfloat16)

    rank = torch.load(args.checkpoints[0], map_location="cpu")["args"]["rank"]
    from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)

    print("=== Text encoding (fixed prompt, once) ===", flush=True)
    conditional_dict = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    act_shape = get_array_shape_from_lmdb(env, "actions")
    NUM_FRAMES = lat_shape[1]
    n_ctx = NUM_FRAME_PER_BLOCK
    n_total = min(args.n_scenes, lat_shape[0])
    print(f"=== Test split: using {n_total} of {lat_shape[0]} held-out scenes | "
          f"frames 0..{n_ctx-1} = context, {n_ctx}..{NUM_FRAMES-1} generated ===", flush=True)

    if args.dmd_schedule:
        # reproduce causal_forcing_dmd_camera.yaml: denoising_step_list is a list of INDICES
        # warped through the full 1000-step (shifted) schedule, not raw timestep values
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        schedule = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (schedule.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = schedule.cpu()
        print(f"=== DMD 4-step schedule: {[round(float(t),1) for t in schedule]} ===", flush=True)
    else:
        model.scheduler.set_timesteps(args.n_steps)
        schedule = model.scheduler.timesteps.to(device)

    psnr_m = ssim_m = fid_m = None
    try:
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        psnr_m = PeakSignalNoiseRatio(data_range=255.0).to(device)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)
    except Exception as e:
        print(f"!! PSNR/SSIM unavailable: {e}", flush=True)
    if not args.skip_fid:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            print("=== FID ready (relative measure only at this sample count) ===", flush=True)
        except Exception as e:
            print(f"!! FID unavailable (likely no inception weights behind the proxy): {e}", flush=True)

    def decode(latent):
        x = model.vae.decode_to_pixel(latent.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    def build_actions(raw, override):
        """raw: (B,30,4). override: None or (dim0, dim1). -> (B, F, 16)"""
        a = raw.copy()
        if override is not None:
            a[:, :, 0] = override[0]
            a[:, :, 1] = override[1]
        out = np.zeros((a.shape[0], NUM_FRAMES, 16), dtype=np.float32)
        for i in range(1, NUM_FRAMES):
            chunk = a[:, 4 * (i - 1):4 * i, :].reshape(a.shape[0], -1)
            out[:, i, :chunk.shape[1]] = chunk
        return out

    def sample(real_latent, action_embed, noise):
        """x0-prediction sampler (validated); only the generated block evolves."""
        B = real_latent.shape[0]
        s = real_latent.clone()
        s[:, n_ctx:] = noise[:, n_ctx:]
        for i, t_val in enumerate(schedule):
            timestep = torch.zeros((B, NUM_FRAMES), device=device, dtype=torch.bfloat16)
            timestep[:, n_ctx:] = t_val.item()
            s[:, :n_ctx] = real_latent[:, :n_ctx]
            _, x0_pred = model.generator(
                noisy_image_or_video=s, conditional_dict=cond_b, timestep=timestep,
                clean_x=real_latent, aug_t=None, viewmats=vm, Ks=ks, action_embed=action_embed)
            x0 = x0_pred.float().clamp(-6, 6)
            if i == len(schedule) - 1:
                s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        s[:, :n_ctx] = real_latent[:, :n_ctx]
        return s

    all_results = {}
    for ckpt_path in args.checkpoints:
        step = int(os.path.basename(ckpt_path).split("_")[-1].split(".")[0])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
        a_mean = ckpt["action_mean"].to(device)
        a_std = ckpt["action_std"].to(device)
        print(f"\n########## checkpoint step {step} ##########", flush=True)
        t0 = time.time()

        psnr_vals, ssim_vals, div_vals, ctx_div = [], [], [], []
        rel_correct = rel_total = abs_correct = abs_total = 0
        if psnr_m: psnr_m.reset()
        if ssim_m: ssim_m.reset()
        if fid_m: fid_m.reset()

        for start in range(0, n_total, args.batch_size):
            end = min(start + args.batch_size, n_total)
            B = end - start
            lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:])
                            for i in range(start, end)])
            act = np.stack([retrieve_row_from_lmdb(env, "actions", np.float32, i, shape=act_shape[1:])
                            for i in range(start, end)])
            real_latent = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)

            cond_b = {k: (v.repeat(B, *([1] * (v.dim() - 1))) if torch.is_tensor(v) and v.shape[0] == 1 else v)
                      for k, v in conditional_dict.items()}
            vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, NUM_FRAMES, 1, 1)
            ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
                .view(1, 1, 3, 3).repeat(B, NUM_FRAMES, 1, 1)

            def embed(override):
                a = torch.tensor(build_actions(act, override), device=device)
                an = (a - a_mean) / a_std
                an[:, 0, :] = 0.0
                assert an.abs().max().item() < SANITY_MAX_SIGMA * 3, "action out of distribution"
                return action_encoder(an.to(torch.bfloat16))

            noise = torch.randn_like(real_latent)   # SAME noise for all variants -> fair comparison

            # ---- MODE A: real action, compare against ground truth ----
            gen_real = sample(real_latent, embed(None), noise)
            f_gen = decode(gen_real)
            f_gt = decode(real_latent)
            n_ctx_px = 1 + 4 * (n_ctx - 1)          # latent 0 -> 1 px frame, then 4 each
            g = torch.tensor(f_gen[:, n_ctx_px:].copy()).float().permute(0, 1, 4, 2, 3).flatten(0, 1).to(device)
            t = torch.tensor(f_gt[:, n_ctx_px:].copy()).float().permute(0, 1, 4, 2, 3).flatten(0, 1).to(device)
            if psnr_m: psnr_m.update(g, t); psnr_vals.append(float(psnr_m.compute()))
            if ssim_m: ssim_m.update(g, t); ssim_vals.append(float(ssim_m.compute()))
            if fid_m:
                fid_m.update(t.to(torch.uint8), real=True)
                fid_m.update(g.to(torch.uint8), real=False)

            # ---- MODE B: swapped actions, controllability ----
            gen_r = sample(real_latent, embed((-DISP_MAX, 0.0)), noise)   # dim0<0 -> arm RIGHT
            gen_l = sample(real_latent, embed((+DISP_MAX, 0.0)), noise)   # dim0>0 -> arm LEFT
            fr, fl = decode(gen_r), decode(gen_l)
            div_vals.append(float(np.abs(fr[:, n_ctx_px:].astype(np.float32) -
                                          fl[:, n_ctx_px:].astype(np.float32)).mean()))
            ctx_div.append(float(np.abs(fr[:, :n_ctx_px].astype(np.float32) -
                                         fl[:, :n_ctx_px].astype(np.float32)).mean()))

            for b in range(B):
                xr, xl = arm_centroid_x(fr[b]), arm_centroid_x(fl[b])
                if np.isnan(xr[n_ctx_px - 1]) or np.isnan(xr[-1]) or np.isnan(xl[-1]):
                    continue
                drift_r = xr[-1] - xr[n_ctx_px - 1]     # expect POSITIVE (moves right)
                drift_l = xl[-1] - xl[n_ctx_px - 1]     # expect NEGATIVE (moves left)
                # relative test (robust: the arm has physical limits, so absolute
                # motion can be blocked, but the two variants should still separate)
                rel_total += 1
                rel_correct += int(drift_r > drift_l)
                # absolute test (stricter: each variant must move its own way)
                abs_total += 2
                abs_correct += int(drift_r > 0) + int(drift_l < 0)
            print(f"    scenes {start}-{end}: div={div_vals[-1]:.2f} ctx={ctx_div[-1]:.2f}", flush=True)

        res = {
            "step": step,
            "n_scenes": n_total,
            "psnr": psnr_vals[-1] if psnr_vals else None,
            "ssim": ssim_vals[-1] if ssim_vals else None,
            "fid": float(fid_m.compute()) if fid_m else None,
            "divergence_generated": float(np.mean(div_vals)),
            "divergence_context": float(np.mean(ctx_div)),
            "direction_acc_relative": (rel_correct / rel_total) if rel_total else None,
            "direction_acc_absolute": (abs_correct / abs_total) if abs_total else None,
            "direction_n_scenes": rel_total,
            "seconds": time.time() - t0,
        }
        all_results[step] = res
        print(f"  >>> step {step}: PSNR={res['psnr']} SSIM={res['ssim']} FID={res['fid']}", flush=True)
        print(f"  >>> divergence gen={res['divergence_generated']:.2f} "
              f"ctx={res['divergence_context']:.2f} | dir_rel={res['direction_acc_relative']} "
              f"dir_abs={res['direction_acc_absolute']} (n={res['direction_n_scenes']})", flush=True)
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n=== saved {args.out_json} ===", flush=True)
    print("=== SUMMARY (controllability vs training steps) ===", flush=True)
    print(f"{'step':>6} {'PSNR':>7} {'SSIM':>7} {'FID':>8} {'div_gen':>8} {'div_ctx':>8} {'dir_rel':>8} {'dir_abs':>8}", flush=True)
    for step in sorted(all_results):
        r = all_results[step]
        def fmt(v, w, p=2): return f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'n/a':>{w}}"
        print(f"{step:>6} {fmt(r['psnr'],7)} {fmt(r['ssim'],7,4)} {fmt(r['fid'],8,1)} "
              f"{fmt(r['divergence_generated'],8)} {fmt(r['divergence_context'],8)} "
              f"{fmt(r['direction_acc_relative'],8,3)} {fmt(r['direction_acc_absolute'],8,3)}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
