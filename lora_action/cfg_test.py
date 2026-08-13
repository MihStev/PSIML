#!/usr/bin/env python
"""
Action classifier-free guidance (CFG) sweep.

We trained a null action embedding via action dropout (p=0.1) precisely so that
CFG would be available at inference, and then never used it. CFG amplifies the
effect of the conditioning:

    x0_guided = x0_null + w * (x0_action - x0_null)

with w = 1.0 reproducing ordinary sampling. Higher w should sharpen the output
and strengthen action control; too high usually over-saturates and hurts
fidelity. This script finds where that trade-off sits for us.

Cost note: CFG needs TWO forwards per sampling step (conditional and null), so
each scale is ~2x an ordinary generation.

Reports both metric families, since they can move in opposite directions:
  - fidelity  (PSNR/SSIM vs ground truth, using the episode's REAL action)
  - control   (action-swap divergence + direction accuracy, using +/-0.07)

Usage:
    python cfg_test.py --scales 1.0 1.5 2.0 3.0 --n_scenes 32
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
from omegaconf import OmegaConf

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402

D = 0.07


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--n_scenes", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--out_json", default="/home/mls10/logs/cfg_results.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


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
    print(f"=== checkpoint step {ckpt['step']} rank {rank} | scales {args.scales} ===", flush=True)

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
    NULL = ckpt["null_action_embedding"].to(device=device, dtype=torch.bfloat16)   # (1536,)
    print(f"=== null embedding loaded, norm={NULL.float().norm():.3f} ===", flush=True)

    cond = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    act_shape = get_array_shape_from_lmdb(env, "actions")
    F = lat_shape[1]
    n_ctx = config.num_frame_per_block
    n_ctx_px = 1 + 4 * (n_ctx - 1)
    n_total = min(args.n_scenes, lat_shape[0])
    print(f"=== {n_total} held-out scenes ===", flush=True)

    model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)

    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    psnr_m = PeakSignalNoiseRatio(data_range=255.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=255.0).to(device)

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    def build(raw, override):
        a = raw.copy()
        if override is not None:
            a[:, :, 0], a[:, :, 1] = override
        out = np.zeros((a.shape[0], F, 16), dtype=np.float32)
        for i in range(1, F):
            c = a[:, 4 * (i - 1):4 * i, :].reshape(a.shape[0], -1)
            out[:, i, :c.shape[1]] = c
        return out

    results = {}
    for w in args.scales:
        t0 = time.time()
        psnr_m.reset(); ssim_m.reset()
        divs, rel_ok, rel_n, abs_ok, abs_n = [], 0, 0, 0, 0

        for start in range(0, n_total, args.batch_size):
            end = min(start + args.batch_size, n_total)
            B = end - start
            lat = np.stack([retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:])
                            for i in range(start, end)])
            act = np.stack([retrieve_row_from_lmdb(env, "actions", np.float32, i, shape=act_shape[1:])
                            for i in range(start, end)])
            real = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
            cond_b = {k: (v.repeat(B, *([1] * (v.dim() - 1))) if torch.is_tensor(v) and v.shape[0] == 1 else v)
                      for k, v in cond.items()}
            vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, F, 1, 1)
            ks = torch.tensor([[.5, 0, .5], [0, .5, .5], [0, 0, 1]], device=device,
                              dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(B, F, 1, 1)
            null_emb = NULL.view(1, 1, -1).expand(B, F, -1).contiguous()

            def embed(override):
                a = torch.tensor(build(act, override), device=device)
                an = (a - a_mean) / a_std
                an[:, 0, :] = 0.0
                return action_encoder(an.to(torch.bfloat16))

            def sample(ae, noise):
                s = real.clone()
                s[:, n_ctx:] = noise[:, n_ctx:]
                for i, t_val in enumerate(schedule):
                    ts = torch.zeros((B, F), device=device, dtype=torch.bfloat16)
                    ts[:, n_ctx:] = t_val.item()
                    s[:, :n_ctx] = real[:, :n_ctx]
                    _, x0_c = model.generator(noisy_image_or_video=s, conditional_dict=cond_b, timestep=ts,
                                              clean_x=real, aug_t=None, viewmats=vm, Ks=ks, action_embed=ae)
                    if w != 1.0:
                        _, x0_n = model.generator(noisy_image_or_video=s, conditional_dict=cond_b, timestep=ts,
                                                  clean_x=real, aug_t=None, viewmats=vm, Ks=ks,
                                                  action_embed=null_emb)
                        x0 = x0_n.float() + w * (x0_c.float() - x0_n.float())
                    else:
                        x0 = x0_c.float()
                    x0 = x0.clamp(-6, 6)
                    if i == len(schedule) - 1:
                        s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
                    else:
                        sn = float(model.scheduler.sigmas[i + 1])
                        s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
                s[:, :n_ctx] = real[:, :n_ctx]
                return s

            noise = torch.randn_like(real)
            # fidelity, using the episode's own action
            g = decode(sample(embed(None), noise))
            t = decode(real)
            gg = torch.tensor(g[:, n_ctx_px:].copy()).float().permute(0, 1, 4, 2, 3).flatten(0, 1).to(device)
            tt = torch.tensor(t[:, n_ctx_px:].copy()).float().permute(0, 1, 4, 2, 3).flatten(0, 1).to(device)
            psnr_m.update(gg, tt); ssim_m.update(gg, tt)

            # control, using swapped actions
            fr = decode(sample(embed((-D, 0.0)), noise))
            fl = decode(sample(embed((+D, 0.0)), noise))
            divs.append(float(np.abs(fr[:, n_ctx_px:].astype(np.float32) -
                                      fl[:, n_ctx_px:].astype(np.float32)).mean()))
            for b in range(B):
                xr, xl = arm_centroid_x(fr[b]), arm_centroid_x(fl[b])
                if np.isnan(xr[n_ctx_px - 1]) or np.isnan(xr[-1]) or np.isnan(xl[-1]):
                    continue
                dr = xr[-1] - xr[n_ctx_px - 1]
                dl = xl[-1] - xl[n_ctx_px - 1]
                rel_n += 1; rel_ok += int(dr > dl)
                abs_n += 2; abs_ok += int(dr > 0) + int(dl < 0)
            print(f"  [w={w}] scenes {start}-{end} div={divs[-1]:.2f}", flush=True)

        results[w] = {
            "psnr": float(psnr_m.compute()), "ssim": float(ssim_m.compute()),
            "divergence": float(np.mean(divs)),
            "dir_rel": rel_ok / rel_n if rel_n else None,
            "dir_abs": abs_ok / abs_n if abs_n else None,
            "seconds": time.time() - t0,
        }
        r = results[w]
        print(f"  >>> w={w}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} div={r['divergence']:.2f} "
              f"dir_rel={r['dir_rel']:.3f} dir_abs={r['dir_abs']:.3f}  ({r['seconds']:.0f}s)", flush=True)
        with open(args.out_json, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print("\n=== CFG SWEEP (w=1.0 is ordinary sampling) ===", flush=True)
    print(f"{'w':>5} {'PSNR':>7} {'SSIM':>8} {'div':>8} {'dir_rel':>8} {'dir_abs':>8}", flush=True)
    for w in args.scales:
        r = results[w]
        print(f"{w:>5.1f} {r['psnr']:>7.2f} {r['ssim']:>8.4f} {r['divergence']:>8.2f} "
              f"{r['dir_rel']:>8.3f} {r['dir_abs']:>8.3f}", flush=True)
    print(f"\n(VAE ceiling for reference: 22.74 dB)", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
