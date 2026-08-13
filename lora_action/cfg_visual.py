#!/usr/bin/env python
"""
Visual companion to cfg_test.py: same scene generated at several guidance
scales, stacked for eye comparison.

Crucially it also stacks the two references that decompose where the blur
actually comes from:

    row 1  RAW PIXELS          -- the original 64x64 frames, no VAE at all
    row 2  VAE ROUND-TRIP      -- the REAL latent decoded back. Everything below
                                  this row is blur the VAE imposes; no
                                  generation can beat it (measured: 22.74 dB)
    row 3+ GENERATED at w=...  -- our output at each guidance scale

So if row 2 already looks soft, the softness is the autoencoder, not the model.
That distinction is the whole point of the comparison.

Usage:
    python cfg_visual.py --scales 1.0 2.0 3.0 --context_idx 3
"""
import argparse
import os
import sys

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
from PIL import Image

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--raw_npz", default="/tmp/bair_raw/test/shard_00000.npz")
    p.add_argument("--context_idx", type=int, default=3)
    p.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--zoom", type=int, default=5, help="upscale factor for viewing")
    p.add_argument("--out_dir", default="/home/mls10/logs/cfg_visual")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

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
    NULL = ckpt["null_action_embedding"].to(device=device, dtype=torch.bfloat16)

    cond = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    act_shape = get_array_shape_from_lmdb(env, "actions")
    F = lat_shape[1]
    n_ctx = config.num_frame_per_block
    n_ctx_px = 1 + 4 * (n_ctx - 1)

    lat = retrieve_row_from_lmdb(env, "latents", np.float16, args.context_idx, shape=lat_shape[1:])
    act = retrieve_row_from_lmdb(env, "actions", np.float32, args.context_idx, shape=act_shape[1:])
    real = torch.from_numpy(lat.astype(np.float32)).to(device=device, dtype=torch.bfloat16).unsqueeze(0)

    vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, F, 1, 1)
    ks = torch.tensor([[.5, 0, .5], [0, .5, .5], [0, 0, 1]], device=device,
                      dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(1, F, 1, 1)
    model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)
    null_emb = NULL.view(1, 1, -1).expand(1, F, -1).contiguous()

    def decode(l):
        x = model.vae.decode_to_pixel(l.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    apl = np.zeros((F, 16), dtype=np.float32)
    for i in range(1, F):
        apl[i] = act[4 * (i - 1):4 * i].flatten()
    an = (torch.tensor(apl, device=device).unsqueeze(0) - a_mean) / a_std
    an[:, 0, :] = 0.0
    ae = action_encoder(an.to(torch.bfloat16))

    noise = torch.randn_like(real)          # identical noise for every scale

    rows = []
    raw = np.load(args.raw_npz)["images"][args.context_idx]      # (30,64,64,3) uint8
    rows.append(("RAW PIXELS (no VAE)", raw[:29]))
    rows.append(("VAE round-trip of REAL latent (the ceiling, 22.74 dB)", decode(real)))

    for w in args.scales:
        s = real.clone()
        s[:, n_ctx:] = noise[:, n_ctx:]
        for i, t_val in enumerate(schedule):
            ts = torch.zeros((1, F), device=device, dtype=torch.bfloat16)
            ts[:, n_ctx:] = t_val.item()
            s[:, :n_ctx] = real[:, :n_ctx]
            _, x0c = model.generator(noisy_image_or_video=s, conditional_dict=cond, timestep=ts,
                                     clean_x=real, aug_t=None, viewmats=vm, Ks=ks, action_embed=ae)
            if w != 1.0:
                _, x0n = model.generator(noisy_image_or_video=s, conditional_dict=cond, timestep=ts,
                                         clean_x=real, aug_t=None, viewmats=vm, Ks=ks, action_embed=null_emb)
                x0 = x0n.float() + w * (x0c.float() - x0n.float())
            else:
                x0 = x0c.float()
            x0 = x0.clamp(-6, 6)
            if i == len(schedule) - 1:
                s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        s[:, :n_ctx] = real[:, :n_ctx]
        f = decode(s)
        rows.append((f"GENERATED  w={w}", f))
        # sharpness proxy: gradient energy on the generated part only
        gp = f[n_ctx_px:].astype(np.float32).mean(axis=-1)
        sharp = np.mean(np.abs(np.diff(gp, axis=1))) + np.mean(np.abs(np.diff(gp, axis=2)))
        print(f"  w={w}: sharpness (mean |gradient|) = {sharp:.3f}", flush=True)

    for name, f in rows[:2]:
        gp = f[n_ctx_px:].astype(np.float32).mean(axis=-1)
        sharp = np.mean(np.abs(np.diff(gp, axis=1))) + np.mean(np.abs(np.diff(gp, axis=2)))
        print(f"  {name}: sharpness = {sharp:.3f}", flush=True)

    n = min(len(f) for _, f in rows)
    grid = np.concatenate([np.concatenate(list(f[:n]), axis=1) for _, f in rows], axis=0)
    big = Image.fromarray(grid).resize((grid.shape[1] * args.zoom, grid.shape[0] * args.zoom), Image.NEAREST)
    png = os.path.join(args.out_dir, f"cfg_visual_idx{args.context_idx}.png")
    big.save(png)
    print(f"\n=== saved {png} ===", flush=True)
    print("    rows top->bottom: " + " | ".join(nm for nm, _ in rows), flush=True)

    # a zoomed crop of a few late frames, where differences are easiest to see
    crop_idx = [n - 6, n - 4, n - 2]
    crop = np.concatenate([np.concatenate([f[i] for i in crop_idx], axis=1) for _, f in rows], axis=0)
    Image.fromarray(crop).resize((crop.shape[1] * 8, crop.shape[0] * 8), Image.NEAREST).save(
        os.path.join(args.out_dir, f"cfg_zoom_idx{args.context_idx}.png"))
    print(f"=== saved zoomed crop (8x) for close inspection ===", flush=True)

    try:
        import imageio
        for name, f in rows:
            key = name.split()[0].lower().replace("(", "").replace(")", "")
            if "w=" in name:
                key = "w" + name.split("w=")[1]
            imageio.mimsave(os.path.join(args.out_dir, f"cfg_{key}_idx{args.context_idx}.mp4"),
                            list(f), fps=4, macro_block_size=1)
        print(f"=== mp4 files saved ===", flush=True)
    except Exception as e:
        print(f"(mp4 skipped: {e})", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
