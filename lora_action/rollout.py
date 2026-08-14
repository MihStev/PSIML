#!/usr/bin/env python
"""
Sliding-window autoregressive rollout: arbitrary length, one action per block.
This is the basis for an interactive (Genie-style) demo.

How it works, and why no KV-cache is needed:

    window 0:  context = REAL frames 0..3     -> generate 4..7  with action A
    window 1:  context = generated 4..7       -> generate 8..11 with action B
    window 2:  context = generated 8..11      -> generate ...   with action C

Each window is exactly the shape the model was trained on (4 clean context
frames + 4 noisy frames under the teacher-forcing mask), so we simply re-encode
the context each step. At our scale that is 128 tokens -- re-encoding is far
cheaper than the engineering cost of a KV-cache, and it keeps us inside the
training regime instead of inventing a new inference path.

HONEST LIMITATION -- error accumulation. The model was trained with teacher
forcing, i.e. it always saw REAL context. Here it consumes its OWN output, which
drifts from the training distribution. This is classic exposure bias, and it is
precisely why minWM has Stage 2/3 (self-forcing) after Stage 1. Our Stage-1
checkpoint is the one the mentor described as "safer, but won't do long
rollout". Expect short rollouts (2-4 blocks) to hold and longer ones to degrade
-- this script measures that decay rather than hiding it.

Timing note: ~24 sampling steps per block. A DMD (distilled) checkpoint needs
only 4, which is what would turn this from "press key, wait a few seconds" into
a genuinely interactive demo.

Usage:
    python rollout.py --actions up up down down right
    python rollout.py --actions right right right right right right   # decay test
"""
import argparse
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
from PIL import Image

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402

D = 0.07
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0), "still": (0.0, 0.0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--context_idx", type=int, default=3)
    p.add_argument("--actions", nargs="+", default=["up", "up", "down", "down"],
                    help="one action per generated block")
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--dmd_schedule", action="store_true",
                    help="use the distilled model's 4-step schedule (same code path as evaluate.py)")
    p.add_argument("--out_dir", default="/home/mls10/logs/rollout")
    return p.parse_args()


def arm_xy(frame):
    f = frame.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = (r > g + 25) & (r > b + 25) & (r < 170)
    if m.sum() < 20:
        return np.nan, np.nan
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    for a in args.actions:
        assert a in DIRS, f"unknown action {a}; choose from {list(DIRS)}"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion  # noqa: E402
    print("=== Loading model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load(args.base_checkpoint, map_location="cpu")
    gen_sd = base.get("generator_ema", base.get("generator"))
    try:
        model.generator.load_state_dict(gen_sd)
    except RuntimeError:
        model.generator.load_state_dict(
            {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}, strict=False)
    model.generator.to(device=device, dtype=torch.bfloat16)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)
    model.vae.to(device=device, dtype=torch.bfloat16)

    from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
    a_mean, a_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    cond = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    F = lat_shape[1]
    n_ctx = config.num_frame_per_block
    assert F == 2 * n_ctx, f"this rollout assumes one context block + one generated block (F={F})"

    real = torch.from_numpy(
        retrieve_row_from_lmdb(env, "latents", np.float16, args.context_idx,
                               shape=lat_shape[1:]).astype(np.float32)
    ).to(device=device, dtype=torch.bfloat16).unsqueeze(0)

    vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, F, 1, 1)
    ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(1, F, 1, 1)
    if args.dmd_schedule:
        # identical to evaluate.py: denoising_step_list are INDICES into the 1000-step
        # (shifted) schedule -> [1000.0, 937.5, 833.3, 625.0]
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        schedule = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (schedule.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = schedule.cpu()
        print(f"=== DMD 4-step schedule: {[round(float(t),1) for t in schedule]} ===", flush=True)
    else:
        model.scheduler.set_timesteps(args.n_steps)
        schedule = model.scheduler.timesteps.to(device)

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    def embed(prev_dir, cur_dir):
        """window frame 0 = zeros (as in training), 1..3 = previous action,
        4..7 = the action being commanded now."""
        apl = np.zeros((F, 16), dtype=np.float32)
        for i in range(1, n_ctx):
            dx, dy = DIRS[prev_dir]
            apl[i] = np.tile([dx, dy, 0.5, 0.25], 4)
        for i in range(n_ctx, F):
            dx, dy = DIRS[cur_dir]
            apl[i] = np.tile([dx, dy, 0.5, 0.25], 4)
        a = torch.tensor(apl, device=device).unsqueeze(0)
        an = (a - a_mean) / a_std
        an[:, 0, :] = 0.0
        assert an.abs().max().item() < 20, "action out of distribution"
        return action_encoder(an.to(torch.bfloat16))

    context = real[:, :n_ctx].clone()      # start from the REAL first block
    all_latents = [context.clone()]
    prev_dir = "still"
    per_block_time = []

    print(f"=== Rollout: {len(args.actions)} blocks, actions = {args.actions} ===", flush=True)
    for bi, cur_dir in enumerate(args.actions):
        t0 = time.time()
        window = torch.cat([context, torch.randn_like(context)], dim=1)   # (1, F, ...)
        ae = embed(prev_dir, cur_dir)
        for i, t_val in enumerate(schedule):
            timestep = torch.zeros((1, F), device=device, dtype=torch.bfloat16)
            timestep[:, n_ctx:] = t_val.item()
            window[:, :n_ctx] = context
            _, x0 = model.generator(noisy_image_or_video=window, conditional_dict=cond, timestep=timestep,
                                    clean_x=torch.cat([context, context], dim=1), aug_t=None,
                                    viewmats=vm, Ks=ks, action_embed=ae)
            x0 = x0.float().clamp(-6, 6)
            if i == len(schedule) - 1:
                window[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                window[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        new_block = window[:, n_ctx:].clone()
        all_latents.append(new_block)
        context = new_block                      # <-- the model now consumes its OWN output
        prev_dir = cur_dir
        dt = time.time() - t0
        per_block_time.append(dt)
        rng = new_block.float()
        print(f"  [block {bi+1}/{len(args.actions)}] action={cur_dir:>5} {dt:5.2f}s "
              f"latent range=({rng.min():+.2f},{rng.max():+.2f})", flush=True)

    full = torch.cat(all_latents, dim=1)
    frames = decode(full)
    print(f"\n=== decoded {len(frames)} frames from {full.shape[1]} latent frames ===", flush=True)

    # ---- measure per-block drift and latent-range drift (the decay signal) ----
    traj = np.array([arm_xy(f) for f in frames])
    px_per_block = 4 * n_ctx        # each block of 4 latent frames -> 16 pixel frames
    print("\n=== PER-BLOCK DRIFT (did each command land?) ===", flush=True)
    print(f"{'block':>6} {'action':>7} {'dx':>8} {'dy':>8} {'expected':>10} {'sec':>6}", flush=True)
    n_ctx_px = 1 + 4 * (n_ctx - 1)
    for bi, a in enumerate(args.actions):
        lo = n_ctx_px - 1 + bi * px_per_block
        hi = min(lo + px_per_block, len(traj) - 1)
        if lo >= len(traj) or np.isnan(traj[lo]).any() or np.isnan(traj[hi]).any():
            print(f"{bi+1:>6} {a:>7}   (arm not tracked)", flush=True)
            continue
        d = traj[hi] - traj[lo]
        exp = {"up": "dy<0", "down": "dy>0", "right": "dx>0", "left": "dx<0", "still": "-"}[a]
        print(f"{bi+1:>6} {a:>7} {d[0]:>+8.2f} {d[1]:>+8.2f} {exp:>10} {per_block_time[bi]:>6.2f}", flush=True)

    print(f"\n=== timing: {np.mean(per_block_time):.2f}s per block of {px_per_block} frames "
          f"({args.n_steps} sampling steps). A 4-step DMD model would be ~{args.n_steps/4:.0f}x faster. ===",
          flush=True)

    grid = np.concatenate(list(frames), axis=1)
    big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST))
    png = os.path.join(args.out_dir, f"rollout_{'_'.join(args.actions)}.png")
    Image.fromarray(big).save(png)
    print(f"=== saved {png} ===", flush=True)
    try:
        import imageio
        mp4 = os.path.join(args.out_dir, f"rollout_{'_'.join(args.actions)}.mp4")
        imageio.mimsave(mp4, list(frames), fps=4, macro_block_size=1)
        print(f"=== saved {mp4} ===", flush=True)
    except Exception as e:
        print(f"(mp4 skipped: {e})", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
