#!/usr/bin/env python
"""Why does control fail in free rollout: does the model stop listening, or mis-aim?

Scheduled sampling halved the picture's decay but left direction accuracy untouched,
which means the two failure modes are separable and the loss of control has some other
cause. This separates two candidate causes.

At each rollout depth, from the SAME self-generated context, we generate two futures with
OPPOSITE actions and measure how far apart they land:

  - divergence STAYS HIGH but direction accuracy falls
        -> the model still obeys; the scene has drifted, so the commanded push lands
           relative to an arm that is no longer where the detector thinks it is.
           Control intact, PERCEPTION broken.

  - divergence COLLAPSES
        -> the model literally stops responding to the action. Control broken.

Divergence alone cannot be read raw: the sampler draws fresh noise at each intermediate
re-noising step, so two runs differ even under identical conditioning (measured: 15.35 at
depth 0 on the main model). That floor may GROW with depth as the context degrades, so we
re-measure it at EVERY depth by running the same action twice, and report the ratio.

Usage:
    python control_diagnosis.py --checkpoint <ckpt> --n_scenes 64 --max_depth 3
"""
import argparse
import json
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

import lmdb                                     # noqa: E402
import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
from omegaconf import OmegaConf                 # noqa: E402

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb   # noqa: E402
from train_lora_action import ActionEncoderV2                                   # noqa: E402

D = 0.07
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0)}
NAMES = list(DIRS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--n_scenes", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--dmd_schedule", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="/home/mls10/logs/control_diagnosis.json")
    return p.parse_args()


def arm_x(frames):
    """Horizontal centroid of the gripper, per frame. NaN where it is not found."""
    f = frames.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = (r > g + 25) & (r > b + 25) & (r < 170)
    out = np.full(f.shape[0], np.nan)
    for i in range(f.shape[0]):
        if m[i].sum() >= 20:
            out[i] = np.nonzero(m[i])[1].mean()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion                     # noqa: E402
    print("=== loading model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
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

    from peft import LoraConfig, inject_adapter_in_model        # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    enc = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    enc.load_state_dict(ckpt["action_encoder_state_dict"])
    a_mean, a_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    F, n_ctx = lat_shape[1], config.num_frame_per_block
    assert F == 2 * n_ctx, f"expects one context block + one generated block (F={F})"
    n_total = min(args.n_scenes, lat_shape[0])

    if args.dmd_schedule:
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        sch = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (sch.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = sch.cpu()
    else:
        model.scheduler.set_timesteps(args.n_steps)
        sch = model.scheduler.timesteps.to(device)
    print(f"=== {len(sch)} denoising steps, {n_total} scenes, depths 1..{args.max_depth} ===",
          flush=True)

    cond_1 = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])

    def embed(B, dx, dy):
        a = np.zeros((B, F, 16), dtype=np.float32)
        for i in range(1, F):
            a[:, i] = np.tile([dx, dy, 0.5, 0.25], 4)
        an = (torch.tensor(a, device=device) - a_mean) / a_std
        an[:, 0, :] = 0.0
        return enc(an.to(torch.bfloat16))

    def step(context, ae):
        """One block from `context` (B, n_ctx, ...) -> new block (B, n_ctx, ...)."""
        B = context.shape[0]
        vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, F, 1, 1)
        ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device,
                          dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(B, F, 1, 1)
        cond = {k: (v.repeat(B, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
                for k, v in cond_1.items()}
        win = torch.cat([context, torch.randn_like(context)], dim=1)
        clean = torch.cat([context, context], dim=1)
        for i, t_val in enumerate(sch):
            ts = torch.zeros((B, F), device=device, dtype=torch.bfloat16)
            ts[:, n_ctx:] = t_val.item()
            win[:, :n_ctx] = context
            _, x0 = model.generator(noisy_image_or_video=win, conditional_dict=cond, timestep=ts,
                                    clean_x=clean, aug_t=None, viewmats=vm, Ks=ks, action_embed=ae)
            x0 = x0.float().clamp(-6, 6)
            if i == len(sch) - 1:
                win[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                win[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        return win[:, n_ctx:].clone()

    def decode(block):
        x = model.vae.decode_to_pixel(block.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    acc = {d: {"div": [], "floor": [], "rel_ok": 0, "rel_n": 0} for d in range(1, args.max_depth + 1)}

    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        B = end - start
        lat = torch.from_numpy(np.stack([
            retrieve_row_from_lmdb(env, "latents", np.float16, i, shape=lat_shape[1:]).astype(np.float32)
            for i in range(start, end)])).to(device=device, dtype=torch.bfloat16)
        context = lat[:, :n_ctx].clone()

        for d in range(1, args.max_depth + 1):
            # --- probe THIS context with opposite actions, and with a repeated action ---
            gen_r = step(context, embed(B, -D, 0.0))     # dim0 < 0 -> arm RIGHT
            gen_l = step(context, embed(B, +D, 0.0))     # dim0 > 0 -> arm LEFT
            gen_r2 = step(context, embed(B, -D, 0.0))    # SAME action again -> noise floor
            fr, fl, fr2 = decode(gen_r), decode(gen_l), decode(gen_r2)
            acc[d]["div"].append(float(np.abs(fr.astype(np.float32) - fl.astype(np.float32)).mean()))
            acc[d]["floor"].append(float(np.abs(fr.astype(np.float32) - fr2.astype(np.float32)).mean()))

            ctx_px = decode(context)
            for b in range(B):
                x0 = arm_x(ctx_px[b])
                xr, xl = arm_x(fr[b]), arm_x(fl[b])
                if np.isnan(x0[-1]) or np.isnan(xr[-1]) or np.isnan(xl[-1]):
                    acc[d]["rel_n"] += 1          # untracked counts as FAILURE, not dropped
                    continue
                acc[d]["rel_n"] += 1
                acc[d]["rel_ok"] += int((xr[-1] - x0[-1]) > (xl[-1] - x0[-1]))

            # --- advance the rollout with a random action, so depth d+1 sees a drifted context ---
            name = NAMES[np.random.randint(len(NAMES))]
            dx, dy = DIRS[name]
            context = step(context, embed(B, dx, dy))

        print(f"  scenes {start}-{end} done", flush=True)

    print(f"\n{'depth':>6} {'divergence':>11} {'noise floor':>12} {'ratio':>7} {'dir_rel':>9} {'n':>5}",
          flush=True)
    out = {}
    for d in range(1, args.max_depth + 1):
        div = float(np.mean(acc[d]["div"]))
        flo = float(np.mean(acc[d]["floor"]))
        rel = acc[d]["rel_ok"] / max(acc[d]["rel_n"], 1)
        out[d] = {"divergence": div, "noise_floor": flo, "ratio": div / max(flo, 1e-9),
                  "direction_rel": rel, "n": acc[d]["rel_n"]}
        print(f"{d:>6} {div:>11.2f} {flo:>12.2f} {div/max(flo,1e-9):>7.2f} {rel:>9.3f} "
              f"{acc[d]['rel_n']:>5}", flush=True)
    json.dump({"checkpoint": args.checkpoint, "n_scenes": n_total, "results": out},
              open(args.out_json, "w"), indent=2)
    print(f"\n  -> {args.out_json}", flush=True)
    print("  ratio ~1 means the action changes nothing beyond sampler noise;", flush=True)
    print("  ratio staying high while dir_rel falls means it obeys but mis-aims.", flush=True)


if __name__ == "__main__":
    main()
