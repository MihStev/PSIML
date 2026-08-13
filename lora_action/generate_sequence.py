#!/usr/bin/env python
"""
Action PROGRAMS: a different action per latent frame within one clip.

This is the demo that justifies the architecture. Our conditioning is per
latent frame (see CLAUDE.md, "ARHITEKTONSKA ODLUKA"), so we can command
"go up for a while, then go down". The pooled-action design we rejected
could not express this even in principle -- the mean of up-then-down is
zero, which is exactly why pooling was disqualifying.

The script does not just render a video: it tracks the gripper centroid
through the sequence and reports the per-segment drift, so the reversal is
MEASURED, not eyeballed.

Sign convention (verified against raw BAIR pixels, no model involved):
    dim0 < 0 -> RIGHT in image      dim0 > 0 -> LEFT
    dim1 < 0 -> UP                  dim1 > 0 -> DOWN

Usage:
    python generate_sequence.py --program up_then_down
    python generate_sequence.py --program all --context_idx 100
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

D = 0.07  # hard range of the real displacement dims
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0), "still": (0.0, 0.0)}

# one entry per latent frame 1..7 (frame 0 never carries an action -- it has no predecessor)
PROGRAMS = {
    "up_then_down":    ["up"] * 4 + ["down"] * 3,
    "down_then_up":    ["down"] * 4 + ["up"] * 3,
    "left_then_right": ["left"] * 4 + ["right"] * 3,
    "right_then_left": ["right"] * 4 + ["left"] * 3,
    "constant_up":     ["up"] * 7,          # control: no reversal
    "square":          ["right"] * 2 + ["down"] * 2 + ["left"] * 2 + ["up"] * 1,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test", help="held-out split")
    p.add_argument("--context_idx", type=int, default=3)
    p.add_argument("--program", default="all")
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--real_dims23", action="store_true",
                    help="use the episode's real dims 2/3 instead of dataset-mean constants;\n                          the eval used real values, the first demo run did not -- this isolates\n                          whether freezing them weakened the conditioning")
    p.add_argument("--out_dir", default="/home/mls10/logs/gen_sequences")
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

    programs = list(PROGRAMS) if args.program == "all" else [args.program]
    print(f"=== Programs: {programs} ===", flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion  # noqa: E402
    print("=== Loading model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load("/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt", map_location="cpu")
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
    NUM_FRAMES = lat_shape[1]
    n_ctx = config.num_frame_per_block
    real_latent = torch.from_numpy(
        retrieve_row_from_lmdb(env, "latents", np.float16, args.context_idx,
                               shape=lat_shape[1:]).astype(np.float32)
    ).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    act_shape = get_array_shape_from_lmdb(env, "actions")
    real_act = retrieve_row_from_lmdb(env, "actions", np.float32, args.context_idx,
                                       shape=act_shape[1:])   # (30,4)

    vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, NUM_FRAMES, 1, 1)
    ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(1, NUM_FRAMES, 1, 1)
    model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)
    noise = torch.randn_like(real_latent)  # SAME noise for every program -> differences are the action

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    rows, summaries = [("REAL (ground truth)", decode(real_latent))], []
    for prog_name in programs:
        prog = PROGRAMS[prog_name]
        assert len(prog) == NUM_FRAMES - 1, f"program must have {NUM_FRAMES-1} entries"

        # per-latent-frame action: frame 0 gets nothing, frames 1..7 follow the program
        apl = np.zeros((NUM_FRAMES, 16), dtype=np.float32)
        for i, name in enumerate(prog, start=1):
            dx, dy = DIRS[name]
            if args.real_dims23:
                # keep the episode's own dims 2/3, override only the displacement dims
                chunk = real_act[4 * (i - 1):4 * i].copy()
                chunk[:, 0], chunk[:, 1] = dx, dy
                apl[i] = chunk.flatten()
            else:
                apl[i] = np.tile([dx, dy, 0.5, 0.25], 4)   # dims 2/3 at their dataset means
        a = torch.tensor(apl, device=device).unsqueeze(0)
        an = (a - a_mean) / a_std
        an[:, 0, :] = 0.0
        action_embed = action_encoder(an.to(torch.bfloat16))

        s = real_latent.clone()
        s[:, n_ctx:] = noise[:, n_ctx:]
        for i, t_val in enumerate(schedule):
            timestep = torch.zeros((1, NUM_FRAMES), device=device, dtype=torch.bfloat16)
            timestep[:, n_ctx:] = t_val.item()
            s[:, :n_ctx] = real_latent[:, :n_ctx]
            _, x0 = model.generator(noisy_image_or_video=s, conditional_dict=cond, timestep=timestep,
                                    clean_x=real_latent, aug_t=None, viewmats=vm, Ks=ks,
                                    action_embed=action_embed)
            x0 = x0.float().clamp(-6, 6)
            if i == len(schedule) - 1:
                s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        s[:, :n_ctx] = real_latent[:, :n_ctx]
        frames = decode(s)
        rows.append((f"PROGRAM {prog_name}", frames))

        # ---- MEASURE the trajectory: did the commanded reversal actually happen? ----
        traj = np.array([arm_xy(f) for f in frames])          # (F, 2)
        n_ctx_px = 1 + 4 * (n_ctx - 1)                         # 13
        gen = traj[n_ctx_px - 1:]                              # from last context frame onward
        half = len(gen) // 2
        seg1 = gen[half] - gen[0]                              # first half drift
        seg2 = gen[-1] - gen[half]                             # second half drift
        summaries.append((prog_name, prog[0], prog[-1], seg1, seg2))
        print(f"[{prog_name}] 1st half dx={seg1[0]:+.2f} dy={seg1[1]:+.2f} | "
              f"2nd half dx={seg2[0]:+.2f} dy={seg2[1]:+.2f}", flush=True)

    n = min(len(f) for _, f in rows)
    grid = np.concatenate([np.concatenate(list(f[:n]), axis=1) for _, f in rows], axis=0)
    big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST))
    png = os.path.join(args.out_dir, f"sequences_idx{args.context_idx}.png")
    Image.fromarray(big).save(png)
    print(f"\n=== saved {png} ===", flush=True)
    print("    rows: " + " | ".join(nm for nm, _ in rows), flush=True)

    # NOTE: an earlier version only checked whether the sign FLIPPED, not whether each
    # half moved the way it was told. That reported 5/6 "reversal seen" when in fact
    # most halves moved the wrong way. This checks the commanded direction per half.
    want = {"up": (1, -1), "down": (1, +1), "right": (0, +1), "left": (0, -1), "still": None}
    print("\n=== DID EACH HALF MOVE AS COMMANDED? (drift per half, pixels) ===", flush=True)
    print(f"{'program':>16} {'1st half':>18} {'2nd half':>18}  1st  2nd", flush=True)
    n_ok = n_tot = 0
    for name, first, last, s1, s2 in summaries:
        marks = []
        for cmd, seg in ((first, s1), (last, s2)):
            w = want[cmd]
            if w is None:
                marks.append(" - "); continue
            axis, sign = w
            ok = (seg[axis] * sign) > 0
            n_tot += 1; n_ok += int(ok)
            marks.append(" OK" if ok else " X ")
        print(f"{name:>16}  ({s1[0]:+6.2f},{s1[1]:+6.2f})  ({s2[0]:+6.2f},{s2[1]:+6.2f}) {marks[0]} {marks[1]}", flush=True)
    print(f"\n  correct halves: {n_ok}/{n_tot} ({100*n_ok/max(n_tot,1):.0f}%)", flush=True)

    try:
        import imageio
        for name, frames in rows[1:]:
            key = name.split()[-1]
            imageio.mimsave(os.path.join(args.out_dir, f"seq_{key}.mp4"), list(frames), fps=4, macro_block_size=1)
        print(f"=== mp4 files saved to {args.out_dir} ===", flush=True)
    except Exception as e:
        print(f"(mp4 skipped: {e})", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
