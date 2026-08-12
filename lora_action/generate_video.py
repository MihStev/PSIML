#!/usr/bin/env python
"""
Generate future frames from a real BAIR context, with a CHOSEN action.

=== V2 -- rewritten after diagnosing why V1 produced green mush ===

Two real bugs were found in V1 (both here, NOT in training):

BUG 1 -- ACTION SCALE (the fatal one). V1's hand-written "canonical" actions used
values like [3.0, 0, 0, 0]. Real BAIR action dims 0/1 have std=0.0405 and range
+/-0.07. After the training-set normalization, 3.0 became a **74-sigma** input.
The ActionEncoder then emitted an embedding with norm ~218 vs ~3.5 for real
actions (62x too large). That embedding is ADDED to the per-frame time embedding
before time_projection, i.e. it becomes the AdaLN shift/scale/gate for all 30 DiT
blocks -- so a 62x oversized vector destroys the modulation of the entire network.
Hence: green mush, and latents exploding monotonically under ODE integration.

BUG 2 -- BROKEN BLOCK STRUCTURE. Training always assigns ONE shared timestep per
block of num_frame_per_block(=4) frames (see BaseModel._get_timestep). V1 set
frame 0 to t=0 while frames 1..3 (same block!) were at t=1000 -- a combination the
model never saw. Also, the teacher-forcing mask (_prepare_teacher_forcing_mask)
gives noisy block i access to clean frames of PREVIOUS blocks only
(noise_context_ends = block_index * attention_block_size), so "1 context frame"
was never a meaningful unit anyway.

V2 therefore generates at BLOCK granularity, exactly matching the training regime:
    frames 0-3 (block 0): real context, clean, timestep 0
    frames 4-7 (block 1): generated from pure noise, one shared timestep
and builds chosen actions by taking a REAL action sequence and overriding only
dims 0/1 (the displacement dims) with values inside the real +/-0.07 range.

It generates several variants in ONE model load so they are directly comparable:
real action, pushed-right, pushed-left -- which doubles as a first look at whether
the action actually controls the output (the real controllability eval is still a
separate, later task).
"""
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

import argparse

import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402
from train_lora_action import ActionEncoderV2  # noqa: E402

# measured from the training set (see diagnosis): dims 0/1 are displacements with
# std ~0.040 and hard range +/-0.07; dims 2/3 are a different quantity (range 0..4)
# and are left untouched at their real values.
DISP_MAX = 0.07
# SIGN CONVENTION -- verified empirically against raw BAIR pixels (no model involved):
# episodes with cumulative dim0 > 0 move the gripper LEFT in the image (-8.5 px mean),
# dim0 < 0 move it RIGHT (+13.2 px mean). So the robot frame's +x is image-left.
# The first labelling here was inverted; the model had learned the correct physics all
# along -- the user spotted that "right" videos drifted left, which is what led to this
# check. Note the real data is itself asymmetric (13.2 px vs 8.5 px), and the model
# reproduces that asymmetry.
ACTION_OVERRIDES = {
    "real":  None,                      # use the episode's own actions unchanged
    "right": (-DISP_MAX, 0.0),          # (dim0, dim1) forced, dims 2/3 kept real
    "left":  (+DISP_MAX, 0.0),
    "up":    (0.0, -DISP_MAX),          # verified: dim1 > 0 moves the gripper DOWN
    "down":  (0.0, +DISP_MAX),          # (+7.7 px) and dim1 < 0 moves it UP (-3.5 px)
}

SANITY_MAX_SIGMA = 6.0  # refuse to run if a normalized action exceeds this


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora/step_2500.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/train")
    p.add_argument("--context_idx", type=int, default=100)
    p.add_argument("--variants", nargs="+", default=["real", "right", "left"])
    p.add_argument("--n_steps", type=int, default=24, help="sampling steps")
    p.add_argument("--sampler", default="x0", choices=["x0", "euler"],
                    help="x0 = predict-x0-then-renoise (re-projects onto the data manifold every\n                          step, robust to imperfect v); euler = raw ODE on the flow prediction")
    p.add_argument("--x0_clamp", type=float, default=6.0,
                    help="clamp x0_pred to +/-this (real latents measured at ~+/-4); 0 disables")
    p.add_argument("--out_dir", default="/home/mls10/logs/generated_videos_v2")
    return p.parse_args()


def build_actions(raw_actions_30x4, n_latent, override):
    """Align raw actions to latent frames exactly as in training (bair_dataset.py),
    optionally overriding the two displacement dims with a chosen constant."""
    a = raw_actions_30x4.copy()
    if override is not None:
        a[:, 0] = override[0]
        a[:, 1] = override[1]
    out = np.zeros((n_latent, 16), dtype=np.float32)
    for i in range(1, n_latent):
        chunk = a[4 * (i - 1):4 * i].flatten()
        out[i, :len(chunk)] = chunk
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"=== Loading checkpoint {args.checkpoint} ===", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    print(f"checkpoint step={ckpt['step']} rank={rank}", flush=True)

    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)
    NUM_FRAME_PER_BLOCK = config.num_frame_per_block

    from model import CameraCausalDiffusion  # noqa: E402

    print("=== Constructing model + loading BASE checkpoint ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load("/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt", map_location="cpu")
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

    print(f"=== Injecting LoRA (rank={rank}) + trained weights ===", flush=True)
    from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
    action_mean = ckpt["action_mean"].to(device)
    action_std = ckpt["action_std"].to(device)

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    act_shape = get_array_shape_from_lmdb(env, "actions")
    NUM_FRAMES = lat_shape[1]
    n_ctx = NUM_FRAME_PER_BLOCK  # block 0 is the real context; blocks 1.. are generated

    real_latent_np = retrieve_row_from_lmdb(env, "latents", np.float16, args.context_idx, shape=lat_shape[1:])
    real_actions = retrieve_row_from_lmdb(env, "actions", np.float32, args.context_idx, shape=act_shape[1:])
    real_latent = torch.from_numpy(real_latent_np.astype(np.float32)).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    print(f"=== Scene: LMDB idx={args.context_idx} | {NUM_FRAMES} latent frames, "
          f"block size {NUM_FRAME_PER_BLOCK} -> frames 0..{n_ctx-1} = real context, "
          f"{n_ctx}..{NUM_FRAMES-1} generated ===", flush=True)

    prompts = ["a robot arm pushing objects on a table"]
    conditional_dict = model.text_encoder(text_prompts=prompts)

    viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, NUM_FRAMES, 1, 1)
    Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(1, NUM_FRAMES, 1, 1)

    model.scheduler.set_timesteps(args.n_steps)
    schedule = model.scheduler.timesteps.to(device)
    print(f"=== Sampling schedule: {args.n_steps} steps, t from {schedule[0]:.0f} to {schedule[-1]:.0f} ===", flush=True)

    def decode(latent):
        x = model.vae.decode_to_pixel(latent.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    results = {}
    for variant in args.variants:
        override = ACTION_OVERRIDES[variant]
        apl = build_actions(real_actions, NUM_FRAMES, override)
        a = torch.tensor(apl, device=device).unsqueeze(0)
        a_norm = (a - action_mean) / action_std
        a_norm[:, 0, :] = 0.0
        max_sigma = a_norm.abs().max().item()
        action_embed = action_encoder(a_norm.to(torch.bfloat16))
        emb_norm = action_embed.norm(dim=-1).mean().item()
        print(f"\n--- variant '{variant}': max|z|={max_sigma:.2f} sigma, "
              f"||action_embed||={emb_norm:.2f} ---", flush=True)
        if max_sigma > SANITY_MAX_SIGMA:
            print(f"    REFUSING: action is {max_sigma:.1f} sigma out of distribution "
                  f"(this was exactly bug #1). Skipping.", flush=True)
            continue

        sample = real_latent.clone()
        sample[:, n_ctx:] = torch.randn_like(sample[:, n_ctx:])

        for i, t_val in enumerate(schedule):
            timestep = torch.zeros((1, NUM_FRAMES), device=device, dtype=torch.bfloat16)
            timestep[:, n_ctx:] = t_val.item()   # one shared timestep per generated block
            sample[:, :n_ctx] = real_latent[:, :n_ctx]

            flow_pred, x0_pred = model.generator(
                noisy_image_or_video=sample,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=real_latent,   # mask gives noisy block i only clean frames of blocks < i
                aug_t=None,
                viewmats=viewmats,
                Ks=Ks,
                action_embed=action_embed,
            )
            if args.sampler == "euler":
                nxt = model.scheduler.step(
                    flow_pred.flatten(0, 1).float(), timestep.flatten(0, 1).float(),
                    sample.flatten(0, 1).float(), to_final=(i == len(schedule) - 1),
                ).unflatten(0, (1, NUM_FRAMES)).to(torch.bfloat16)
                sample[:, n_ctx:] = nxt[:, n_ctx:]
            else:
                # predict-x0-then-renoise: re-project onto the data manifold each step
                x0 = x0_pred.float()
                if args.x0_clamp > 0:
                    x0 = x0.clamp(-args.x0_clamp, args.x0_clamp)
                if i == len(schedule) - 1:
                    sample[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
                else:
                    s_next = float(model.scheduler.sigmas[i + 1])
                    renoised = (1 - s_next) * x0 + s_next * torch.randn_like(x0)
                    sample[:, n_ctx:] = renoised[:, n_ctx:].to(torch.bfloat16)
            if i % 6 == 0 or i == len(schedule) - 1:
                g = sample[:, n_ctx:].float()
                xp = x0_pred[:, n_ctx:].float()
                print(f"    [step {i:2d}] t={t_val.item():6.1f} "
                      f"x0_pred=({xp.min():+.2f},{xp.max():+.2f}) |x0|={xp.abs().mean():.3f} "
                      f"sample=({g.min():+.2f},{g.max():+.2f})", flush=True)

        sample[:, :n_ctx] = real_latent[:, :n_ctx]
        results[variant] = decode(sample)

    frames_real = decode(real_latent)
    rows = [("REAL (ground truth)", frames_real)] + [(f"GEN action={k}", v) for k, v in results.items()]
    n = min(len(f) for _, f in rows)
    grid = np.concatenate([np.concatenate(list(f[:n]), axis=1) for _, f in rows], axis=0)
    grid_big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST))
    png = os.path.join(args.out_dir, f"v2_idx{args.context_idx}.png")
    Image.fromarray(grid_big).save(png)
    print(f"\n=== saved {png} ===", flush=True)
    print("    rows (top->bottom): " + " | ".join(name for name, _ in rows), flush=True)

    try:
        import imageio
        for variant, frames in results.items():
            mp4 = os.path.join(args.out_dir, f"v2_idx{args.context_idx}_{variant}.mp4")
            imageio.mimsave(mp4, list(frames), fps=4, macro_block_size=1)
            print(f"=== saved {mp4} ===", flush=True)
    except Exception as e:
        print(f"(mp4 export skipped: {e})", flush=True)

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
