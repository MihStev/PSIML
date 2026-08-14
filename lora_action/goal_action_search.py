#!/usr/bin/env python
"""Pick an action by imagining its consequence (Danilo's demo idea).

Given a scene's real context and a GOAL frame, sample a grid of candidate actions,
predict one block for each, and score each prediction against the goal. The action
whose imagined future lands closest to the goal is the chosen one.

This is the forward model read backwards: the same network that answers "what happens
if I do X" is used to answer "what X gets me there" -- forward vs inverse dynamics,
as Danilo put it, two views of one model.

Why this is a real test and not a trick: the goal frame is the episode's OWN recorded
future, so the episode's real action is the ground truth answer. We can therefore check
whether the chosen action agrees with the recorded one, instead of just asserting that
the picture looks right.

All candidates are evaluated in ONE batched forward pass per denoising step, and share
the SAME noise -- so differences between candidates come from the action alone, exactly
as in evaluate.py's mode B.

Output:
  - goal_search_idx<N>.png : goal frame, best/worst imagined future, and the score
    surface over the (dx, dy) grid
  - goal_search_idx<N>.json : chosen action, true action, ranking, agreement
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

import lmdb                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
from PIL import Image                                          # noqa: E402

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb   # noqa: E402
from train_lora_action import ActionEncoderV2                                   # noqa: E402

D = 0.07                 # hard range of the real displacement dims
SANITY_MAX_SIGMA = 6.0   # never feed an action further out than this (bug #1 of the project)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test", help="HELD-OUT split")
    p.add_argument("--context_idx", type=int, default=3)
    p.add_argument("--grid", type=int, default=6, help="grid**2 candidate actions")
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--dmd_schedule", action="store_true")
    p.add_argument("--metric", default="l1", choices=["l1", "psnr"])
    p.add_argument("--out_dir", default="/home/mls10/logs/goal_search")
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

    from model import CameraCausalDiffusion                     # noqa: E402
    print("=== Loading model ===", flush=True)
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
                                       shape=act_shape[1:])                # (30, 4)

    # the episode's own action over the GENERATED latent frames -> ground-truth answer
    raw_lo, raw_hi = 4 * (n_ctx - 1), 4 * (NUM_FRAMES - 1)
    true_dx, true_dy = real_act[raw_lo:raw_hi, 0].mean(), real_act[raw_lo:raw_hi, 1].mean()
    print(f"=== episode's own action over generated frames: dx={true_dx:+.4f} dy={true_dy:+.4f} ===",
          flush=True)

    # ---- candidate grid -------------------------------------------------------
    gs = args.grid
    axis = np.linspace(-D, D, gs)
    cand = np.array([(dx, dy) for dy in axis for dx in axis], dtype=np.float32)   # (gs*gs, 2)
    B = len(cand)
    print(f"=== {B} candidate actions on a {gs}x{gs} grid over [-{D}, {D}]^2 ===", flush=True)

    apl = np.zeros((B, NUM_FRAMES, 16), dtype=np.float32)
    for i in range(1, NUM_FRAMES):                      # frame 0 never carries an action
        for b, (dx, dy) in enumerate(cand):
            apl[b, i] = np.tile([dx, dy, 0.5, 0.25], 4)     # dims 2/3 at dataset means
    a = torch.tensor(apl, device=device)
    an = (a - a_mean) / a_std
    an[:, 0, :] = 0.0
    worst_sigma = float(an.abs().max())
    assert worst_sigma < SANITY_MAX_SIGMA, f"action {worst_sigma:.1f} sigma out of distribution"
    action_embed = action_encoder(an.to(torch.bfloat16))

    vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(B, NUM_FRAMES, 1, 1)
    ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(B, NUM_FRAMES, 1, 1)
    condB = {k: (v.repeat(B, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
             for k, v in cond.items()}

    if args.dmd_schedule:
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        schedule = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (schedule.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = schedule.cpu()
        print(f"=== DMD 4-step schedule: {[round(float(t),1) for t in schedule]} ===", flush=True)
    else:
        model.scheduler.set_timesteps(args.n_steps)
        schedule = model.scheduler.timesteps.to(device)

    latB = real_latent.repeat(B, 1, 1, 1, 1)
    noise = torch.randn_like(real_latent).repeat(B, 1, 1, 1, 1)   # SAME noise for all candidates

    s = latB.clone()
    s[:, n_ctx:] = noise[:, n_ctx:]
    for i, t_val in enumerate(schedule):
        timestep = torch.zeros((B, NUM_FRAMES), device=device, dtype=torch.bfloat16)
        timestep[:, n_ctx:] = t_val.item()
        s[:, :n_ctx] = latB[:, :n_ctx]
        _, x0 = model.generator(noisy_image_or_video=s, conditional_dict=condB, timestep=timestep,
                                clean_x=latB, aug_t=None, viewmats=vm, Ks=ks,
                                action_embed=action_embed)
        x0 = x0.float().clamp(-6, 6)
        if i == len(schedule) - 1:
            s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
        else:
            sn = float(model.scheduler.sigmas[i + 1])
            s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
    s[:, :n_ctx] = latB[:, :n_ctx]

    def decode(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

    preds = np.concatenate([decode(s[i:i + 8]) for i in range(0, B, 8)], axis=0)   # (B,F,H,W,3)
    goal_all = decode(real_latent)[0]                                             # (F,H,W,3)
    goal = goal_all[-1].astype(np.float32)                                        # LAST real frame

    # ---- score every imagined future against the goal -------------------------
    scores = []
    for b in range(B):
        p = preds[b, -1].astype(np.float32)
        if args.metric == "l1":
            scores.append(float(np.abs(p - goal).mean()))            # lower is better
        else:
            mse = float(((p - goal) ** 2).mean())
            scores.append(-10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))  # negated PSNR
    scores = np.array(scores)
    best, worst = int(scores.argmin()), int(scores.argmax())
    bdx, bdy = cand[best]

    # does the chosen action agree with the episode's real one?
    agree_x = bool(np.sign(bdx) == np.sign(true_dx)) if abs(true_dx) > 1e-4 else None
    agree_y = bool(np.sign(bdy) == np.sign(true_dy)) if abs(true_dy) > 1e-4 else None
    print(f"=== CHOSEN  dx={bdx:+.4f} dy={bdy:+.4f}  ({args.metric}={scores[best]:.3f}) ===", flush=True)
    print(f"=== TRUE    dx={true_dx:+.4f} dy={true_dy:+.4f} ===", flush=True)
    print(f"=== sign agreement: x={agree_x}  y={agree_y} ===", flush=True)
    print(f"=== score spread across candidates: min={scores.min():.3f} max={scores.max():.3f} "
          f"(a flat spread would mean the action barely matters) ===", flush=True)

    # ---- figure ---------------------------------------------------------------
    H, W = goal.shape[:2]
    surf = scores.reshape(gs, gs)
    surf_n = (surf - surf.min()) / (surf.max() - surf.min() + 1e-9)
    heat = np.stack([1 - surf_n, 1 - surf_n, np.ones_like(surf_n)], -1)   # blue = better
    heat_img = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.NEAREST))

    panels = [("GOAL (real future)", goal.astype(np.uint8)),
              ("BEST imagined", preds[best, -1]),
              ("WORST imagined", preds[worst, -1]),
              (f"score over dx,dy ({gs}x{gs})", heat_img)]
    canvas = Image.new("RGB", (W * len(panels), H), (255, 255, 255))
    for i, (_, img) in enumerate(panels):
        canvas.paste(Image.fromarray(img), (i * W, 0))
    canvas = canvas.resize((W * len(panels) * 3, H * 3), Image.NEAREST)
    png = os.path.join(args.out_dir, f"goal_search_idx{args.context_idx}.png")
    canvas.save(png)
    print("  panels:", " | ".join(t for t, _ in panels), flush=True)
    print("  ->", png, flush=True)

    order = np.argsort(scores)
    out = {
        "context_idx": args.context_idx, "metric": args.metric, "n_candidates": B,
        "grid": gs, "n_steps": len(schedule),
        "chosen": {"dx": float(bdx), "dy": float(bdy), "score": float(scores[best])},
        "true": {"dx": float(true_dx), "dy": float(true_dy)},
        "sign_agreement": {"x": agree_x, "y": agree_y},
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "top5": [{"dx": float(cand[i][0]), "dy": float(cand[i][1]), "score": float(scores[i])}
                 for i in order[:5]],
    }
    js = os.path.join(args.out_dir, f"goal_search_idx{args.context_idx}.json")
    json.dump(out, open(js, "w"), indent=2)
    print("  ->", js, flush=True)


if __name__ == "__main__":
    main()
