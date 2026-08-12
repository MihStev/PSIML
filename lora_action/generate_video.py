#!/usr/bin/env python
"""
Actually GENERATE a video: given a real BAIR frame as the starting scene and a
CHOSEN action sequence (not from the dataset -- typed in / picked by us), roll
out the future frames from pure noise using the TRAINED LoRA + ActionEncoder,
decode, save as mp4 + a comparison PNG.

Sampling method: same iterative-refinement technique already validated in
resolution_compare.py (fixed descending noise levels + a few refinement
steps) -- proven to give coherent output at 64x64. NOT the distilled-model
CausalInferencePipeline (that needs `denoising_step_list`, which only exists
in the later-stage configs, not our teacher-forcing Stage 1 config).

Usage:
    python generate_video.py --checkpoint /home/mls10/checkpoints/bair_lora/step_2500.pt \
        --context_idx 100 --action right    # or --action left / --action custom --action_vec ...
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

FIXED_SIGMAS = [0.9, 0.7, 0.5, 0.3, 0.15]  # 5 steps, starting from ~pure noise this time
# (more steps than resolution_compare.py's 3, since we start from noise, not a lightly-noised real sample)

# hand-picked "canonical" gripper delta directions for --action left/right/up/down
# (BAIR action = [dx, dy, ...]; exact scale doesn't matter much, direction does)
CANONICAL_ACTIONS = {
    "left":  np.array([-3.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "right": np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "up":    np.array([0.0, -3.0, 0.0, 0.0], dtype=np.float32),
    "down":  np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32),
    "still": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/train")
    p.add_argument("--context_idx", type=int, default=100, help="LMDB index to take the starting scene from")
    p.add_argument("--action", default="right", choices=list(CANONICAL_ACTIONS) + ["custom"])
    p.add_argument("--action_vec", type=float, nargs=4, default=None,
                    help="used only with --action custom, e.g. --action_vec 3 0 0 0")
    p.add_argument("--out_dir", default="/home/mls10/logs/generated_videos")
    return p.parse_args()


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
    default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    from model import CameraCausalDiffusion  # noqa: E402

    print("=== Constructing model + loading BASE checkpoint ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base_ckpt = torch.load("/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt", map_location="cpu")
    gen_sd = base_ckpt.get("generator_ema", base_ckpt.get("generator"))
    try:
        model.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}
        model.generator.load_state_dict(fixed, strict=False)
    model.generator.to(device=device, dtype=torch.bfloat16)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)
    model.vae.to(device=device, dtype=torch.bfloat16)

    print(f"=== Injecting LoRA (rank={rank}) + loading TRAINED weights ===", flush=True)
    from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
    lora_config = LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"])
    inject_adapter_in_model(lora_config, model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
    action_mean = ckpt["action_mean"].to(device)
    action_std = ckpt["action_std"].to(device)

    print(f"=== Real starting scene: LMDB idx={args.context_idx} ===", flush=True)
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    NUM_FRAMES = lat_shape[1]
    context_latent_np = retrieve_row_from_lmdb(env, "latents", np.float16, args.context_idx, shape=lat_shape[1:])
    context_latent = torch.from_numpy(context_latent_np.astype(np.float32)).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    real_full_latent = context_latent.clone()  # keep the real continuation too, for side-by-side comparison

    # build the CHOSEN action, aligned to latent frames the same way as training (frame 0 = none,
    # frames 1..F-1 get the SAME chosen action repeated -- we don't have per-transition detail for
    # a hand-picked action, so we broadcast one direction across the whole rollout)
    if args.action == "custom":
        assert args.action_vec is not None, "--action custom requires --action_vec DX DY DZ DW"
        base_action = np.array(args.action_vec, dtype=np.float32)
    else:
        base_action = CANONICAL_ACTIONS[args.action]
    print(f"=== Chosen action: {args.action} = {base_action.tolist()} ===", flush=True)

    actions_per_latent = np.zeros((NUM_FRAMES, 16), dtype=np.float32)
    for i in range(1, NUM_FRAMES):
        actions_per_latent[i] = np.tile(base_action, 4)  # same action for all 4 raw-frame slots
    actions_t = torch.tensor(actions_per_latent, device=device, dtype=torch.float32).unsqueeze(0)
    actions_norm = (actions_t - action_mean) / action_std
    actions_norm[:, 0, :] = 0.0
    action_embed = action_encoder(actions_norm.to(torch.bfloat16))

    print("=== Real text encoding ===", flush=True)
    prompts = ["a robot arm pushing objects on a table"]
    conditional_dict = model.text_encoder(text_prompts=prompts)
    unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])

    viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, NUM_FRAMES, 1, 1)
    Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(1, NUM_FRAMES, 1, 1)

    print(f"=== Generating: frame 0 kept as real context, frames 1..{NUM_FRAMES-1} from PURE NOISE ===", flush=True)
    # keep the real first (context) latent frame; generate the rest from noise
    noise = torch.randn_like(context_latent)
    current = context_latent.clone()
    current[:, 1:] = noise[:, 1:]

    for step_i, sigma in enumerate(FIXED_SIGMAS):
        timestep = torch.full((1, NUM_FRAMES), sigma * model.scheduler.num_train_timesteps,
                               device=device, dtype=torch.bfloat16)
        timestep[:, 0] = 0.0  # frame 0 (real context) always at timestep 0 -- it's given, not noisy
        noisy = model.scheduler.add_noise(
            current.flatten(0, 1).float(), torch.randn_like(current).flatten(0, 1).float(),
            timestep.flatten(0, 1).float()
        ).unflatten(0, (1, NUM_FRAMES)).to(torch.bfloat16)
        noisy[:, 0] = context_latent[:, 0]  # frame 0 always exactly the real context, never noised

        flow_pred, x0_pred = model.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=context_latent if getattr(model, "teacher_forcing", False) else None,
            aug_t=None,
            viewmats=viewmats,
            Ks=Ks,
            action_embed=action_embed,
        )
        current = x0_pred
        current[:, 0] = context_latent[:, 0]
        print(f"[refine step {step_i}] sigma={sigma} x0_pred range="
              f"({x0_pred.float().min().item():.2f},{x0_pred.float().max().item():.2f})", flush=True)

    def decode(latent):
        x = model.vae.decode_to_pixel(latent.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    frames_generated = decode(current)
    frames_real_continuation = decode(real_full_latent)  # what actually happened in the dataset

    n = min(frames_generated.shape[0], frames_real_continuation.shape[0])
    grid = np.concatenate([
        np.concatenate(list(frames_real_continuation[:n]), axis=1),
        np.concatenate(list(frames_generated[:n]), axis=1),
    ], axis=0)
    grid_big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST))
    png_path = os.path.join(args.out_dir, f"gen_idx{args.context_idx}_action-{args.action}.png")
    Image.fromarray(grid_big).save(png_path)
    print(f"=== saved comparison PNG: {png_path} (top=real continuation in dataset, "
          f"bottom=OUR generated rollout with chosen action) ===", flush=True)

    try:
        import imageio
        mp4_path = os.path.join(args.out_dir, f"gen_idx{args.context_idx}_action-{args.action}.mp4")
        imageio.mimsave(mp4_path, [f for f in frames_generated], fps=4)
        print(f"=== saved mp4: {mp4_path} ===", flush=True)
    except ImportError:
        print("=== imageio not available, skipping mp4 (PNG comparison is enough for now) ===", flush=True)

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
