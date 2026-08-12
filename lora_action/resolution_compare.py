#!/usr/bin/env python
"""
Test (b): cleaner resolution diagnostic. Fixed, UNIFORM moderate noise level
across all frames (not the per-frame training-sampled noise, which confounded
the earlier single-shot test with "high noise = mush" regardless of
resolution), plus a few ITERATIVE refinement steps instead of one shot.

Also directly answers the second question: runs the SAME protocol at
64x64 (native BAIR), 128x128, and 256x256 (bicubic-upsampled raw frames,
re-encoded through the VAE) so the three are apples-to-apples comparable.

No LoRA (raw pretrained backbone, same as resolution_diagnostic.py).
"""
import os
import sys

os.environ.setdefault("USER", "mls10")
os.environ.setdefault("LOGNAME", "mls10")
os.environ.setdefault("HOME", "/home/mls10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/mls10/minWM-dawidzard/Wan21")
sys.path.insert(0, "/home/mls10/minWM-dawidzard/shared")
os.chdir("/home/mls10/minWM-dawidzard")

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

RESOLUTIONS = [64, 128, 256]
N_SAMPLES = 2
N_REFINE_STEPS = 3
FIXED_SIGMAS = [0.7, 0.4, 0.15]  # descending, moderate -> low noise
OUT_DIR = "/home/mls10/logs/resolution_compare"
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_grad_enabled(False)
device = torch.device("cuda")

config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

from model import CameraCausalDiffusion  # noqa: E402

print("=== Constructing model (NO LoRA) ===", flush=True)
model = CameraCausalDiffusion(config, device=device)

print("=== Loading teacher-forcing checkpoint ===", flush=True)
ckpt_path = "/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
gen_sd = state_dict.get("generator_ema", state_dict.get("generator"))
try:
    model.generator.load_state_dict(gen_sd)
except RuntimeError:
    fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}
    model.generator.load_state_dict(fixed, strict=False)

model.generator.to(device=device, dtype=torch.bfloat16)
model.text_encoder.to(device=device, dtype=torch.bfloat16)
model.vae.to(device=device, dtype=torch.bfloat16)

print("=== Real text encoding ===", flush=True)
prompts = ["mock bair action-conditioned clip"]
conditional_dict = model.text_encoder(text_prompts=prompts)

print("=== Loading raw BAIR frames ===", flush=True)
d = np.load("/tmp/bair_raw/test/shard_00000.npz")
raw_images = d["images"]  # (256,30,64,64,3) uint8
sample_indices = [7, 42]


def encode_at_resolution(raw_seq_30x64x64x3, res):
    """Upsample (bicubic) raw 64x64 frames to res x res, then VAE-encode."""
    frames = []
    for f in raw_seq_30x64x64x3:
        img = Image.fromarray(f)
        if res != 64:
            img = img.resize((res, res), Image.BICUBIC)
        frames.append(np.array(img))
    frames = np.stack(frames)  # (30,res,res,3)
    pixel = torch.from_numpy(frames).float().permute(3, 0, 1, 2)  # (3,30,res,res)
    pixel = ((pixel / 255.0 - 0.5) * 2.0).unsqueeze(0).to(device, torch.bfloat16)  # (1,3,30,res,res)
    latent = model.vae.encode_to_latent(pixel)  # (1,F,16,res/8,res/8), returned as float32
    return latent.to(torch.bfloat16)


def decode_frames(latent_1xFxCxHxW):
    x = model.vae.decode_to_pixel(latent_1xFxCxHxW.to(device))
    return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()


scheduler = model.scheduler

for res in RESOLUTIONS:
    print(f"\n########## RESOLUTION {res} ##########", flush=True)
    for si, idx in enumerate(sample_indices):
        raw_seq = raw_images[idx]
        clean_latent = encode_at_resolution(raw_seq, res)  # (1,F,16,H,W)
        _, F, C, H, W = clean_latent.shape

        noise = torch.randn_like(clean_latent)
        current = clean_latent.clone()  # will be progressively re-noised & refined
        for step_i, sigma in enumerate(FIXED_SIGMAS):
            timestep = torch.full((1, F), sigma * scheduler.num_train_timesteps,
                                   device=device, dtype=torch.bfloat16)
            noisy = scheduler.add_noise(
                clean_latent.flatten(0, 1).float(), noise.flatten(0, 1).float(),
                timestep.flatten(0, 1).float()
            ).unflatten(0, (1, F)).to(torch.bfloat16)

            flow_pred, x0_pred = model.generator(
                noisy_image_or_video=noisy,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent if getattr(model, "teacher_forcing", False) else None,
                aug_t=None,
            )
            current = x0_pred
            noise = torch.randn_like(clean_latent)  # fresh noise for next re-noise
            print(f"[res={res} sample={si} step={step_i} sigma={sigma}] "
                  f"x0_pred range=({x0_pred.float().min().item():.2f},{x0_pred.float().max().item():.2f}) "
                  f"nan={torch.isnan(x0_pred.float()).any().item()}", flush=True)

        frames_real = decode_frames(clean_latent)
        frames_pred = decode_frames(current)
        n = min(frames_real.shape[0], frames_pred.shape[0])
        grid = np.concatenate([
            np.concatenate(list(frames_real[:n]), axis=1),
            np.concatenate(list(frames_pred[:n]), axis=1),
        ], axis=0)
        scale = max(1, 512 // res)
        grid_big = np.array(Image.fromarray(grid).resize(
            (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST))
        out_path = os.path.join(OUT_DIR, f"res{res}_sample{si}_idx{idx}.png")
        Image.fromarray(grid_big).save(out_path)
        print(f"[res={res} sample={si}] saved {out_path}", flush=True)

print("\n=== DONE ===", flush=True)
