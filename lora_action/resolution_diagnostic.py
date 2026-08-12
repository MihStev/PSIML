#!/usr/bin/env python
"""
Resolution diagnostic (see CLAUDE.md, risk #1 from the stronger-model review):
does the PRETRAINED (no LoRA, no our changes) teacher-forcing model produce
anything coherent at BAIR's 8x8 latent resolution, or is 64x64 simply too
small a spatial footprint for this backbone (trained at 480p+)?

Reuses generator_loss() as-is (same call as real_training_benchmark.py,
already proven to run) but this time reads log_dict["x0_pred"] -- the
model's single-step prediction of the clean latent from a noisy real BAIR
clip -- instead of discarding it. Decodes both x0_pred and the real x0
through the VAE and saves them side by side as PNGs for visual inspection.

No LoRA injected here on purpose: this tests the frozen pretrained backbone's
raw capacity at this resolution, independent of anything we're adding.
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

import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402

N_SAMPLES = 4  # different real clips, different random noise/timestep each
OUT_DIR = "/home/mls10/logs/resolution_diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

torch.set_grad_enabled(False)
device = torch.device("cuda")
gpu = device

config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

from model import CameraCausalDiffusion  # noqa: E402

print("=== Constructing model (NO LoRA -- testing raw pretrained backbone) ===", flush=True)
model = CameraCausalDiffusion(config, device=device)

print("=== Loading teacher-forcing checkpoint ===", flush=True)
ckpt_path = "/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt"
state_dict = torch.load(ckpt_path, map_location="cpu")
try:
    gen_sd = state_dict["generator_ema"]
except KeyError:
    gen_sd = state_dict["generator"]
try:
    model.generator.load_state_dict(gen_sd)
except RuntimeError:
    fixed = {}
    for k, v in gen_sd.items():
        if k.startswith("model._fsdp_wrapped_module."):
            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[k] = v
    model.generator.load_state_dict(fixed, strict=False)

model.generator.to(device=device, dtype=torch.bfloat16)
model.text_encoder.to(device=device, dtype=torch.bfloat16)
model.vae.to(device=device, dtype=torch.bfloat16)


def decode_to_frames(latent_1xFx16x8x8):
    """(1,F,16,8,8) latent -> (F',64,64,3) uint8 numpy via WanVAEWrapper.decode_to_pixel."""
    x = model.vae.decode_to_pixel(latent_1xFx16x8x8.to(device))  # (1,F,3,64,64) in [-1,1]
    frames = ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()
    return frames


print("=== Real text encoding ===", flush=True)
prompts = ["mock bair action-conditioned clip"]
ids, mask = model.text_encoder.tokenizer(prompts, return_mask=True, add_special_tokens=True)
conditional_dict = model.text_encoder(text_prompts=prompts)
unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])

print("=== Opening real BAIR LMDB ===", flush=True)
env = lmdb.open("/tmp/bair_lmdb/train", readonly=True, lock=False)
lat_shape = get_array_shape_from_lmdb(env, "latents")
NUM_FRAMES = lat_shape[1]
n_total = lat_shape[0]

rng = np.random.default_rng(42)
sample_indices = rng.choice(n_total, size=N_SAMPLES, replace=False)

viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, NUM_FRAMES, 1, 1)
Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
    .view(1, 1, 3, 3).repeat(1, NUM_FRAMES, 1, 1)

for i, idx in enumerate(sample_indices):
    idx = int(idx)
    latent_np = retrieve_row_from_lmdb(env, "latents", np.float16, idx, shape=lat_shape[1:])
    clean_latent = torch.from_numpy(latent_np.astype(np.float32)).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    image_latent = clean_latent[:, 0:1, ...]
    image_or_video_shape = [1, NUM_FRAMES, 16, 8, 8]

    loss, log_dict = model.generator_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        clean_latent=clean_latent,
        initial_latent=image_latent,
        viewmats=viewmats,
        Ks=Ks,
    )

    x0_real = log_dict["x0"]
    x0_pred = log_dict["x0_pred"]
    print(f"[sample {i}] idx={idx} loss={loss.item():.4f} "
          f"x0_pred range=({x0_pred.float().min().item():.2f},{x0_pred.float().max().item():.2f}) "
          f"nan={torch.isnan(x0_pred.float()).any().item()}", flush=True)

    frames_real = decode_to_frames(x0_real.to(torch.bfloat16))
    frames_pred = decode_to_frames(x0_pred.to(torch.bfloat16))

    n = min(frames_real.shape[0], frames_pred.shape[0])
    grid = np.concatenate([
        np.concatenate(list(frames_real[:n]), axis=1),
        np.concatenate(list(frames_pred[:n]), axis=1),
    ], axis=0)  # top row = real, bottom row = model's x0 prediction
    grid_big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST))
    out_path = os.path.join(OUT_DIR, f"sample_{i}_idx{idx}.png")
    Image.fromarray(grid_big).save(out_path)
    print(f"[sample {i}] saved {out_path}", flush=True)

print("=== DONE ===", flush=True)
