#!/usr/bin/env python
"""
Second half of the overfit-single-batch sanity check (see CLAUDE.md /
stronger-model review): loads a checkpoint saved by train_lora_action.py
--overfit_single_batch, rebuilds the model with those trained LoRA +
ActionEncoder weights, runs ONE forward pass on the SAME fixed batch it was
trained on, decodes the prediction through the VAE, and saves it next to the
real frames for visual comparison.

Loss going near-zero (train_lora_action.py's own log) is necessary but not
sufficient -- this is the actual "does it visually match" check.
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

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from bair_dataset import BairActionLatentDataset
from train_lora_action import ActionEncoderV2  # reuse exact same class

CKPT_PATH = "/home/mls10/checkpoints/bair_lora/step_300.pt"
LMDB_PATH = "/tmp/bair_lmdb/train"
BATCH_SIZE = 8
OUT_PATH = "/home/mls10/logs/overfit_visual_check.png"

torch.set_grad_enabled(False)
device = torch.device("cuda")

print(f"=== Loading checkpoint {CKPT_PATH} ===", flush=True)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
print(f"checkpoint step={ckpt['step']} args={ckpt['args']}", flush=True)

config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

from model import CameraCausalDiffusion  # noqa: E402

print("=== Constructing model + loading BASE checkpoint ===", flush=True)
model = CameraCausalDiffusion(config, device=device)
base_ckpt_path = "/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt"
state_dict = torch.load(base_ckpt_path, map_location="cpu")
gen_sd = state_dict.get("generator_ema", state_dict.get("generator"))
try:
    model.generator.load_state_dict(gen_sd)
except RuntimeError:
    fixed = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}
    model.generator.load_state_dict(fixed, strict=False)
model.generator.to(device=device, dtype=torch.bfloat16)
model.text_encoder.to(device=device, dtype=torch.bfloat16)
model.vae.to(device=device, dtype=torch.bfloat16)

rank = ckpt["args"]["rank"]
print(f"=== Injecting LoRA (rank={rank}) and loading TRAINED weights ===", flush=True)
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

lora_config = LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"])
inject_adapter_in_model(lora_config, model.generator.model)
missing, unexpected = model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
print(f"LoRA load: {len(ckpt['lora_state_dict'])} tensors applied", flush=True)

action_encoder = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
action_encoder.load_state_dict(ckpt["action_encoder_state_dict"])
null_action_embedding = ckpt["null_action_embedding"].to(device)
action_mean = ckpt["action_mean"].to(device)
action_std = ckpt["action_std"].to(device)

print("=== Rebuilding the EXACT same fixed batch (same seed/order as training) ===", flush=True)
dataset = BairActionLatentDataset(LMDB_PATH)


def collate(batch):
    return {
        "clean_latent": torch.stack([b["clean_latent"] for b in batch]),
        "actions_per_latent": torch.stack([b["actions_per_latent"] for b in batch]),
    }


loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate, drop_last=True)
batch = next(iter(loader))  # shuffle=False -> first batch, matches training's fixed_batch

clean_latent = batch["clean_latent"].to(device=device, dtype=torch.bfloat16)
actions_raw = batch["actions_per_latent"].to(device)
actions_norm = (actions_raw - action_mean) / action_std
actions_norm[:, 0, :] = 0.0
action_embed = action_encoder(actions_norm.to(torch.bfloat16))  # NO dropout at eval time

print("=== Real text encoding ===", flush=True)
prompts = ["a robot arm pushing objects on a table"]
conditional_dict = model.text_encoder(text_prompts=prompts)
unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])

NUM_FRAMES = clean_latent.shape[1]
viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(BATCH_SIZE, NUM_FRAMES, 1, 1)
Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
    .view(1, 1, 3, 3).repeat(BATCH_SIZE, NUM_FRAMES, 1, 1)

print("=== Forward pass with TRAINED weights ===", flush=True)
loss, log_dict = model.generator_loss(
    image_or_video_shape=[BATCH_SIZE, NUM_FRAMES, 16, 8, 8],
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=clean_latent,
    initial_latent=clean_latent[:, 0:1, ...],
    viewmats=viewmats,
    Ks=Ks,
    action_embed=action_embed,
)
print(f"[eval] loss={loss.item():.4f}", flush=True)

x0_real = log_dict["x0"]
x0_pred = log_dict["x0_pred"]


def decode(latent):
    x = model.vae.decode_to_pixel(latent.to(device))
    return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()  # (B,F,64,64,3)


frames_real = decode(x0_real)
frames_pred = decode(x0_pred)

rows = []
for b in range(BATCH_SIZE):
    top = np.concatenate(list(frames_real[b]), axis=1)
    bot = np.concatenate(list(frames_pred[b]), axis=1)
    rows.append(np.concatenate([top, bot], axis=0))
grid = np.concatenate(rows, axis=0)
grid_big = np.array(Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST))
Image.fromarray(grid_big).save(OUT_PATH)
print(f"=== saved {OUT_PATH} (top=real, bottom=trained-model-prediction, per sample in batch) ===", flush=True)
print("=== DONE ===", flush=True)
