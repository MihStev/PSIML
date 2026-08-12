#!/usr/bin/env python
"""
Proof-of-concept: does injecting an action embedding into the text context's
unused (zero-padded) slots actually flow through the real model without shape
errors, and does gradient reach both the new ActionEncoder MLP and the LoRA
adapter? Uses REAL text encoder output (not fully synthetic like last night's
mock test) but still synthetic BAIR-shaped video latents (real data pipeline
comes after this is validated).
"""
import os
import sys
import time

os.environ.setdefault("USER", "mls10")
os.environ.setdefault("LOGNAME", "mls10")
os.environ.setdefault("HOME", "/home/mls10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/mls10/minWM/Wan21")
sys.path.insert(0, "/home/mls10/minWM/shared")
os.chdir("/home/mls10/minWM")

import torch
import torch.nn as nn
from omegaconf import OmegaConf

torch.set_grad_enabled(True)
device = torch.device("cuda")
gpu = device

config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

from model import CameraCausalDiffusion  # noqa: E402

print("=== Constructing model ===")
model = CameraCausalDiffusion(config, device=device)

print("=== Loading teacher-forcing checkpoint ===")
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

print("=== Injecting LoRA (rank=8 this time, per today's plan) ===")
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

LORA_RANK = 8
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_RANK * 2,
    target_modules=["q", "k", "v", "ffn.0", "ffn.2"],
)
inject_adapter_in_model(lora_config, model.generator.model)
for name, param in model.generator.model.named_parameters():
    param.requires_grad_("lora_" in name)

n_lora = sum(p.numel() for n, p in model.generator.model.named_parameters() if "lora_" in n)
print(f"=== LoRA (rank={LORA_RANK}) trainable params: {n_lora:,} ({n_lora/1e6:.2f}M) ===")

# ===== NEW: action encoder =====
TEXT_DIM = 4096  # umt5-xxl output dim, confirmed from wan/modules/t5.py

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=4, out_dim=TEXT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, action):
        return self.net(action)

action_encoder = ActionEncoder().to(device=device, dtype=torch.bfloat16)
n_action_params = sum(p.numel() for p in action_encoder.parameters())
print(f"=== ActionEncoder params: {n_action_params:,} ===")

# Both LoRA and ActionEncoder params go into the same optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.generator.model.parameters() if p.requires_grad]
    + list(action_encoder.parameters()),
    lr=2e-6,
)

# ===== Real text encoding, to get real seq_lens (how much padding room we have) =====
print("=== Real text encoding (frozen, no_grad) ===")
prompts = ["mock bair action-conditioned clip"]
with torch.no_grad():
    ids, mask = model.text_encoder.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    print(f"real caption token length: {seq_lens.tolist()} (budget: 512)")

    conditional_dict = model.text_encoder(text_prompts=prompts)
    unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])
    unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}

# ===== THE ACTUAL INJECTION =====
BATCH = 1
action_vec = torch.randn(BATCH, 4, device=device, dtype=torch.bfloat16)  # dummy BAIR action
action_embed = action_encoder(action_vec)  # [B, TEXT_DIM]

prompt_embeds = conditional_dict["prompt_embeds"].clone()  # [B, 512, 4096], don't mutate cached
for b in range(BATCH):
    slot = seq_lens[b].item()
    assert slot < prompt_embeds.shape[1] - 1, "no room left in text budget for action token!"
    prompt_embeds[b, slot, :] = action_embed[b]
conditional_dict = {**conditional_dict, "prompt_embeds": prompt_embeds}

print(f"=== Action embedding injected at position {seq_lens.tolist()} (out of 512) ===")

# ===== Synthetic BAIR-scale video latent (real data pipeline comes next, after this passes) =====
NUM_FRAMES = 16
clean_latent = torch.randn(BATCH, NUM_FRAMES, 16, 8, 8, device=device, dtype=torch.bfloat16)
image_latent = clean_latent[:, 0:1, ...]
image_or_video_shape = [BATCH, NUM_FRAMES, 16, 8, 8]

viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(BATCH, NUM_FRAMES, 1, 1)
Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
    .view(1, 1, 3, 3).repeat(BATCH, NUM_FRAMES, 1, 1)

torch.cuda.synchronize(gpu)
torch.cuda.reset_peak_memory_stats(gpu)

print("=== Running ONE training step with action injection ===")
t0 = time.time()
loss, log_dict = model.generator_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=clean_latent,
    initial_latent=image_latent,
    viewmats=viewmats,
    Ks=Ks,
)
optimizer.zero_grad()
loss.backward()

# Sanity check: did gradient actually reach the action encoder?
action_grad_norm = sum(
    p.grad.norm().item() for p in action_encoder.parameters() if p.grad is not None
)
optimizer.step()
torch.cuda.synchronize(gpu)
t1 = time.time()

print(f"[RESULT] loss={loss.item():.4f}")
print(f"[RESULT] ActionEncoder grad norm (should be > 0!): {action_grad_norm:.6f}")
print(f"[RESULT] wall time: {t1 - t0:.3f}s")
print(f"[MEM] peak: max_allocated={torch.cuda.max_memory_allocated(gpu)/1e9:.2f} GB, "
      f"max_reserved={torch.cuda.max_memory_reserved(gpu)/1e9:.2f} GB")
print("=== DONE ===")
