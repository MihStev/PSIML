#!/usr/bin/env python
"""
Step 3: REAL (not mock, not synthetic) memory/speed benchmark of a training
step, on actual BAIR-scale data. Combines everything validated separately so
far into one training step:

  - real VAE-encoded BAIR latents + real actions, read from the LMDB built by
    build_bair_lmdb.py (not random tensors, like the earlier mock test)
  - the action-injection mechanism from poc_action_injection.py (ActionEncoder
    MLP writing into the text-embedding pad slot)
  - LoRA (rank configurable via --rank -- see CLAUDE.md, the earlier rank=64
    default was inherited from the VideoX-Fun example recipe, not an actual
    team decision; run with --rank 8/16/64 to compare against what was
    actually discussed)
  - gradient checkpointing explicitly enabled (confirmed already built into
    minWM, see CLAUDE.md -- this just turns it on for this run)
  - configurable --batch_size (repo's own onboarding-world-model skill says
    bs < 8 is not enough for controllability -- default here is 8)

Action handling caveat (still an OPEN design question, not resolved here):
BAIR gives 30 actions/episode (one per frame transition), but the injection
mechanism takes ONE action vector. This benchmark uses the mean of the 30
actions as a placeholder so we can measure memory/speed now -- it does NOT
answer the per-AR-block injection design question, which is a separate
discussion (see kontekst_za_jaci_model.md).

Runs several steps over different real samples and reports the steady-state
(skip first, cold-start) average -- same practice as the earlier mock test.

Usage:
    python real_training_benchmark.py --rank 64 --batch_size 8
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
os.chdir("/home/mls10/minWM-dawidzard")

import lmdb
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=5)
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/train")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def expand_batch(d, bs):
    """Repeat any tensor with leading dim 1 to batch size bs."""
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 and bs != 1:
            reps = [bs] + [1] * (v.dim() - 1)
            out[k] = v.repeat(*reps)
        else:
            out[k] = v
    return out


args = parse_args()
torch.set_grad_enabled(True)
device = torch.device("cuda")
gpu = device

print(f"=== CONFIG: rank={args.rank} batch_size={args.batch_size} n_steps={args.n_steps} ===", flush=True)
t_start = time.time()

config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

from model import CameraCausalDiffusion  # noqa: E402

print("=== Constructing model ===", flush=True)
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

print("=== Enabling gradient checkpointing (confirmed built-in, on by default in configs) ===", flush=True)
model.generator.enable_gradient_checkpointing()

print(f"=== Injecting LoRA (rank={args.rank}) ===", flush=True)
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402

lora_config = LoraConfig(
    r=args.rank, lora_alpha=args.rank * 2,
    target_modules=["q", "k", "v", "ffn.0", "ffn.2"],
)
inject_adapter_in_model(lora_config, model.generator.model)
for name, param in model.generator.model.named_parameters():
    param.requires_grad_("lora_" in name)

n_lora = sum(p.numel() for n, p in model.generator.model.named_parameters() if "lora_" in n)
print(f"=== LoRA (rank={args.rank}) trainable params: {n_lora:,} ({n_lora/1e6:.2f}M) ===", flush=True)

TEXT_DIM = 4096


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
print(f"=== ActionEncoder params: {n_action_params:,} ===", flush=True)

optimizer = torch.optim.AdamW(
    [p for p in model.generator.model.parameters() if p.requires_grad]
    + list(action_encoder.parameters()),
    lr=2e-6,
)

print("=== Real text encoding (frozen, no_grad) ===", flush=True)
prompts = ["mock bair action-conditioned clip"]
with torch.no_grad():
    ids, mask = model.text_encoder.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    seq_lens = mask.gt(0).sum(dim=1).long()
    base_conditional_dict = model.text_encoder(text_prompts=prompts)
    base_unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])
    base_unconditional_dict = {k: v.detach() for k, v in base_unconditional_dict.items()}

base_conditional_dict = expand_batch(base_conditional_dict, args.batch_size)
base_unconditional_dict = expand_batch(base_unconditional_dict, args.batch_size)
seq_lens = seq_lens.repeat(args.batch_size) if args.batch_size != 1 else seq_lens

print("=== Opening real BAIR LMDB ===", flush=True)
env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
lat_shape = get_array_shape_from_lmdb(env, "latents")
act_shape = get_array_shape_from_lmdb(env, "actions")
n_total = lat_shape[0]
NUM_FRAMES = lat_shape[1]
print(f"LMDB: {n_total} samples, latent shape {lat_shape[1:]}, actions {act_shape[1:]}", flush=True)

rng = np.random.default_rng(args.seed)

viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(args.batch_size, NUM_FRAMES, 1, 1)
Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
    .view(1, 1, 3, 3).repeat(args.batch_size, NUM_FRAMES, 1, 1)

step_times = []
peak_mem_gb = 0.0

for step in range(args.n_steps):
    batch_idx = rng.choice(n_total, size=args.batch_size, replace=False)

    latents_np = np.stack([
        retrieve_row_from_lmdb(env, "latents", np.float16, int(i), shape=lat_shape[1:]) for i in batch_idx
    ])
    actions_np = np.stack([
        retrieve_row_from_lmdb(env, "actions", np.float32, int(i), shape=act_shape[1:]) for i in batch_idx
    ])

    clean_latent = torch.from_numpy(latents_np.astype(np.float32)).to(device=device, dtype=torch.bfloat16)
    image_latent = clean_latent[:, 0:1, ...]
    image_or_video_shape = [args.batch_size, NUM_FRAMES, 16, 8, 8]

    # placeholder: mean-pool the 30 raw actions into one vector per sample (open design Q, see docstring)
    action_vec = torch.from_numpy(actions_np.mean(axis=1)).to(device=device, dtype=torch.bfloat16)  # (bs,4)
    action_embed = action_encoder(action_vec)  # (bs, TEXT_DIM)

    prompt_embeds = base_conditional_dict["prompt_embeds"].clone()
    for b in range(args.batch_size):
        slot = seq_lens[b].item()
        prompt_embeds[b, slot, :] = action_embed[b]
    step_conditional_dict = {**base_conditional_dict, "prompt_embeds": prompt_embeds}

    torch.cuda.synchronize(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    t0 = time.time()

    loss, log_dict = model.generator_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=step_conditional_dict,
        unconditional_dict=base_unconditional_dict,
        clean_latent=clean_latent,
        initial_latent=image_latent,
        viewmats=viewmats,
        Ks=Ks,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(gpu)
    t1 = time.time()

    dt = t1 - t0
    peak = torch.cuda.max_memory_allocated(gpu) / 1e9
    peak_mem_gb = max(peak_mem_gb, peak)
    step_times.append(dt)
    print(f"[step {step}] loss={loss.item():.4f} time={dt:.2f}s peak={peak:.2f}GB", flush=True)

steady_times = step_times[1:] if len(step_times) > 1 else step_times
avg_steady = sum(steady_times) / len(steady_times)

print("=== SUMMARY ===", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] all times: {[f'{t:.2f}s' for t in step_times]}", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] steady-state avg (skip first): {avg_steady:.2f}s/step", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] peak memory: {peak_mem_gb:.2f} GB / 40 GB", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] LoRA trainable params: {n_lora/1e6:.2f}M", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] projected 2500 iterations: {avg_steady * 2500 / 3600:.2f}h", flush=True)
print(f"[RESULT rank={args.rank} bs={args.batch_size}] total wall time incl. model load: {time.time() - t_start:.0f}s", flush=True)
print("=== DONE ===", flush=True)
