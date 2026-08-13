#!/usr/bin/env python
"""
Real BAIR LoRA + action-conditioning training loop.

Combines every validated piece so far into one script:
  - BairActionLatentDataset (bair_dataset.py): real LMDB latents + per-latent-
    frame aligned raw actions (see CLAUDE.md, "ARHITEKTONSKA ODLUKA")
  - ActionEncoder v2: 16 -> 256 -> 256 -> dim(=1536, verified from Wan2.1-T2V-1.3B/config.json), zero-init final layer,
    injected via the per-frame timestep/AdaLN path (causal_model.py, verified
    -- NOT the old text-embedding-slot mechanism, which is retired)
  - LoRA (rank configurable; cost is ~identical across 8/16/64 per the
    benchmark sweep, so this is a quality/overfitting decision, not a
    performance one -- pick with --rank)
  - gradient checkpointing (confirmed already built into minWM, just enabled)
  - reuses generator_loss()'s existing timestep-sampling code path unchanged
    (does NOT reinvent the noise schedule)
  - action normalization (mean/std over a sample of the training set, cached
    in the checkpoint) and action dropout (p=0.1, learned null embedding,
    enables classifier-free guidance at inference)

Usage:
    python train_lora_action.py --rank 16 --batch_size 8 --max_steps 2500
    python train_lora_action.py --overfit_single_batch --max_steps 300   # sanity check first!
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
sys.path.insert(0, "/home/mls10/minWM-dawidzard/lora_action")
os.chdir("/home/mls10/minWM-dawidzard")

import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from bair_dataset import BairActionLatentDataset, compute_action_stats, ACTIONS_PER_LATENT_DIM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=2500)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--lr_action", type=float, default=3e-4)
    p.add_argument("--action_dropout_p", type=float, default=0.1)
    p.add_argument("--checkpoint_every", type=int, default=250)
    p.add_argument("--checkpoint_dir", default="/home/mls10/checkpoints/bair_lora")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/train")
    p.add_argument("--overfit_single_batch", action="store_true",
                    help="sanity check: train on ONE fixed batch repeatedly, loss should -> ~0")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="bair-action-lora")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="disable W&B (e.g. for quick local debugging)")
    p.add_argument("--val_lmdb_path", default="/tmp/bair_lmdb/test",
                    help="held-out split -- never trained on, so val loss is the honest signal")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_batches", type=int, default=8)
    p.add_argument("--base_checkpoint", default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt",
                    help="which pretrained checkpoint to adapt; point at the DMD (distilled)\n                          model to run the mentor's open question")
    return p.parse_args()


class ActionEncoderV2(nn.Module):
    """Per-latent-frame action encoder. Final layer zero-init so the model's
    behavior at step 0 is identical to the pretrained checkpoint (nothing is
    destroyed by the fresh, randomly-initialized module at training start)."""

    def __init__(self, in_dim=ACTIONS_PER_LATENT_DIM, hidden=256, out_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, actions_per_latent):  # (B, F, in_dim) -> (B, F, out_dim)
        return self.net(actions_per_latent)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"=== CONFIG: {vars(args)} ===", flush=True)

    if not args.no_wandb:
        run_name = args.wandb_run_name or f"rank{args.rank}_bs{args.batch_size}_{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"=== W&B run: {wandb.run.url} ===", flush=True)

    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    from model import CameraCausalDiffusion  # noqa: E402

    print("=== Constructing model ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)

    print("=== Loading teacher-forcing checkpoint ===", flush=True)
    ckpt_path = args.base_checkpoint
    print(f"    base: {ckpt_path}", flush=True)
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

    print("=== Enabling gradient checkpointing ===", flush=True)
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
    print(f"=== LoRA trainable params: {n_lora:,} ({n_lora/1e6:.2f}M) ===", flush=True)

    action_encoder = ActionEncoderV2().to(device=device, dtype=torch.bfloat16)
    null_action_embedding = nn.Parameter(torch.zeros(1536, dtype=torch.bfloat16, device=device))
    n_action_params = sum(p.numel() for p in action_encoder.parameters())
    print(f"=== ActionEncoder params: {n_action_params:,} ===", flush=True)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.generator.model.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": list(action_encoder.parameters()) + [null_action_embedding], "lr": args.lr_action},
    ])

    print("=== Real text encoding (fixed prompt, cached, encoded ONCE) ===", flush=True)
    prompts = ["a robot arm pushing objects on a table"]
    with torch.no_grad():
        conditional_dict = model.text_encoder(text_prompts=prompts)
        unconditional_dict = model.text_encoder(text_prompts=[config.negative_prompt])
        unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}

    print("=== Building dataset ===", flush=True)
    dataset = BairActionLatentDataset(args.lmdb_path)
    print(f"Dataset size: {len(dataset)}", flush=True)

    print("=== Computing action normalization stats (once) ===", flush=True)
    action_mean, action_std = compute_action_stats(dataset)
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)
    print(f"action_mean={action_mean.tolist()}\naction_std={action_std.tolist()}", flush=True)

    def collate(batch):
        return {
            "clean_latent": torch.stack([b["clean_latent"] for b in batch]),
            "actions_per_latent": torch.stack([b["actions_per_latent"] for b in batch]),
        }

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.overfit_single_batch,
                         num_workers=0, collate_fn=collate, drop_last=True)

    val_loader = None
    if args.val_lmdb_path and os.path.exists(args.val_lmdb_path):
        val_ds = BairActionLatentDataset(args.val_lmdb_path)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate, drop_last=True)
        print(f"=== Validation split: {len(val_ds)} held-out samples "
              f"({args.val_batches} batches every {args.val_every} steps) ===", flush=True)

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = next(iter(loader))
        print("=== OVERFIT-SINGLE-BATCH sanity check mode ===", flush=True)

    NUM_FRAMES = dataset[0]["clean_latent"].shape[0]
    viewmats = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(args.batch_size, NUM_FRAMES, 1, 1)
    Ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device, dtype=torch.bfloat16) \
        .view(1, 1, 3, 3).repeat(args.batch_size, NUM_FRAMES, 1, 1)

    def get_batch(loader_iter):
        if fixed_batch is not None:
            return fixed_batch
        try:
            return next(loader_iter)
        except StopIteration:
            return None

    loader_iter = iter(loader)
    t_start = time.time()
    loss_history = []

    for step in range(args.max_steps):
        batch = get_batch(loader_iter)
        if batch is None:
            loader_iter = iter(loader)
            batch = get_batch(loader_iter)

        clean_latent = batch["clean_latent"].to(device=device, dtype=torch.bfloat16)  # (B,F,16,8,8)
        actions_raw = batch["actions_per_latent"].to(device)  # (B,F,16)

        # normalize (skip latent-0 rows, which are always-zero "no action" by design)
        actions_norm = (actions_raw - action_mean) / action_std
        actions_norm[:, 0, :] = 0.0  # keep the "no action" frame exactly zero after normalization too

        # action dropout -> learned null embedding, per-sample (enables CFG at inference)
        action_embed = action_encoder(actions_norm.to(torch.bfloat16))  # (B,F,1536)
        if args.action_dropout_p > 0:
            drop_mask = (torch.rand(args.batch_size, device=device) < args.action_dropout_p)
            if drop_mask.any():
                action_embed[drop_mask] = null_action_embedding.view(1, 1, -1)

        loss, log_dict = model.generator_loss(
            image_or_video_shape=[args.batch_size, NUM_FRAMES, 16, 8, 8],
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=clean_latent[:, 0:1, ...],
            viewmats=viewmats,
            Ks=Ks,
            action_embed=action_embed,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t_start
            recent = loss_history[-args.log_every:]
            avg_recent = sum(recent) / len(recent)
            print(f"[step {step}/{args.max_steps}] loss={loss.item():.4f} "
                  f"avg_recent={avg_recent:.4f} elapsed={elapsed:.0f}s", flush=True)
            if not args.no_wandb:
                wandb.log({"loss": loss.item(), "avg_recent_loss": avg_recent,
                           "elapsed_s": elapsed}, step=step)

        if (step + 1) % args.checkpoint_every == 0 or step == args.max_steps - 1:
            ckpt = {
                "step": step + 1,
                "lora_state_dict": {n: p for n, p in model.generator.model.named_parameters() if "lora_" in n},
                "action_encoder_state_dict": action_encoder.state_dict(),
                "null_action_embedding": null_action_embedding.detach(),
                "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }
            out_path = os.path.join(args.checkpoint_dir, f"step_{step+1}.pt")
            torch.save(ckpt, out_path)
            print(f"[checkpoint] saved {out_path}", flush=True)

        if val_loader is not None and ((step + 1) % args.val_every == 0 or step == args.max_steps - 1):
            # honest signal: these episodes were never trained on
            val_losses = []
            with torch.no_grad():
                for vi, vbatch in enumerate(val_loader):
                    if vi >= args.val_batches:
                        break
                    vlat = vbatch["clean_latent"].to(device=device, dtype=torch.bfloat16)
                    vact = vbatch["actions_per_latent"].to(device)
                    vn = (vact - action_mean) / action_std
                    vn[:, 0, :] = 0.0
                    vemb = action_encoder(vn.to(torch.bfloat16))  # no dropout at val time
                    vloss, _ = model.generator_loss(
                        image_or_video_shape=[args.batch_size, NUM_FRAMES, 16, 8, 8],
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        clean_latent=vlat,
                        initial_latent=vlat[:, 0:1, ...],
                        viewmats=viewmats, Ks=Ks, action_embed=vemb,
                    )
                    val_losses.append(vloss.item())
            vmean = sum(val_losses) / len(val_losses)
            train_recent = sum(loss_history[-50:]) / len(loss_history[-50:])
            print(f"[VAL step {step+1}] val_loss={vmean:.4f}  train_recent={train_recent:.4f}  "
                  f"gap={vmean - train_recent:+.4f}", flush=True)
            if not args.no_wandb:
                wandb.log({"val_loss": vmean, "train_val_gap": vmean - train_recent}, step=step)

            if not args.no_wandb:
                # visual sample: real vs. this step's x0_pred (same batch just used), for
                # the first item in the batch only -- cheap, gives visual training feedback
                with torch.no_grad():
                    def decode(latent):
                        x = model.vae.decode_to_pixel(latent[:1].to(device))
                        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()
                    real_frames = decode(log_dict["x0"])
                    pred_frames = decode(log_dict["x0_pred"])
                    n = min(real_frames.shape[0], pred_frames.shape[0])
                    grid = np.concatenate([
                        np.concatenate(list(real_frames[:n]), axis=1),
                        np.concatenate(list(pred_frames[:n]), axis=1),
                    ], axis=0)
                    wandb.log({"sample (top=real, bottom=pred)": wandb.Image(Image.fromarray(grid))}, step=step)

    if not args.no_wandb:
        wandb.finish()
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
