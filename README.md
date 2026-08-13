# Action-Conditioned Video World Model on BAIR

Teaching a pretrained video diffusion model to obey a **robot action**: given a few
frames of a scene and a commanded gripper displacement, predict the future frames —
and have the arm actually move the way it was told.

Built on [minWM](https://github.com/shengshu-ai/minWM) (Wan2.1-T2V-1.3B backbone).
The upstream project conditions on **camera pose**; conditioning on **robot action**
is the gap this work fills. The upstream README is preserved as
[`README_upstream_minWM.md`](README_upstream_minWM.md).

PSIML project, 5 days. Everything below was measured on a **held-out test split the
model never trained on**.

---

## Result

Same scene, same random noise, only the commanded action changed:

| | dx (px) | dy (px) | commanded | |
|---|---|---|---|---|
| `right` | **+7.27** | +2.74 | dx > 0 | ✅ |
| `left` | **−7.19** | +5.02 | dx < 0 | ✅ |
| `up` | +9.72 | **−6.76** | dy < 0 | ✅ |
| `down` | −1.54 | **+3.07** | dy > 0 | ✅ |

Gripper displacement measured in pixels on a 64×64 frame. The horizontal axis is
near-symmetric: a **14.5 px separation** between opposite commands, ~23% of the frame
width, from changing one number.

**Metrics on 64 held-out scenes, best checkpoint:**

| metric | value | note |
|---|---|---|
| direction accuracy (relative) | **100%** | does `right` end up right of `left`? |
| direction accuracy (absolute) | ~87% | did each command move its own way? |
| action-swap divergence | 43.8 | mean L1 between opposite-action futures (0–255 scale) |
| — same, on context frames | **0.00** | built-in control: context must be identical |
| PSNR / SSIM / FID | 18.40 / 0.779 / 27.2 | read against the VAE ceiling below |

---

## Two findings worth carrying elsewhere

**1. Control and fidelity converge on different timescales.**
Direction accuracy reaches 100% at step 1000 and then stays flat for the remaining
7000 steps, while PSNR/SSIM/FID keep improving to the very end. Neither validation
loss nor any single metric shows this — we only saw it by evaluating all 16
checkpoints.

Stated in **samples rather than steps**, because "iterations" is not portable across
batch sizes: control saturates at roughly **32k samples seen** (1000 steps × batch 32
≈ 0.74 epochs).

**2. The autoencoder, not the model, sets the blur floor.**
Decoding a *real* latent back to pixels — no generation involved — gives **22.74 dB**.
That is the hard ceiling at this resolution; our 18.40 dB is 81% of it. A good VAE at
native resolution reaches 30+ dB; ours is low because 64×64 gives an 8×8 latent, far
outside the regime the encoder was built for.

This is a quantitative argument that higher resolution would raise the *ceiling*,
rather than being cosmetic.

---

## How it works

BAIR provides **30 actions per episode**, one per frame transition. Pooling them into
a single vector is not merely suboptimal, it is **disqualifying**: the actions are
displacements, so the mean of a back-and-forth motion is ≈ 0 and the model has no
usable signal.

So conditioning is **per latent frame**, injected through the model's *existing*
per-frame timestep/AdaLN pathway rather than through the text stream:

```
actions (30×4) ── aligned to latent frames, FLATTENED not averaged ──▶ (F, 16)
                                                                        │
                            ActionEncoder  16→256→256→1536  (zero-init) │
                                                                        ▼
       e = time_embedding(t)  +  action_embed        ──▶ time_projection ──▶ AdaLN
                                                            (shift/scale/gate,
                                                             every DiT block)
```

Alignment follows the VAE's causal temporal compression: latent 0 has no preceding
action, latent *i* carries the 4 raw actions covering it. The encoder's final layer is
**zero-initialised**, so at step 0 the model behaves exactly like the pretrained
checkpoint and nothing is destroyed at training start.

Adaptation is **LoRA rank 16** on `q, k, v, ffn.0, ffn.2`. A rank sweep (8/16/64)
showed near-identical cost, so rank was chosen on quality grounds, not compute.

Total change to upstream code: **15 lines across 3 files**.

---

## Repository layout

Everything we wrote lives in [`lora_action/`](lora_action/):

**Pipeline**
| file | what it does |
|---|---|
| `extract_bair_windows.py` | BAIR TFRecord → sharded raw npz (TF venv, CPU) |
| `build_bair_lmdb.py` | VAE-encode into an LMDB matching the repo's dataset schema, plus an `actions` field |
| `bair_dataset.py` | dataset class; **the action↔latent-frame alignment lives here** |
| `train_lora_action.py` | the training loop |
| `evaluate.py` | 5 metrics, two modes, many checkpoints in one model load |

**Demos**
| file | what it does |
|---|---|
| `generate_video.py` | one clip, a chosen action |
| `generate_sequence.py` | a *different* action per latent frame within one clip |
| `rollout.py` | sliding-window free rollout, one action per block |
| `interactive_demo.ipynb` | button-driven demo; model stays resident so each press costs only sampling (~4.5 s) |

**Diagnostics** (kept as a record of how conclusions were reached)
`resolution_diagnostic.py`, `resolution_compare.py`, `cfg_test.py`, `cfg_visual.py`,
`rollout_metrics.py`, `overfit_visual_check.py`, `real_training_benchmark.py`,
`poc_action_injection.py`

---

## Reproducing

```bash
# 1. data  (~17 min for the VAE encode)
python lora_action/extract_bair_windows.py --split train --out_dir /tmp/bair_raw/train
python lora_action/build_bair_lmdb.py --in_dir /tmp/bair_raw/train --out_dir /tmp/bair_lmdb/train

# 2. sanity check FIRST -- loss should approach zero on a single batch
python lora_action/train_lora_action.py --overfit_single_batch --max_steps 300

# 3. train  (~7.6 h for 8000 steps on one A100)
python lora_action/train_lora_action.py --rank 16 --batch_size 32 --max_steps 8000 \
    --lr_lora 2e-4 --lr_action 6e-4 --checkpoint_every 500 --val_every 500

# 4. evaluate  (all checkpoints, one model load)
python lora_action/evaluate.py --n_scenes 64
```

Requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `USER`/`LOGNAME`/`HOME`
set — see [`CLAUDE.md`](CLAUDE.md) for the cluster-specific gotchas.

Peak memory is **16.7 GB** at batch 32; batch size barely affects it, since ~15 GB is
fixed model weights and activations are negligible at this token count.

---

## Known limitations

- **Free rollouts degrade.** All reported metrics use teacher-forced context: real
  frames in, one block generated. When the model consumes its *own* output block after
  block, quality falls off and per-block direction accuracy drops towards chance by ~6
  blocks. This is exposure bias, and it is exactly why minWM places self-forcing
  (Stage 2/3) *after* the teacher-forcing stage we adapted. The repo's own
  `noise_augmentation_max_timestep` is a cheap partial mitigation and is currently 0.
- **Absolute vs relative control.** Relative direction accuracy is 100%; absolute is
  ~87%, because the arm's absolute trajectory is also driven by scene dynamics the
  action does not override.
- **Classifier-free guidance does not help here.** We trained a null action embedding
  for it, swept w ∈ {1, 1.5, 2, 3}, and measured a clean fidelity-for-control trade —
  but relative direction accuracy is already saturated at 100%, so CFG amplifies
  nothing while costing fidelity and a second forward pass. Reported as a measured
  negative result.
- **FID/FVD sample counts** are far below convention (~1024 frames / 256 clips), so
  those are valid only as *relative* comparisons between our own checkpoints.
- **64×64 is the dataset's native resolution.** The original 512×640 BAIR release is
  not publicly available; the Berkeley server hosts only the 64×64 tar.

---

## Credits

Base framework: [minWM](https://github.com/shengshu-ai/minWM) (Wan2.1-T2V-1.3B).
LoRA recipe for Wan2.1 adapted from [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
— used only for *how* to inject LoRA, not as the base pipeline, since it is
bidirectional and has no autoregressive rollout.
Dataset: [BAIR robot pushing](https://sites.google.com/view/sna-visual-mpc/).

Mentors: Nedko Savov (INSAIT) and Danilo, whose questions drove several of the
measurements above.
