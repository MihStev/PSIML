# Action-Conditioned Video World Model on BAIR

Teaching a pretrained video diffusion model to obey a **robot action**: given a few
frames of a scene and a commanded gripper displacement, predict the future frames —
and have the arm actually move the way it was told.

Built on [minWM](https://github.com/shengshu-ai/minWM) (Wan2.1-T2V-1.3B backbone).
The upstream project conditions on **camera pose**; conditioning on **robot action**
is the gap this work fills. The upstream README is preserved as
[`README_upstream_minWM.md`](README_upstream_minWM.md).

PSIML project, 5 days, one A100 (40GB), shared with a second engineer working the
same GPU in parallel. Everything below was measured on a **held-out test split the
model never trained on** (256 scenes unless noted otherwise).

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

**Metrics, full 256-scene held-out test set, best checkpoint (8000 steps, rank 16),
teacher-forced context, 24-step sampling:**

| metric | value | note |
|---|---|---|
| direction accuracy (relative) | **99.6%** (255/256) | does `right` end up right of `left`? |
| direction accuracy (absolute) | 84.8% | did each command move its own way? |
| PSNR / SSIM / FID | 18.56 / 0.785 / 11.1 | read against the VAE ceiling below |
| **value of the action signal** | **+5.29 dB** | PSNR(real action) − PSNR(null action), see below |

The action-conditioning "ladder" — isolating exactly how much the action contributes,
not just whether it does something:

| condition | PSNR | reads as |
|---|---|---|
| no fine-tuning at all | 7.12 dB | pretrained backbone can't reconstruct BAIR |
| fine-tuned, **null** action (learned CFG dropout embedding) | 13.27 dB | fine-tuning alone, action signal removed |
| fine-tuned, **real** action | 18.56 dB | **+5.29 dB** — the pure value of conditioning |
| fine-tuned, **wrong** action (opposite of real) | 12.45 dB | −0.82 dB below null: actively misleading the model costs a little |

The earlier, cruder "action-swap divergence" metric (Euclidean distance between
opposite-action rollouts) is superseded by this ladder — divergence alone can't tell
"different" from "correctly different" (see the free-rollout section below, where the
same confusion showed up again and had to be caught).

---

## Two model variants: teacher-forcing base vs. DMD-distilled

minWM ships two checkpoints upstream of any fine-tuning: the Stage-1 teacher-forcing
model (~50-step diffusion sampling) and a Stage-3 DMD-distilled model (4-step
sampling, same 1.3B architecture — distillation changes *how fast the model samples*,
not its size). Both were fine-tuned with the identical LoRA + ActionEncoder recipe,
to answer an open question from the mentor: does flow-matching fine-tuning undo the
distillation that makes DMD fast?

**256-scene test set, both checkpoints, both sampling schedules:**

| base checkpoint | steps | PSNR | FID | direction accuracy (relative) |
|---|---|---|---|---|
| teacher-forcing | 4 | 18.81 | 16.80 | 100% |
| teacher-forcing | 24 | 18.56 | 11.12 | 99.6% |
| DMD-distilled | 4 | 18.46 | 16.80 | 100% |
| DMD-distilled | 24 | 18.17 | 10.95 | 99.6% |

**It does not.** The gap between the two checkpoints is inside measurement noise
(0.2–0.8%) on every metric, while the gap between 4 and 24 sampling steps is a
consistent ~25% FID cost either way. DMD's 4-step sampling remains free — same
accuracy, same rough fidelity, ~6x fewer diffusion steps — after fine-tuning. (Fewer
steps costs FID but not PSNR/SSIM/direction accuracy; the two families of metrics
disagree about what "4 steps is enough" means, which is itself worth noting.)

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
checkpoint and nothing is destroyed at training start. Action dropout (p=0.1) trains a
learned null embedding, which is what makes both the PSNR ladder above and
classifier-free guidance possible.

Adaptation is **LoRA rank 16** on `q, k, v, ffn.0, ffn.2`. A rank sweep (8/16/64)
showed near-identical cost (±0.3 GB, ±0.02 s/step at batch 8), so rank was chosen on
quality grounds, not compute — the new knowledge lives in the ActionEncoder, not in
how much LoRA capacity is available.

Total change to upstream code: **15 lines across 3 files**
(`wan_wrapper.py`, `causal_model.py`, `camera_diffusion.py` — all additive, all
default to `None` so nothing changes when the new argument isn't passed).

---

## Free rollout: what actually degrades

All the numbers above use **teacher-forced** context: real frames in, one block
(4 latent frames) generated out. A free rollout — where block *N*'s context is block
*N−1*'s own generated output, chained forward — is a materially different regime, and
the first read of it was wrong in an instructive way.

**Monte Carlo free rollout** (256 independent scenes, each with its own randomly
sampled action per block, depths 1–3):

| depth | direction accuracy (absolute) | 95% CI | FVD\* | FID |
|---|---|---|---|---|
| 1 | 78.9% | [73.5%, 83.5%] | 76.2 | 35.35 |
| 2 | 71.9% | [66.1%, 77.0%] | 89.5 | 54.31 |
| 3 | 74.2% | [68.5%, 79.2%] | 98.2 | 69.56 |

Read naively, this says "control gets worse with depth." **It doesn't — the position
does.** A second, targeted diagnostic (`control_diagnosis.py`: from the same
self-generated context, issue opposite commands and check whether the two outputs
still diverge in the *right* relative direction) shows relative direction accuracy
staying at **98–100% through all three self-generated blocks**. What the Monte Carlo
table is actually catching is that the arm's *absolute* position drifts away from
where it started — so a "move right" command still moves the arm rightward from
wherever it now is, but "wherever it now is" is no longer where the ground truth
expects. Image quality (FVD\*/FID) genuinely does degrade with depth; the model's
responsiveness to the command does not.

This matters for the story: the mechanism (**exposure bias** — the model was trained
teacher-forced and has never had to consume its own imperfect output as context) is a
known, named failure mode with known mitigations, not a sign that action-conditioning
itself is fragile.

**Scheduled sampling ablation** (fine-tuned a third checkpoint with p=0.5 probability
of feeding the model its own one-step self-prediction instead of ground truth during
training — a cheap approximation of Stage-2 self-forcing, ~10 lines, +37% time/step):

| | FVD\* growth, depth 1→3 | FID growth, depth 1→3 | direction accuracy, depth 1 |
|---|---|---|---|
| teacher-forcing only | +28.9% | +96.8% | 78.9% (single-step PSNR 17.26) |
| **+ scheduled sampling** | **+15.3%** | **+57.3%** | 73.0% (single-step PSNR 15.78, **−1.5 dB**) |

Roughly halves the rate of quality decay through a free rollout, at the cost of
1.5 dB single-step fidelity — a real trade, not a free lunch, and (surprisingly) it
does **not** change the direction-accuracy story: control was never the thing that
was breaking. Used here as a measured finding, not swapped in as the deployed model.

---

## Inverse dynamics: choosing an action by imagining outcomes

The same forward model — "what happens if I do X" — runs backwards for free:
given a context and a **goal** frame, sample a grid of candidate actions
(6×6 over the action range, one batched forward pass, shared noise), decode each
imagined future, and rank by distance to the goal. The episode's own recorded action
is the ground-truth answer, so this is checkable, not just "looks plausible."

**10 scenes with the largest true action magnitude** (`goal_action_search.py`):
sign agreement **15/20** (x: 6/10, y: 9/10) — well above chance, not a triumph. Most
of the x-axis misses are on scenes where the true `dx` is ≈0.001–0.005, i.e. the sign
itself is close to meaningless at that magnitude.

---

## Repository layout

Everything we wrote lives in [`lora_action/`](lora_action/):

**Pipeline**
| file | what it does |
|---|---|
| `extract_bair_windows.py` | BAIR TFRecord → sharded raw npz (TF venv, CPU) |
| `build_bair_lmdb.py` | VAE-encode into an LMDB matching the repo's dataset schema, plus an `actions` field |
| `bair_dataset.py` | dataset class; **the action↔latent-frame alignment lives here** |
| `train_lora_action.py` | the training loop (LoRA + ActionEncoder, W&B logging, `--overfit_single_batch` sanity mode) |
| `evaluate.py` | PSNR/SSIM/FID/direction-accuracy/delta-PSNR, two conditioning modes, `--base_checkpoint`/`--dmd_schedule` for the DMD variant, many checkpoints in one model load |

**Free-rollout & diagnostics**
| file | what it does |
|---|---|
| `rollout.py` | sliding-window free rollout, one hand-picked action per block, qualitative clips |
| `rollout_metrics.py` | free-rollout FVD\*/FID/direction-accuracy vs. depth, one fixed action per run |
| `rollout_metrics_mc.py` | the Monte Carlo version — independent random action per scene per block, confidence intervals |
| `control_diagnosis.py` | same self-generated context, opposite commands — isolates *relative* direction accuracy from position drift |
| `cfg_test.py` / `cfg_visual.py` | classifier-free guidance sweep: numeric trade-off and visual artifact comparison |
| `goal_action_search.py` | inverse dynamics: batched candidate-action search against a goal frame |

**Demos**
| file | what it does |
|---|---|
| `generate_video.py` | one clip, a chosen action |
| `generate_sequence.py` | a *different* action per latent frame within one clip |
| `interactive_demo.ipynb` | button-driven demo; model stays resident so each press costs only sampling (~4.5 s) |
| `demo_fallback_cell.py` | function-call fallback for the notebook demo when `ipywidgets` breaks (version mismatch with `ipykernel` 7.x — see `CLAUDE.md`) |
| `build_static_demo.py` / `extract_demo1_widget.py` / `web_demo.py` | self-contained HTML click-to-play demo pages (one ships inside the presentation as an embedded widget) |
| `upscale_video.py` / `real_sr_video.py` / `diffusion_sr_video.py` | three super-resolution methods compared for display only (Lanczos / Real-ESRGAN / diffusion upscaler) — **cosmetic, none feed back into any reported metric** |

**Presentation**
| file | what it does |
|---|---|
| `build_presentation_pptx.py` | generates the full pitch deck (python-pptx) — architecture diagram, controllability ladder, free-rollout cards, embedded video widget, Q&A |
| `make_search_tree_frames.py` | click-through inverse-dynamics slide frames (real search, frozen to stills) |
| `prep_psiml_logo.py` / `prep_qa_assets.py` | one-off asset prep (transparent logo, cropped headshots + QR codes) for the deck |

**Early diagnostics** (kept as a record of how conclusions were reached)
`resolution_diagnostic.py`, `resolution_compare.py`, `overfit_visual_check.py`,
`real_training_benchmark.py`, `poc_action_injection.py`

---

## Reproducing

```bash
# 1. data  (~17 min for the VAE encode)
python lora_action/extract_bair_windows.py --split train --out_dir /tmp/bair_raw/train
python lora_action/build_bair_lmdb.py --in_dir /tmp/bair_raw/train --out_dir /tmp/bair_lmdb/train

# 2. sanity check FIRST -- loss should approach zero on a single batch
python lora_action/train_lora_action.py --overfit_single_batch --max_steps 300

# 3. train  (~7.6 h for 8000 steps on one A100, batch 32)
python lora_action/train_lora_action.py --rank 16 --batch_size 32 --max_steps 8000 \
    --lr_lora 2e-4 --lr_action 6e-4 --checkpoint_every 500 --val_every 500

# 4. evaluate  (all checkpoints, one model load)
python lora_action/evaluate.py --n_scenes 256

# optional: same recipe on the DMD-distilled checkpoint instead
python lora_action/train_lora_action.py --base_checkpoint <dmd_model.pt> ...
python lora_action/evaluate.py --base_checkpoint <dmd_model.pt> --dmd_schedule --n_scenes 256

# optional: free-rollout diagnostics
python lora_action/rollout_metrics_mc.py --max_depth 3 --n_scenes 256
python lora_action/control_diagnosis.py --n_scenes 64 --max_depth 3
```

Requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `USER`/`LOGNAME`/`HOME`
set — see [`CLAUDE.md`](CLAUDE.md) for the cluster-specific gotchas (shared-GPU
scheduling, `/dev/shm` limits, storage quirks, and a full day-by-day project log).

Peak memory is **16.7 GB** at batch 32 for the training loop; batch size barely
affects it, since ~15 GB is fixed model weights and activations are negligible at this
token count (128 latent tokens per clip, vs. 31,200 at the backbone's native
resolution). `rollout_metrics_mc.py` runs closer to 22 GB and should not be run
alongside another GPU job.

---

## Known limitations

- **Free rollouts drift in position, not in obedience.** See the free-rollout section
  above — this used to be reported as "control degrades with depth," which was wrong;
  the corrected reading is that direction accuracy holds at 98–100% while absolute
  position and image quality degrade. Scheduled sampling roughly halves the quality
  decay rate at a fixed −1.5 dB single-step cost; full Stage-2 self-forcing (not
  attempted — 15–30 s/step, 30+ GPU-hours) would be the complete fix.
- **Classifier-free guidance does not help here.** A null action embedding was trained
  for it, swept w ∈ {1, 1.5, 2, 3}, and the trade is exactly as CFG theory predicts —
  but relative direction accuracy is already saturated at 100% at w=1, so CFG
  amplifies nothing while costing fidelity and a second forward pass per step.
  Reported as a measured negative result, not a bug.
- **FID/FVD\* sample counts** are far below convention (256 clips vs. thousands in the
  literature), and FVD\* uses a torchvision S3D backbone (Kinetics-400) rather than the
  canonical I3D checkpoint, evaluated at 64×64 upsampled into a 224×224-expecting
  network. Both are valid only as *relative* comparisons between our own checkpoints,
  never against published numbers.
- **The VAE, not the model, sets the blur floor.** Decoding a *real* latent back to
  pixels — no generation involved — gives **22.74 dB**. That is the hard ceiling at
  this resolution; the best reported PSNR (18.56–18.81 dB) is 81–83% of it. A good VAE
  at native resolution reaches 30+ dB; ours is low because 64×64 gives an 8×8 latent,
  far outside the regime the encoder was built for. This is a quantitative argument
  that higher resolution would raise the *ceiling*, not just look nicer.
- **64×64 is the dataset's native resolution.** The original 512×640 BAIR release is
  not publicly available; the Berkeley server hosts only the 64×64 tar. `RoboNet`
  (128×128, same lab, same robot, 3.75x more episodes) was identified as the natural
  next step but not pursued given the timeline.

---

## Presentation

`build_presentation_pptx.py` generates the full pitch deck end-to-end — architecture
diagram, the delta-PSNR ladder, an embedded click-to-play widget (real generated
clips, not screen recordings), the inverse-dynamics search walked through step by
step, and the free-rollout limitations honestly stated. Output is a standalone
`.pptx`; the companion `arm_control_panel.html` widget must sit next to it (relative
hyperlink) for the in-deck demo link to resolve.

---

## Credits

Authors: Mihajlo Stevanović & David Marković.

Base framework: [minWM](https://github.com/shengshu-ai/minWM) (Wan2.1-T2V-1.3B).
LoRA recipe for Wan2.1 adapted from [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
— used only for *how* to inject LoRA, not as the base pipeline, since it is
bidirectional and has no autoregressive rollout.
Dataset: [BAIR robot pushing](https://sites.google.com/view/sna-visual-mpc/).

Mentors: Nedko Savov (INSAIT), Danilo Đorđević, and Nataša Jovanović — several of the
measurements above (the DMD comparison, the FVD\* rollout curve, the delta-PSNR
ladder) exist because a mentor asked a question the first round of results didn't
answer.
