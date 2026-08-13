# Status report — Day 3 (Aug 13)

Addressed to both mentors. Sections 1–2 answer Danilo's two questions directly,
since both identified real gaps; the rest is the day's results.

---

## 1. Danilo: "FID is temporally blind — add FVD"

**Agreed, and the criticism is exactly right for our project.** FID pools per-frame
Inception features, so a video with individually realistic frames but jittery or
physically wrong motion scores well. Our entire claim is about *motion* (action →
displacement), so FID is measuring the one thing we are not really testing.

**Status: not yet implemented.** It is not in `torchmetrics`, so it needs an I3D
(Kinetics-pretrained) backbone pulled in separately. Planned for day 4.

Two caveats we will state alongside the number rather than let them pass silently:
- FVD, like FID, is a distribution distance and normally uses hundreds to thousands
  of videos. We have **256 held-out episodes**, so the absolute value will not be
  comparable to published numbers.
- Standard FVD implementations resize to 224×224 for I3D. Our frames are **64×64**,
  so most of what I3D sees will be upsampling artifacts. We will report it as a
  **relative** measure between our own checkpoints only.

Same caveat already applies to our FID (~1024 frames per checkpoint), and we report
it that way.

## 2. Danilo: "Is every generation so far teacher-forced? Do we have free rollouts?"

**Correct — and this is the sharpest observation of the day.** Splitting it honestly:

**Every number we have reported is teacher-forced.** In all metric runs the context
is the *real* ground-truth latent (frames 0–3) and the model generates frames 4–7 in
a single block, with `clean_x` set to the real latent. That covers the full metric
table below, the CFG sweep, and the per-action direction results.

**We do have free autoregressive rollouts, but only qualitatively.** `rollout.py`
(written and run today) chains sliding windows: the context for block N is the
*generated* output of block N−1. Two runs today: `up up down down` (4 blocks) and
`right ×6` (6 blocks).

**What they show — and it is not flattering:**
- Latent magnitudes stay in the correct range (±3.2 vs real ±3–4) across 4 blocks,
  so there is no numerical blow-up.
- But image quality degrades progressively, and by 6 blocks the output is
  incoherent.
- Per-block commanded direction is roughly chance in free rollout (e.g. 2/6 correct
  on `right ×6`), against ~87% absolute / 100% relative in the teacher-forced regime.

**Mechanism.** The model was trained with teacher forcing, so it only ever consumed
*clean, real* context. In rollout it consumes its own output, which is a conditional
mean — slightly blurry, slightly off-distribution — and is then treated as if it were
ground truth. Error compounds multiplicatively, and blur feeds back on itself: a
blurrier context is a more ambiguous scene, so the model averages over more possible
futures, producing more blur. This is textbook exposure bias, and it is precisely why
minWM places Stage 2/3 (self-forcing) *after* Stage 1. Nedko predicted this when he
described the teacher-forcing checkpoint as "safer, but won't do long rollout".

**Two concrete fixes we identified, neither taken:**
- `noise_augmentation_max_timestep` already exists in the repo and is **0** in our
  config. Setting it > 0 noises the clean context during training so the model learns
  to tolerate imperfect context. One config line, but a 7.6h retrain.
- Stage 2 self-forcing — the proper fix, a whole additional training stage.

**The honest gap:** we have not run the metric suite *on* free rollouts. Quantifying
the degradation curve (metrics vs rollout length) is the obvious next measurement and
is cheap, since the code exists.

---

## 3. Results — full evaluation

16 checkpoints × 64 **held-out** test scenes (never trained on), 5 metrics,
24 sampling steps, identical noise across action variants.

```
 step   PSNR    SSIM     FID    div   div_ctx  dir_rel  dir_abs
  500  14.40  0.6324   43.1  39.73    0.00     98.2%    85.1%
 1000  16.04  0.6873   34.4  44.03    0.00    100.0%    88.3%
 1500  16.19  0.6981   31.4  45.53    0.00    100.0%    86.7%
 2000  15.78  0.6865   32.7  44.56    0.00    100.0%    86.7%
 2500  16.77  0.7178   30.6  44.09    0.00    100.0%    84.4%
 3000  16.67  0.7180   30.9  44.62    0.00    100.0%    89.8%
 3500  17.02  0.7307   29.3  44.61    0.00    100.0%    89.1%
 4000  17.39  0.7480   29.4  43.66    0.00    100.0%    89.1%
 4500  17.65  0.7553   28.2  46.50    0.00    100.0%    86.7%
 5000  17.47  0.7483   28.5  46.99    0.00    100.0%    88.3%
 5500  17.92  0.7621   27.9  45.36    0.00    100.0%    87.5%
 6000  18.01  0.7673   27.8  44.61    0.00    100.0%    90.6%
 6500  17.91  0.7640   28.2  44.43    0.00    100.0%    92.2%
 7000  17.83  0.7607   28.9  43.56    0.00    100.0%    88.3%
 7500  17.53  0.7530   27.9  43.48    0.00    100.0%    89.1%
 8000  18.40  0.7794   27.2  43.78    0.00     98.4%    86.7%
```

`div` = action-swap divergence: same scene, same noise, displacement dim forced to
+0.07 vs −0.07, mean L1 between the two generated futures (0–255 scale).
`div_ctx` is the same measure on the *context* frames, which must be ~0 — it is
exactly 0.00 at every checkpoint, so the divergence is not an artifact.

**Main finding: control and fidelity converge on different timescales.**
- Direction accuracy reaches 100% at step 1000 and stays flat for the remaining 7000
  steps. Divergence oscillates 43–47 with no trend.
- PSNR/SSIM/FID keep improving to the end; the final checkpoint is best on all three.
- Neither validation loss nor any single metric would have shown this.

**Control threshold stated in samples, not steps:** 1000 steps × batch 32 =
**~32,000 samples** (0.74 epochs). "2500 iterations" is not portable across batch
sizes — at batch 8 that is 20k samples, at batch 32 it is 80k. Our first run
(2500 × 8 = 20k) stopped *below* this threshold, which likely explains why it looked
weaker.

## 4. VAE ceiling — context for every PSNR number

Decoding the *real* latent back to pixels and comparing against raw frames:
**22.74 dB** (32 scenes, min 21.56, max 24.69). No generation can beat this; it is
pure autoencoder loss.

Our best is 18.40 dB = **81% of the achievable ceiling**.

A good VAE at native resolution reaches 30+ dB. Ours sits at 22.74 because we run it
far outside its design regime: 8×8 latent instead of the ~60×104 it was built for. A
visual stack (raw → VAE round-trip → generated) confirms the largest quality drop is
raw → VAE, which involves no generation at all.

**This is our first quantitative argument for higher resolution:** it raises the
*ceiling*, rather than being cosmetic.

## 5. CFG — tested, measured, rejected

We trained a null action embedding (action dropout p=0.1) specifically to enable
classifier-free guidance, then swept it:

```
   w    PSNR    SSIM     div   dir_rel  dir_abs
 1.0   19.21  0.7974   43.05    100%     87.5%
 1.5   19.01  0.7908   43.24    100%     87.5%
 2.0   18.85  0.7822   44.10    100%     89.1%
 3.0   18.33  0.7606   45.42    100%     92.2%
```

It behaves exactly as theory predicts — a clean monotonic trade of fidelity for
control, roughly 1:1. **But it is not worth it for us**, because relative direction
accuracy is already 100% at w=1.0: CFG amplifies a signal that is already saturated,
while costing fidelity and a second forward pass per step. Visually, higher w adds
colour fringing and speckle.

Reported as a measured negative result so nobody repeats it.

## 6. Architecture (unchanged since day 2, for completeness)

Action conditioning is **per latent frame**, injected through the model's existing
per-frame timestep/AdaLN pathway rather than the text stream. BAIR gives 30 actions
per episode; a single pooled action is unusable because averaging the deltas of a
back-and-forth motion cancels to zero. Actions are aligned to latent frames and
**flattened, not averaged** (16 dims per latent frame). The encoder's final layer is
zero-initialised so the model starts identical to the pretrained checkpoint.

Sign convention was verified against raw pixels with no model involved: robot-frame
+x maps to image-left (−8.5 px vs +13.2 px, n=25). Our first labels were inverted;
the model had the physics right.

## 7. Next

1. **FVD** (Danilo) — day 4.
2. **Metrics on free rollouts** — quantify the degradation curve vs rollout length.
   Cheap; the code exists.
3. **LoRA on the distilled DMD checkpoint** (Nedko's secondary question) — still
   open. Worth noting the memory premise does not hold: `model/dmd.py` holds three
   transformer copies, so real DMD training costs ~3× more, not less; plain LoRA on
   the DMD checkpoint costs the same as now. The real benefit is inference speed
   (4 denoising steps vs our 24), which is what would make an interactive demo
   comfortable rather than clunky (currently 4.5 s per 16-frame block).
4. **Presentation.**

Deliberately **not** doing: higher-resolution training. The original 512×640 BAIR is
not publicly available (the Berkeley server hosts only the 64×64 v0 tar, byte-identical
to our copy, and the tar's own arithmetic confirms 64×64). RoboNet 128 exists but mixes
four labs' robots and cameras, so it would need filtering and a per-view conditioning
flag; Bridge's full set is 441 GB. Documented as future work with the VAE-ceiling
measurement as justification.
