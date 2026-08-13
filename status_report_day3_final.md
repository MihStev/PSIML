# Status Report — Day 3 (Aug 13)

## Headline

- Training finished (8000 steps, 5.9 epochs). Full evaluation run: **5 metrics × 16 checkpoints × 64 held-out scenes.**
- **Main finding: control and fidelity converge on different timescales.** Direction accuracy hits 100% at step 1000 and stays flat; PSNR/SSIM/FID keep improving all the way to step 8000. The last 7000 steps bought fidelity, not control.
- Both of Danilo's points were real gaps. Both are now answered with measurements, and one of them changed how we report our results.
- We also measured the ceiling we are working against: the **VAE alone caps us at 22.74 dB**, and we are at 81% of it.

## Danilo's Two Points — What We Found

**"FID is temporally blind — add FVD."**
- Agreed, and it is the right criticism for this project specifically: FID pools per-frame features, so a video with realistic frames but physically wrong motion scores well. Motion is the only thing we actually claim.
- Implemented and used (see the rollout table below). Caveat we state up front: canonical FVD needs a specific Kinetics I3D checkpoint that is not installable in our environment, so we use torchvision's **S3D** (also Kinetics-400). We label it **FVD\***, not FVD — with 256 clips at 64×64 fed to a network expecting 224×224, the absolute value is not comparable to published numbers. Valid as a **relative** measure between our own conditions only. The same caveat applies to our FID.
- It earned its place immediately: FVD\* rises **+48%** from rollout depth 1→2 while FID rises +38%, i.e. temporal incoherence appears before per-frame quality collapses.

**"Is every generation so far teacher-forced? Do we have free rollouts?"**
- **Correct — every number we had reported was teacher-forced.** Context is the real ground-truth latent, one block generated, `clean_x` set to the real latent.
- We did have free rollouts (model consumes its own output block after block) but only qualitatively. We have now measured them properly:

```
depth  frames    FVD*     FID    divergence   direction
    1      17   106.4    99.6        48.19       96.9%
    2      33   157.6   137.9        52.84       53.1%
    3      49   169.8   154.6        55.68       68.8%
    4      65   176.1   167.6        58.54       62.5%
    5      81   176.7   179.6        61.01       59.4%
    6      97   183.5   183.8        63.22       68.8%
```

- **Depth 1 (96.9%) independently reproduces our teacher-forced result (100% on 64 scenes)** — the two measurements agree, so this is not a different methodology producing different numbers.
- **Control collapses after exactly one self-generated block**: 96.9% → 53.1%, essentially chance. Depths 3–6 hover at 59–69%, all statistically indistinguishable from each other (±17% at n=32).
- Real BAIR episodes are only 30 frames, so beyond depth 1 there is no paired ground truth and PSNR/SSIM are undefined. Instead, at every depth we compare the **most recently generated 16-frame window** against real 16-frame windows — the same question asked identically at each depth.

**Mechanism.** The model only ever saw *clean, real* context during training. In rollout it consumes its own output — a conditional mean, slightly blurry and off-distribution — treated as if it were ground truth. Error compounds multiplicatively, and blur feeds back on itself: a blurrier context is a more ambiguous scene, so the model averages over more futures and produces more blur. This is exposure bias, and it is exactly why minWM places self-forcing (Stage 2/3) *after* the teacher-forcing stage we adapted. Nedko predicted this when recommending Stage 1 as "safer, but won't do long rollout."

**Two fixes identified, neither taken:** the repo's own `noise_augmentation_max_timestep` (currently 0 in our config — one line, but a 7.6h retrain), and Stage 2 self-forcing (a whole additional training stage).

## Full Evaluation — 16 checkpoints × 64 held-out scenes

```
 step   PSNR    SSIM     FID    div   div_ctx  dir_rel  dir_abs
  500  14.40  0.6324   43.1  39.73    0.00     98.2%    85.1%
 1000  16.04  0.6873   34.4  44.03    0.00    100.0%    88.3%
 2000  15.78  0.6865   32.7  44.56    0.00    100.0%    86.7%
 4000  17.39  0.7480   29.4  43.66    0.00    100.0%    89.1%
 6000  18.01  0.7673   27.8  44.61    0.00    100.0%    90.6%
 8000  18.40  0.7794   27.2  43.78    0.00     98.4%    86.7%
```
(abridged; full 16-row table in the repo)

- `div` = action-swap divergence: same scene, same noise, displacement dim forced ±0.07, mean L1 between the two generated futures on a 0–255 scale.
- `div_ctx` is the same measure on the **context** frames, which must be ~0. It is exactly **0.00 at every checkpoint** — a built-in control confirming the divergence is not an artifact.
- **Control threshold stated in samples, not steps**: 1000 steps × batch 32 ≈ **32,000 samples** (0.74 epochs). "2500 iterations" is not portable across batch sizes — at batch 8 that is 20k samples, at batch 32 it is 80k. Our first run (2500 × 8 = 20k) stopped *below* this threshold, which likely explains why it looked weaker.

## The Ceiling We Are Working Against

Decoding a **real** latent back to pixels — no generation at all — gives **22.74 dB**. That is the hard ceiling at this resolution; our best is 18.40 dB = **81% of it**.

A good VAE at native resolution reaches 30+ dB. Ours is low because 64×64 yields an 8×8 latent, far outside the regime the encoder was built for. A visual stack (raw pixels → VAE round-trip → generated) confirms the **largest quality drop is raw → VAE**, which involves no generation.

This is our first quantitative argument that higher resolution would raise the *ceiling* rather than being cosmetic.

## Classifier-Free Guidance — Tested and Rejected

We trained a null action embedding (action dropout p=0.1) specifically to enable CFG, then swept it:

```
   w    PSNR    SSIM   divergence   dir_rel   dir_abs
 1.0   19.21  0.7974      43.05      100%      87.5%
 2.0   18.85  0.7822      44.10      100%      89.1%
 3.0   18.33  0.7606      45.42      100%      92.2%
```

It behaves exactly as theory predicts — a clean fidelity-for-control trade, roughly 1:1 — **but it is not worth it here**, because relative direction accuracy is already saturated at 100%. CFG amplifies a signal with no headroom while costing fidelity and a second forward pass per step. Visually it adds colour fringing. Reported as a measured negative result.

## Problems and Corrections

- **Survivorship bias in our own metric.** Our direction test tracks the gripper by colour; on badly degraded rollouts the tracker fails and those scenes were being *excluded* rather than counted as failures. Sample counts fell from 32 to 24 at depth 5, making deep rollouts look better than they are. The table above is **corrected** (untracked = failure); the uncorrected numbers would have read 71/71/79/85% at depths 3–6, implying a recovery that is not real.
- **Divergence alone is misleading.** Across rollout depth, divergence *rises* (48 → 63) while direction accuracy *falls* (97% → ~60%). The two rollouts diverge more, but not in a controlled way — they are both drifting independently. Divergence measures "different"; direction measures "correctly different". Tracking only divergence would have suggested control *strengthens* with depth, the opposite of the truth.
- **A demo-script ablation was inconclusive and we discarded it.** Testing whether freezing two action dims weakened control gave 83% vs 42% — but the same configuration re-run with different noise gave 42% vs 83%. No fixed seed, 12 binary measurements, one scene. The reliable number remains the 64-scene evaluation.

## Also Working

- **Interactive demo** (Jupyter, button-driven): the model stays resident in the kernel, so each press costs only sampling — ~4.5 s per 16-frame block. Free rollout with one action per press.
- Action *programs*: a different action per latent frame within one clip, which the pooled-action design we rejected could not express even in principle.

## Next

1. **LoRA on the distilled (DMD) checkpoint** — Nedko's secondary question, planned overnight. Worth noting the memory premise does not hold: `model/dmd.py` holds three transformer copies, so true DMD training costs ~3× more, not less; plain LoRA on the DMD checkpoint costs the same as now. The real benefit is inference speed (4 denoising steps vs our 24), which is what would turn the interactive demo from clunky into genuinely real-time.
2. Presentation.

**Deliberately not doing:** higher-resolution training. The original 512×640 BAIR is not publicly available — the Berkeley server hosts only the 64×64 tar, byte-identical to our copy, and the tar's own arithmetic confirms 64×64. RoboNet 128 exists but mixes four labs' robots and cameras, so it needs filtering and a per-view conditioning flag; Bridge's full set is 441 GB. Documented as future work, with the VAE-ceiling measurement as justification.
