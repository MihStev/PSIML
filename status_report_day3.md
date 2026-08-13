# Status Report — Day 3 (Aug 13)

## Headline

- Training finished (8000 steps, 5.9 epochs) and fully evaluated: 5 metrics × 16 checkpoints × 64 held-out scenes.
- Main finding: **control and fidelity converge on different timescales.** Direction accuracy hits 100% at step 1000 and stays flat; PSNR/SSIM/FID keep improving to step 8000. The last 7000 steps bought fidelity, not control.
- Both of your points were real gaps. Both are now answered with measurements, and one of them changed how we report results.

## Your Two Points — What We Found

**"FID is temporally blind — add FVD."**
- Agreed, and the right criticism here specifically: FID pools per-frame features, so realistic frames with wrong motion still score well. Motion is the only thing we claim.
- Implemented and used the same day. Canonical FVD needs a Kinetics I3D checkpoint we could not install, so we use torchvision S3D (also Kinetics-400) and label it FVD*.
- Not comparable to published numbers — different backbone, 256 clips, 64×64 frames fed to a network expecting 224×224. Relative measure between our own conditions only. Same caveat as our FID.
- Earned its place immediately: FVD* rises +48% from rollout depth 1→2 while FID rises +38%, then saturates while FID keeps climbing. Temporal incoherence appears before per-frame quality collapses.

**"Is every generation teacher-forced? Do we have free rollouts?"**
- Correct — every number we had reported was teacher-forced: real context in, one block generated.
- We did have free rollouts but only qualitatively. Now measured properly, 32 held-out scenes, depths 1–6:

```
depth  frames    FVD*     FID     div   direction
    1      17   106.4    99.6   48.19      96.9%
    2      33   157.6   137.9   52.84      53.1%
    3      49   169.8   154.6   55.68      68.8%
    4      65   176.1   167.6   58.54      62.5%
    5      81   176.7   179.6   61.01      59.4%
    6      97   183.5   183.8   63.22      68.8%
```

- Depth 1 (96.9%) independently reproduces the teacher-forced result (100% on 64 scenes) — the two methods agree.
- Control collapses after exactly one self-generated block: 96.9% → 53.1%, essentially chance. Depths 3–6 sit at 59–69%, mutually indistinguishable (±17% at n=32).
- Quality degrades monotonically: FVD* and FID both roughly double from depth 1 to 6.
- Mechanism is exposure bias — the model only ever saw clean real context, so consuming its own slightly-blurry output compounds error, and blur feeds back on itself. This is why minWM places Stage 2 self-forcing after the stage we adapted.

## What Was Measured

- Full evaluation, best checkpoint (step 8000): PSNR 18.40, SSIM 0.779, FID 27.2, divergence 43.8, direction 100% relative / ~87% absolute.
- Built-in control: the same divergence measured on context frames is exactly 0.00 at every checkpoint, confirming it is not an artifact.
- Control threshold in samples, not steps: saturates at ~32,000 samples (1000 steps × batch 32). "2500 iterations" is not portable across batch sizes — our first run stopped at 20k samples, below this threshold, which likely explains why it looked weaker.
- VAE ceiling: decoding a real latent to pixels — no generation — gives 22.74 dB. Our 18.40 dB is 81% of that. A good VAE at native resolution reaches 30+ dB; ours is low because 64×64 yields an 8×8 latent. Most of the visible blur is the autoencoder, not the model.
- CFG tested and rejected: swept w ∈ {1, 1.5, 2, 3} using the null action embedding we trained for it. Clean fidelity-for-control trade, roughly 1:1 — but relative direction accuracy is already 100% at w=1, so it amplifies a saturated signal while costing fidelity and a second forward pass. Measured negative result.

## Problems We Hit (and What They Taught Us)

- Survivorship bias in our own metric: the colour-based arm detector fails on degraded rollouts, and those scenes were excluded rather than counted as failures. Sample counts fell 32 → 24 by depth 5, and the raw numbers implied a recovery to 85% that does not exist. Table above is corrected (untracked = failure).
- Divergence alone is misleading: across rollout depth it rises (48 → 63) while direction accuracy falls (97% → ~60%). Both rollouts drift independently. Divergence measures "different"; direction measures "correctly different". Tracking only divergence would have suggested control strengthens with depth.
- An ablation we discarded: testing whether freezing two action dims weakened control gave 83% vs 42% — but re-running the same configuration with different noise gave 42% vs 83%. No fixed seed, 12 binary measurements, one scene. The 64-scene evaluation remains the reliable number.
- Second camera view checked and correctly discarded: the arm occupies 3.4% of pixels versus 7.5% in the main view, and part of the frame is blown out.

## Also Working

- Interactive demo (Jupyter, button-driven): model stays resident in the kernel, so each press costs only sampling — ~4.5 s per 16-frame block, one action per press.
- Action programs: a different action per latent frame within one clip — something the pooled-action design we rejected could not express even in principle.

## Currently Running

- Nothing. Option A (below) ran tonight; GPU is free pending your input on B/C.

## Option A — Done: 256-Trial Monte Carlo, Randomized Per-Block Actions

Ran the statistics option below the same night. Instead of one shared command per batch
(e.g. "everyone goes right"), every one of 256 held-out scenes now draws its OWN random
action per block (up/down/left/right, i.i.d.), depths 1–3. Motivated by two earlier ad-hoc
clips (up-up-down-down: 50% landed; right×6: 33% landed) hinting the specific command might
matter — this tests that properly instead of anecdotally.

```
depth    n    FVD*    FID   direction (abs.)      95% CI       PSNR    SSIM
    1  256    76.2   35.35        78.9%      [73.5%,83.5%]    17.26   0.7734
    2  256    89.5   54.31        71.9%      [66.1%,77.0%]      n/a     n/a
    3  256    98.2   69.56        74.2%      [68.5%,79.2%]      n/a     n/a
```

PSNR/SSIM only at depth 1 — undefined beyond it, no stored ground truth (BAIR episodes are
30 raw frames).

**A real bug the smoke test caught first:** the first version scored the generated block
(under a random command) against the real future (which followed the scene's actual,
different action) — apples to oranges, gave PSNR 11.2. Fixed by running a second, separate
generation at depth 1 under the scene's *true recorded* action, used only for PSNR/SSIM.
After the fix: 17.26 / 0.773, in line with the 64-scene evaluation (18.40 / 0.779).

**Note this `direction accuracy` is not the same statistic as the `direction` column in the
depth table above** — that one was a paired/relative comparison (easier); this one is
absolute, per-trial (harder, matches `dir_abs` in the full evaluation, historically 84–92%).
79% at depth 1 is consistent with that, not a regression from the 97% figure above.

**Why FVD*/FID are lower here than in the 32-scene run at the same depth** (76.2 vs 106.4;
35.35 vs 99.57): Fréchet distance needs its covariance estimated in a 2048-d (FID) / 1024-d
(FVD*) feature space. 512 frames (32 scenes) under-samples that badly, a known FID failure
mode; 4096 frames (256 scenes) is far better conditioned. The earlier number was likely
inflated by sample size, not a different result.

**Reading the curve:** the 1→2 drop looks real (CIs barely touch); 2 vs 3 (71.9% vs 74.2%)
overlap widely and are not distinguishable — we are treating this as noise, not recovery.
This confirms, on much firmer footing, the earlier finding: control takes its hit after
exactly one self-generated block, then holds flat while FVD*/FID keep climbing (89.5→98.2,
54.31→69.56) — the same "two different timescales" result, independently replicated.

Script: `lora_action/rollout_metrics_mc.py`. Raw output: `/home/mls10/logs/rollout_metrics_mc.json`.

## Where We Would Value Your Opinion

A is done (above). B and C are still open.

- **B. LoRA on the distilled (DMD) checkpoint.** Your secondary question, still open. One correction: the memory premise does not hold — model/dmd.py holds three transformer copies, so true DMD training costs ~3× more, and plain LoRA on the DMD checkpoint costs the same as what we run now. The real benefit is inference speed (4 denoising steps vs our 24), which is what would make the interactive demo genuinely real-time. Its risk is also its interest: the model was distilled to skip steps and our loss teaches it to predict the true flow again, partially undoing that. All three outcomes are reportable.
- **C. Stop experimenting and consolidate.** The result is already complete: action conditioning verified on held-out data in both axes, full metric sweep, VAE ceiling, measured degradation curve.

Our inclination now is B if there's time, otherwise C, but we would rather follow your judgement. The presentation still has to be built, so whatever runs has to leave room for it.

## Not Doing

- Higher-resolution training. The original 512×640 BAIR is not publicly available — the Berkeley server hosts only the 64×64 tar, byte-identical to our copy, and the tar's own arithmetic confirms 64×64. RoboNet 128 mixes four labs' robots and cameras, so it would need filtering and a per-view conditioning flag; Bridge's full set is 441 GB. Future work, with the VAE ceiling as the quantitative justification.
