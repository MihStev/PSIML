# Status report — Day 3 (Aug 13)

Addressed to both mentors. Sections 1–2 answer Danilo's two questions directly,
since both identified real gaps; the rest is the day's results.

---

## 1. Danilo: "FID is temporally blind — add FVD"

**Agreed, and the criticism is exactly right for our project.** FID pools per-frame
Inception features, so a video with individually realistic frames but jittery or
physically wrong motion scores well. Our entire claim is about *motion* (action →
displacement), so FID is measuring the one thing we are not really testing.

**Status: implemented and used the same day** — results in section 2 below, where it
immediately earned its place.

Implementation note and its limits, stated up front rather than buried: canonical FVD
uses a specific Kinetics-pretrained I3D checkpoint that we could not install in this
environment (no package available through our proxy), so we use torchvision's **S3D** —
also Kinetics-400, architecturally the successor to I3D — and take the Fréchet distance
over its 1024-dim features. We therefore label it **FVD\***, not FVD:

- Different backbone, so not numerically comparable to published FVD.
- A distribution distance normally computed over hundreds to thousands of videos; we
  have **256 held-out episodes**.
- The network expects 224×224 and our frames are **64×64**, so much of what it sees is
  upsampling artifact.

**Valid as a relative measure between our own conditions only.** The same caveat
already applies to our FID (~1024 frames per checkpoint) and we report it that way.

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

**Update — measured (Aug 13, later the same day).** `rollout_metrics.py`: same free
rollout as above, but now scored with FVD* (Fréchet distance over torchvision S3D
features — a Kinetics-400 backbone, not the canonical I3D checkpoint, so labelled
`FVD*` and read only as a relative measure across our own depths, same caveat as our
FID) plus FID, action-swap divergence, and direction accuracy, all as a function of
rollout depth. 32 held-out scenes, depths 1–6 (17–97 frames), checkpoint step 8000.

```
depth  frames    FVD*     FID     div  dir_rel  n_dir
    1      17   106.4   99.57   48.19   0.969     32
    2      33   157.6  137.90   52.84   0.531     32
    3      49   169.8  154.63   55.68   0.710     31
    4      65   176.1  167.56   58.54   0.714     28
    5      81   176.7  179.60   61.01   0.792     24
    6      97   183.5  183.83   63.22   0.846     26
```

**Headline finding — quantifies what was qualitative before.** FVD* and FID both rise
monotonically with depth (FVD* 106→184, FID 100→184, roughly doubling from depth 1 to
6). This is a clean, unambiguous confirmation of the exposure-bias mechanism described
above: quality degrades steadily as the model consumes more of its own output. This is
our answer to Danilo's question #2, with numbers attached.

**One result we are flagging, not claiming.** Direction accuracy is *not* a clean
decay: it drops sharply at depth 2 (96.9%→53.1%, near chance) then partially recovers
to 84.6% by depth 6. We are not treating this as a real "recovery" effect — `n_dir`
falls from 32 to 24-26 as depth increases (the pixel-colour arm detector fails to find
the arm more often in the blurrier later frames, shrinking and biasing the sample), and
this is a single run at one seed, not averaged over repeats. Flagged honestly rather
than either oversold or discarded.

**Resolved without a new run.** `n_dir` does not shrink at random: scenes where the
detector fails were being *excluded* from the average rather than counted as failures.
That is survivorship bias — the worst rollouts leave the sample, so the survivors'
average rises. Counting untracked as failure (always /32, which is the honest reading:
the image degraded until the arm is not visible, and that *is* a control failure):

```
depth 1: 96.9%    depth 3: 68.8%    depth 5: 59.4%
depth 2: 53.1%    depth 4: 62.5%    depth 6: 68.8%
```

The apparent recovery disappears. What remains: a sharp drop after the first
self-generated block, then flat at 59–69% — above chance (50%) but far below the
initial 97%. At n=32 the confidence intervals are ±17%, so depths 2–6 are mutually
indistinguishable; only depth 1 versus the rest is solid. The fix in the script is one
line (count NaN as failure instead of skipping); until then the corrected table above
is the one to quote, not the raw JSON.

---

## 2b. Two methodological lessons from the above

**FVD\* earns its place immediately.** From depth 1→2 it rises **+48%** while FID rises
+38%, and it saturates around depth 4–5 (176→177→184) while FID keeps climbing
(168→180→184). Temporal incoherence appears *before* per-frame quality collapses —
which is exactly the argument for adding it, now with numbers.

**Divergence alone is misleading.** Across rollout depth, divergence *rises*
(48→63) while direction accuracy *falls* (97%→~60%). The two rollouts do diverge more,
but not in a controlled way — both are drifting independently. Divergence measures
"different"; direction measures "correctly different". Had we tracked only divergence,
we would have concluded that control *strengthens* with depth, the opposite of the
truth. Worth stating because it generalises to anyone measuring controllability.

**A demo-script ablation was inconclusive and discarded.** Testing whether freezing two
action dims weakened control gave 83% vs 42% — but re-running the *same* configuration
with different noise gave 42% vs 83%. No fixed seed, 12 binary measurements, one scene.
The reliable number remains the 64-scene evaluation.

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

## 7. Where we would value your opinion

We have roughly **1.5 days left**, of which the presentation needs a solid half. That
realistically leaves **one substantial experiment**, plus one cheap one. We have a
default in mind but would rather hear you first — all four options are prepared and
could start tonight.

**A. More statistics on what we already have** — *cheap, ~30 min*
Our per-scene results (rollout depth, direction accuracy) come from a single seed at
n=32, with ±17% confidence intervals. Multiple seeds and proper error bars would let us
state which of our differences are real. Also fixes the one-line survivorship-bias
issue in §2. Low risk, strengthens claims we have already made rather than adding new
ones.

**B. LoRA on the distilled (DMD) checkpoint** — *~4.5 h, runs overnight*
Nedko's secondary question, still open. One correction to the original framing: the
memory premise does not hold — `model/dmd.py` holds three transformer copies, so true
DMD training costs ~3× more, not less, and plain LoRA on the DMD checkpoint costs the
same as what we already run. The real benefit is **inference speed**: 4 denoising steps
instead of our 24, which is what would turn the interactive demo from clunky (4.5 s per
press) into genuinely real-time.
Its risk is also its interest: the model was distilled to *skip* steps, and our
flow-matching loss teaches it to predict the true flow again — i.e. partially undoing
the distillation. All three outcomes (4-step still works / distillation undone /
unstable) are reportable findings, so this is a measurement rather than a gamble.

**C. Retrain with `noise_augmentation_max_timestep > 0`** — *~7.6 h, runs overnight*
The repo's built-in mitigation for exactly the exposure bias we measured in §2; it is
currently 0 in our config. One config line. Would plausibly extend usable rollout depth
beyond one block, but by an unknown amount, and it costs the same night as B.

**D. Stop experimenting and put everything into the presentation.**
We already have a complete result: action conditioning verified on held-out data in
both axes, a full metric sweep, the VAE ceiling, and a measured degradation curve.
Nothing below depends on further runs.

**Our default if we do not hear otherwise: A tonight (cheap, tightens existing claims),
then B overnight, then D tomorrow.** We are choosing B over C because it answers a
question you posed and produces a demo asset, whereas C improves a secondary property
we are already reporting honestly as a limitation.

**One thing we are deliberately not doing:** higher-resolution training. The original
512×640 BAIR is not publicly available — the Berkeley server hosts only the 64×64 v0
tar, byte-identical to our copy, and the tar's own arithmetic confirms 64×64. RoboNet
128 exists but mixes four labs' robots and cameras, so it would need filtering and a
per-view conditioning flag; Bridge's full set is 441 GB. Documented as future work,
with the VAE-ceiling measurement (§4) as the quantitative justification.

**Also open, if you have a view:** for the 10-minute presentation, is the more useful
headline (a) action conditioning works in 2D on unseen scenes, (b) control and fidelity
converge on *different* timescales — control saturates at ~32k samples while fidelity
improves throughout, or (c) the autoencoder, not the model, sets the quality ceiling at
this resolution? We lean towards (b) as the least obvious and most transferable, with
(a) as the foundation and (c) as the bridge to future work.
