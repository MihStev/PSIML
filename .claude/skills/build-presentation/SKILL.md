---
name: build-presentation
description: "Structure and content guidance for building the presentation/pitch of the BAIR LoRA action-conditioning world model project, based on mentor Danilo Djordjevic's framing (14.08 Discord message). Use when drafting presentation slides, deciding what to include/cut, prioritizing remaining work before a deadline, or picking up his two stretch-goal ideas (scheduled sampling training, goal-conditioned action search demo)."
license: MIT
---

# build-presentation

Captures Danilo Djordjevic's (mentor) presentation guidance so it survives context
compaction/new sessions. Source: Discord message, 14.08, sent because he wasn't available for
a call. Use this skill when the task is "build slides" / "what goes in the presentation" /
"what's left before we can present" for the BAIR action-LoRA project.

## Priority (his words)

> Priority is I would say now to make a MVP-style presentation. Then, as a stretch goal, try to
> improve the training by introducing a longer context into training.

Read literally: **finish the presentation before touching the stretch goal.** If both are
in flight near a deadline, the presentation wins. See project CLAUDE.md, "PLAN ZA DAN 4 i 5" —
the team already encoded this as a hard rule: whatever isn't done by day 4 18:00 gets cut, the
presentation is the one thing that must not be sacrificed.

## Slide structure (his structure, use in this order)

1. **Motivation** — world models let robots imagine the consequences of their actions. Video
   models produce future frames conditioned on current frame + action → main question: can we
   use them for action *control*?
2. **Goals** — finetuning a pretrained video model with LoRA (+ the other stated goals of the
   project).
3. **Method** — describe the approach starting from the model, its inputs/outputs, the dataset
   it was finetuned on, how action embeddings were built, how video generation is conditioned on
   them. **These two (how are embeddings built, how is conditioning done) are the main questions
   a person will ask during the talk** — don't gloss over them.
4. **Results** — lead with the WOW result first, then fidelity/how it improves. Show the numbers,
   all metrics. **Prerender good demos — cannot be stressed enough.** Do not rely on live
   inference during the talk.
5. **Demo videos** of next-frame prediction. **End on limitations and bad cases** (e.g.
   autoregressive rollout degrading) — don't hide them, they belong in the talk.

For this project specifically, pull current numbers/plots from `CLAUDE.md` in the repo root —
it is the running log of every measured result (control accuracy, PSNR/SSIM/FID curves, VAE
ceiling, delta-PSNR ladder, MC rollout degradation curve, DMD comparison, etc.). Don't
hardcode numbers into this skill — they keep changing as new evals land; treat CLAUDE.md as the
source of truth and re-read it fresh each time this skill is used.

The "WOW result" to lead Results with, per that log: direction control accuracy (`dir_rel`)
saturates near 100% early and holds through training — that is the headline number, not the
fidelity metrics (PSNR/SSIM/FID), which are the secondary "how good does it look" story.

## Stretch goal: scheduled sampling (longer-context training)

His description, verbatim intent: current training is purely teacher-forced, samples like
`(x_t, x_t+1)` where `x_t` is real ground truth input. Introduce a scheme where, with some
probability `p`, each sample instead uses `(x_t_pred, x_t+1)` — i.e. the model's own previous
prediction as input instead of ground truth. Goal: model learns to predict correctly even when
its input is worse than GT, which should improve autoregressive multi-step (rollout) quality —
directly targets the exposure-bias failure mode already measured in the free-rollout eval.

Implementation note already validated in this project (see CLAUDE.md "SCHEDULED SAMPLING"): this
does NOT require the repo's DMD/self-forcing distillation machinery (three-transformer setup) —
it's a ~10 line change in the training loop's own sampling step, single transformer. Cost
measured: +37% time/step over plain teacher forcing (one extra no-grad forward pass), because
4-step context generation is cheap. Check CLAUDE.md for current status/results before restarting
this from scratch — it may already be past the smoke-test stage.

## Demo idea A: goal-conditioned action search (forward vs. inverse dynamics)

Danilo's framing, his own words: **"two views of the same model: forward vs inverse dynamics."**
Given a start frame and a goal frame, sample a grid of candidate actions, predict one block ahead
for each candidate (batched, same noise), and rank candidates by similarity (L1/SSIM/FID) to the
goal frame. The episode's own recorded future is a fair ground truth — its true action is a
verifiable answer, not just "looks plausible." If already implemented, check CLAUDE.md for
results before re-deriving the approach.

## Demo idea B: action-arrows visualization

Simpler, cheaper demo: visualize different actions as arrows pointing in different directions,
and show the model's predicted next frame corresponding to each arrow, from the same starting
context. This is the one to prerender for the talk per priority #4 above — it directly shows
control without requiring the viewer to understand the goal-search ranking mechanism.

## When invoked

Read the current state of `CLAUDE.md` first (results, checkpoint status, what's done vs. open)
before drafting slide content or deciding what fits the MVP cut — this skill is the *structure*
Danilo asked for, not a snapshot of the numbers.
