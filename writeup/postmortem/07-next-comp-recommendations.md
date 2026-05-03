# 07 — Recommendations for the next competition

Anchored on the public/private divergence and missed-by-11 result
in [06-final-results.md](06-final-results.md). Each recommendation
maps to a concrete change in `writeup/skill/kaggle-comp/`.

## R1 — Calibrate against private-LB proxy, not public LB alone

**Problem**: OOF→public was tight (5–10bp); public→private was wide
(50–100bp). We trusted public as the ground truth and optimized
into it.

**Fix**: a **second calibration anchor** that proxies private. Two
options, in order of confidence:

1. **Two-anchor OOF**: compute OOF under a *different* CV scheme
   (e.g. GroupKFold on a row-id hash, or repeated stratified with a
   different seed) and require gates to pass under BOTH OOFs.
   Mechanisms that overfit one fold geometry will diverge.
2. **Public-LB shake test**: across a calibration ladder of N
   submissions, compute correlation of (mechanism public lift) vs
   (mechanism magnitude on OOF). If correlation > 0.95, the public
   LB is a noiseless slice and you can chase it. If correlation <
   0.7, the public LB is noisy and you should NOT select on it.

**Maps to**: a new guardrail #12 in `skill/kaggle-comp/guardrails.md`
and a new step in the Calibration-loop.

## R2 — Final selection should hedge along the public-LB axis

**Problem**: HEDGE was an *orthogonal-mechanism* hedge (RF-natural
vs override). Both submissions overfit public LB the same way. The
actually-best private submissions (`idea5`, `W3_MHonly`,
`bagginglr`) were rejected because they regressed −2 to −44bp on
public.

**Fix**: PRIMARY = best public; HEDGE = best OOF *that regressed
on public by ≤30bp*. The hedge is explicitly designed to win when
public LB selection bias inverts on private. This costs nothing —
both slots are already used for hedging; we just hedge on the
right axis.

**Maps to**: `skill/kaggle-comp/kickoff.md` final-3-day section, and
a new walked example `examples/public-private-hedge.md`.

## R3 — Stop chasing public lift below the noise floor

**Problem**: 48 saturation events at public 0.98150 — most clustered
within ±0.00005 of each other on public, and ALL clustered within
±0.0001 on private. Days 17–18 were spent A/B-testing within the
private noise floor.

**Fix**: when daily LB result spread < 2× the public-private OOF
gap, declare the comp **converged on public** and pivot effort to:
(a) hedge selection (R2), (b) external research (existing
Research-loop), (c) a single high-variance mechanism family that
hasn't been touched. Stop testing decoration variants.

**Maps to**: extension of guardrail #4 NEVER-GIVE-UP — add a
"stop chasing public micro-lifts" sub-clause.

## R4 — Use the daily 10/day budget; record a daily-budget audit

**Problem**: 84 LB submissions over ~10 days = 8.4/day average — we
left ~16 slots on the table. With 5 of our own submissions beating
PRIMARY on private, we had not enough probes of orthogonal
candidates and too many of the same family.

**Fix**: at end-of-day audit, the Bookkeeper logs `slots_used /
slots_available`. If we sat on slots, the next-day plan must
include a low-risk probe to consume them.

**Maps to**: existing guardrail #11 already says "use the daily
10/day". Tighten the trigger: if `slots_used < 7` for two
consecutive days, mandate a Research-loop probe.

## R5 — Pre-deadline: submit the OOF-best regression candidate

**Problem**: we had `idea5_anchor_switch` sitting at public 0.98148
(−2bp vs PRIMARY) on Day 17 with the *same* 4-gate verdict. We
treated it as inferior. Private result: 0.98058 — our best.

**Fix**: in the final 3-day window, add a mandatory final probe of
the **OOF-best candidate that was rejected for public regression**.
One slot, no other variants, single-shot. If it private-LB beats
PRIMARY on the final reveal, that's the lesson; if not, we lose
one slot.

**Maps to**: final-day section of `skill/kaggle-comp/kickoff.md`.

## R6 — External research debt: what did the leader do

**Problem**: Kevin moved 0.98219 → 0.98236 on public in the last few
days. The pack moved 0.98114 → 0.98151 in 10 days. We don't know
why. That's a +0.00037 gap we can't explain.

**Fix**: post-comp, before next comp starts, spend one focused
research session on:

1. Read the top-5 winners' write-ups (they post within 1–2 weeks).
2. Read the 3–5 most-upvoted public notebooks.
3. Catalogue the mechanisms NOT in our `audit/`.
4. Promote any portable mechanism to `LEARNINGS.md`.

**Maps to**: a new "post-comp" section in
`skill/kaggle-comp/kickoff.md` and a friction-tag `research-debt`.

## R7 — On the override mechanism specifically

The override family produced our best public score and our worst
public→private gap. Override mechanisms are inherently a public-LB
overfitting risk because they target a small row count (108
flips × 80% public = ~22 public-relevant flips). Under hard rules:

- Override candidates require **all four gates PASS on a SECOND OOF
  scheme** (R1).
- Override-flip count > 200 requires **explicit PI sign-off**.
  Below 200 flips, the public-LB signal is dominated by the public
  split's row sampling.
- An override-family submission **cannot be PRIMARY**; it can only
  be HEDGE. PRIMARY must be a non-override mechanism.

**Maps to**: a new guardrail in `skill/kaggle-comp/guardrails.md`
specifically for low-flip-count override mechanisms.

## R8 — Update the framework's top-5% claim

The framework's stated target is "top 5% reliably". This comp:
top 5.24%, missed by 11 ranks of 4 315. One data point is not
evidence either way, but the framework should:

1. Track per-comp final rank and percentile in `improvements.md`.
2. After 3 comps, recompute the achieved-percentile distribution.
3. If the median is > 5%, demote the target to "top 10% reliably,
   top 5% achievable".

**Maps to**: `skill/kaggle-comp/self-improvement.md` cross-comp
metric tracking.

## Priority for the next comp

Implementation order, tightest to widest:

1. R1 (two-anchor OOF) → R2 (hedge along public-LB axis) → R5
   (final OOF-best regression probe) → R7 (override-mechanism rules)
2. R3, R4 (saturation + budget discipline) → R6 (external-research
   debt) → R8 (framework metric tracking)

Total: 1–2 days of skill edits before the next kickoff.
