# 11 — Loops

Five loops, nested by frequency. Each has a trigger, an
exploration/exploitation weighting, and an exit criterion.

## Day-loop (every session)

**Trigger**: session start.
**Weighting**: 70% exploitation (run queued experiments) /
30% exploration (one new hypothesis per day minimum).
**Steps**:

1. **State load**: read `comp-context.md`, last 3 audit entries,
   `lb_status.py` output. (≤30s, no model queries.)
2. **Pick experiment**: from the Planner's queue OR a fresh
   hypothesis if no queue. Heuristic-first if mechanism is novel.
3. **Execute**: run smoke → 1-fold time-probe → 5-fold production.
4. **Evaluate**: 4-gate filter. Reviewer audit entry.
5. **Audit**: Bookkeeper writes end-of-day `audit/YYYY-MM-DD-*.md`,
   updates calibration ladder, saturation counter if applicable.
6. **End-of-day summary**: 3-bullet to the human PI.

**Exit**: PI says stop, or LB submission budget exhausted for the
day.

## Experiment-loop (per hypothesis)

**Trigger**: a new hypothesis enters the queue.
**Weighting**: pure exploitation once kicked off.
**Steps**:

1. **Heuristic baseline first**: closed-form rule, threshold,
   hand-coded baseline. Bound the lift available.
2. **Smoke**: 1 fold / 1 trial / 50k subsample. Wall-time ≤5 min.
   Catches infra bugs (sibling imports, shape mismatches, OOM,
   GPU/CUDA mismatches, output-path permissions).
3. **1-fold time-probe**: full feature set, full data, fold 0 only.
   *Trustworthy* wall-time estimate for the 5-fold production run.
   If ≥1h projected, shrink the config.
4. **5-fold production**: full run. Emit OOF + test `.npy`.
5. **4-gate filter**: G1 standalone OOF / G2 blend lift / G3
   net-rare-class-flip ratio / G4 direction asymmetry.
6. **Minimal-input meta sanity check**: train candidate meta with
   ONLY 2 components (anchor + new). If 2-comp OOF < anchor, stop.
7. **Ask-to-submit**: Reviewer + Planner agree → ask PI. PI submits
   single-shot.

**Exit**: gate failure → null entry in audit, hypothesis closed.
Gate pass + LB probe → result lands in calibration ladder.

## Calibration-loop (every N submissions)

**Trigger**: every 5 submissions OR after any negative-gap entry
(LB above OOF) OR after any leakage incident.
**Weighting**: pure exploration (we're updating priors).
**Steps**:

1. **Refresh the calibration ladder**: pull all (OOF, LB) pairs.
2. **Compute the running OOF→LB gap**: per mechanism family.
3. **Check for drift**: if any family's gap moves > 5bp from its
   trailing average, flag in CLAUDE.md.
4. **Refit blend weights** (if applicable): on the updated bank.

**Exit**: ladder + gap stats committed to `calibration_ladder.md`.

## Research-loop (every plateau, mandatory)

**Trigger**: 3 consecutive null candidates OR 5 saturation events at
the same LB OR 2 days without LB lift.
**Weighting**: 100% exploration.
**Steps**:

1. **Web search**: top public notebooks for the comp slug.
2. **Read 2 prior-comp writeups** in the same domain (synthetic
   tabular / similar metric / similar class imbalance).
3. **List untried mechanisms**: at least 5, with citation. Each
   must be NOT already in `audit/`.
4. **Rank by predicted EV × cost-to-test**.
5. **Add the top 3 to the experiment queue.**

**This is the rule the agent will most want to skip and the human
must enforce.** Every plateau-break in the irrigation comp came
from this loop, not introspection.

**Exit**: 3 new hypotheses queued, with citations.

## Weekly-loop (mid-comp)

**Trigger**: every 7 days (and at the start of the final 3-day
window).
**Weighting**: 50/50 retrospection / repointing.
**Steps**:

1. **Re-read CLAUDE.md ⚠️ rules** in full.
2. **Audit ceiling thesis**: is "structural ceiling" being claimed?
   If yes, demand 3 untried mechanisms (Research-loop).
3. **Rotate at least one persona**: invoke one Layer-B persona on
   a stuck problem.
4. **Update REPORT.md** with the week's results.
5. **Submission-budget audit**: are we using the daily 10/day? If
   we sat on slots, why?

**Exit**: a 5-line weekly summary committed to `audit/`.

## Loop interaction

Day-loop wraps Experiment-loop wraps the smoke / 1-fold / 5-fold /
gate sub-steps. Calibration-loop runs *across* day boundaries
(triggered by submission count). Research-loop interrupts the
day-loop on plateau detection. Weekly-loop is a calendar-driven
audit.

The agent's default failure mode is to stay in the Experiment-loop
indefinitely, ignoring the Research-loop trigger. This is what the
human PI watches for.
