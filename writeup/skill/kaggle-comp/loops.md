# Loops — agent-instruction form

Five loops. The agent runs Day-loop → Experiment-loop nested. The
human triggers Calibration / Research / Weekly when conditions hit.

## Day-loop

**Trigger**: every session start.
**Stop**: PI says stop, or daily LB budget exhausted.

```
1. Load state (Haiku): comp-context.md, last 3 audit/, lb_status.py
2. Pick experiment (Sonnet): from queue or new hypothesis
                             (heuristic-first if novel)
3. Execute Experiment-loop (see below)
4. End-of-day audit: write audit/YYYY-MM-DD-<topic>.md, update
                     calibration ladder, 3-bullet PI summary
5. Queue next session's first 3 experiments
```

## Experiment-loop

**Trigger**: an experiment is selected.

```
1. Heuristic baseline (skip if N/A):
   - closed-form rule / threshold / hand-coded
   - bound the lift available
   - if heuristic already clears the candidate's predicted lift,
     stop here

2. Smoke (5-min cap):
   - 1 fold / 1 trial / 50k subsample
   - verify: COMPLETE status, expected artifacts emitted, no shape
     mismatch, no permission errors
   - if smoke fails, fix and re-smoke; do NOT push to production

3. 1-fold time-probe:
   - full feature set, full data, fold 0
   - measure wall time on same hardware as production
   - if 5-fold projection ≥ 1h, shrink config (folds, n_ens, epochs,
     subsample)

4. 5-fold production:
   - emit OOF + test .npy
   - compute candidate OOF on metric

5. 4-gate filter:
   - G1 standalone OOF clears anchor at recipe-bias
   - G2 blend with anchor lifts at α* > 0
   - G3 net rare-class-flip ratio ≥ 0.5
   - G4 direction asymmetry: more correct flips than incorrect

6. Minimal-input meta sanity check (if stacking candidate):
   - train candidate meta with ONLY 2 components (anchor + new)
   - if 2-comp OOF < anchor at recipe-bias, STOP. Do not LB-probe.

7. Reviewer audit:
   - subagent invocation, no parent-context anchoring
   - emit gate-result audit entry

8. Ask PI to submit:
   - present: candidate name, OOF, gate results, predicted LB lift,
     remaining slots today
   - PI confirms → single-shot kaggle competitions submit
   - log result, update calibration ladder
```

**Stop condition**: any gate fails, or PI declines, or LB result
lands.

## Calibration-loop

**Trigger**: every 5 LB submissions, OR after any negative-gap
entry > 5bp (LB above OOF), OR after any leakage incident.

```
1. Refresh calibration ladder (Haiku): parse all (OOF, LB) pairs,
   compute per-mechanism-family gap
2. Drift check: if any family's gap moves > 5bp from its trailing
   average, flag in CLAUDE.md and pause new submissions in that
   family
3. Refit blend weights if applicable (Sonnet) + re-run minimal-input
   meta on the refit
4. Commit calibration_ladder.md
```

## Research-loop

**Trigger** (mandatory): 3 consecutive nulls, OR 5 saturation events
at the same LB, OR 2 days without LB lift.

```
1. Web search (Opus): top public notebooks for the comp slug, top
   discussion threads in last 30 days
2. Read 2 prior-comp writeups in same domain (same metric, similar
   class imbalance, similar data type)
3. Persona rotation (Opus): invoke ML Researcher, return 5 untried
   mechanisms with citations
4. Rank by predicted EV × cost-to-test → top 3 to experiment queue,
   emit audit/YYYY-MM-DD-research.md with citations
```

The agent will want to skip this. The human PI should enforce it.

## Weekly-loop

**Trigger**: every 7 days, plus the start of the final 3-day window.

```
1. Re-read CLAUDE.md ⚠️ rules in full
2. Audit ceiling thesis: if any session claimed "structural ceiling",
   trigger Research-loop now
3. Persona rotation: rotate at least one on a stuck problem
4. Update REPORT.md with the week's results
5. Submission-budget audit: did we use 10/day? if not, why?
6. Commit a 5-line weekly summary to audit/
```

## Loop interaction

Day-loop wraps Experiment-loop. Calibration-loop is triggered by
submission count or leakage drift. Research-loop is triggered by
plateau detection (mandatory). Weekly-loop runs across day boundaries.

The agent's default failure mode is to stay in Experiment-loop,
ignoring Research-loop triggers. This is what the PI watches for.
