# CLAUDE.md — {{COMP_SLUG}}

Running log + ⚠️ rules. Cap ≤50k tokens; archive when bloated.

## ⚠️ Top-level rules (inherited from kaggle-comp framework)

These eleven invariants are LOAD-BEARING. Do not skip.

1. **Ask-first / no-loop on submissions.** Every `kaggle competitions
   submit` is single-shot, explicitly approved. Never wrap in retry
   / `until` / `while` / `for`. Monitors that POLL are fine; monitors
   that WRITE are forbidden.
2. **Smoke + 1-fold time-probe + 1h GPU cap.** Smoke at 1 fold /
   50k rows. Then 1-fold full-data probe. If 5-fold projection ≥1h,
   shrink. If a kernel is in preprocessing at t+30min with no fold
   output, kill it.
3. **4-gate leakage filter pre-LB-probe.** G1 standalone OOF clears
   anchor; G2 blend lift; G3 net rare-class-flip ratio ≥0.5; G4
   direction asymmetry. Plus minimal-input-meta sanity check.
4. **NEVER-GIVE-UP / saturation-is-bounded / never-lock-and-stop.**
   Saturation evidence proves we tested known levers, not that no
   lever exists. After every null, brainstorm 3 untried mechanisms.
   Locking is for the final 3-day window only.
5. **Keep CLAUDE.md fresh / archive-on-bloat.** Cap ≤50k tokens.
   Subagents load slices, not full files.
6. **Heuristics before heavy compute.** Closed-form rule / threshold
   / hand-coded baseline before Optuna / GPU / 5-fold-bagging.
7. **Research before saturation.** At every plateau (3 nulls / 5
   sat at same LB / 2 days no lift): web search top notebooks,
   read 2 prior-comp writeups, list 5 untried mechanisms with
   citations BEFORE declaring ceiling.
8. **Settled-once facts** live in `comp-context.md`. Never re-ask.
9. **File-size cap ≤150 lines** for any committed doc.
10. **Pull-style updates.** No proactive minute-level chatter; on
    PI pull, 1-2 sentences with the latest fact.
11. **Model routing.** Haiku for routine read-only checks; Sonnet
    default; Opus for hard reasoning. Use the daily 10/day budget.

## ⚠️ Defaults baked in from prior-comp postmortem

These were the R1/R2/R7 changes from irrigation-water postmortem-07.
They override the kickoff-time defaults.

- **R1 — Two-anchor OOF.** Every gated candidate must pass under
  TWO CV schemes: (a) standard 5-fold StratifiedKFold seed=42, AND
  (b) GroupKFold on a row-id hash (or repeated stratified with a
  different seed). Mechanisms that overfit one fold geometry will
  diverge.
- **R2 — Final selection along the public-LB axis.** PRIMARY = best
  public LB. HEDGE = best OOF that *regressed ≤30bp on public*. NOT
  another orthogonal-mechanism hedge. (Last comp: 5 of our subs
  beat PRIMARY on private; all had been rejected for public regress.)
- **R7 — Override-mechanism rules.** Override candidates with flip
  count <200 cannot be PRIMARY (only HEDGE). Override candidates
  with >200 flips need explicit PI sign-off. Override flips on a
  small public split overfit deterministically.
- **R5 — Final OOF-best regression probe.** In the final 3-day
  window, 1 mandatory probe of the OOF-best candidate that was
  rejected for public regression.
- **R8 — End-of-comp**: log final percentile to
  `~/.claude/skills/kaggle-comp/improvements.md`.

## Current state (Bookkeeper updates daily)

```yaml
day: 1
lb_best_today: {{LB_BEST}}        # leader's score on public LB today
our_lb_best: null                 # update after first submit
submissions_used_today: 0
submissions_used_total: 0
saturation_count: 0
mechanism_families_explored: []
plateau_days: 0
```

## Calibration ladder

Updated by the Calibration-loop. Format: mechanism / OOF / LB / gap.

| Mechanism | OOF | LB | Gap | Notes |
|---|---:|---:|---:|---|
| (baseline_lgbm, after Day-1 first submit) | | | | |

## Hypothesis board

```
- (queued for next session)
- (queued for next session)
- (queued for next session)
```

## Friction log pointer

Friction one-liners go in `audit/friction.md`, NOT here. Distill
weekly per `~/.claude/skills/kaggle-comp/self-improvement.md`.

## Pointers

- `comp-context.md` — settled-once facts.
- `brief.md` — verbatim host material.
- `LEARNINGS.md` — portable patterns from this comp.
- `REPORT.md` — structured work report.
- `audit/` — timestamped per-experiment results.
- `audit/friction.md` — friction one-liners (rotated weekly).
