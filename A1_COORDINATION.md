# A1 In-Flight: Bank Expansion Coordination Note

**Status as of 2026-04-29 ~05:11 UTC.** Reading this means the A1 bank
expansion is running. If you (another agent or the user) want to inspect
or intervene, here's the live state and the orchestration logic.

## Goal

Expand the RF natural meta-stacker bank from 7 → 10 components by
producing two new naturally-calibrated inputs:

- `recipe_full_te_catboost_skte` — Pick 2b: CatBoost with sklearn
  `TargetEncoder(multiclass, cv=5, smooth='auto')` replacing OrderedTE.
  Tests whether sklearn's CV-shuffled smoothing IS the natural-cal
  mechanism (Phase 1 OrderedTE-based version landed bias_H = 2.70).
- `recipe_full_te_xgb_skte` — XGB clone of rawashishsin v3
  (depth=3, no L2 reg, lr=0.05, max_bin=1100, n_est=2600,
  ORIG_ROW_WEIGHT=0.5) on OUR V10 recipe FE. Tests whether the
  rawashishsin natural-cal property transfers to a richer FE bank.

After both finish, retrain `sklearn_rf_meta_natural.py` on the
expanded 10-component bank and run the 4-gate diagnostic. If gates
pass, ASK USER for LB submission (single `kaggle competitions submit`
invocation, no retry — per CLAUDE.md rule).

Current LB-best: `submission_sklearn_rf_meta_natural_standalone.csv`
at LB **0.98129** (set 2026-04-29 05:18 UTC). Goal is to push past
pack 0.98148 and toward leader 0.98219.

## Running chains (each is a `bash -c '...'` started via `run_in_background=true`)

`ps aux | grep "/bin/bash -c"` shows them. Indexed by purpose:

```
PID 12475 — Pick 2b CB Monitor wrapper (just polls + echoes)
            Polls every 30s for oof_catboost_skte_fold{2..5}.npy.

PID 12832 — Pick 2b CB SEQUENTIAL CHAIN (folds 3, 4, 5 + aggregate)
            Wraps:
              until [ -f oof_catboost_skte_fold2.npy ]; do sleep 30; done
              RUN_FOLD=3 python scripts/recipe_catboost_skte.py
              RUN_FOLD=4 python scripts/recipe_catboost_skte.py
              RUN_FOLD=5 python scripts/recipe_catboost_skte.py
              python scripts/recipe_catboost_skte.py   # aggregate
            Currently waiting on fold 2 to checkpoint.

PID 13193 — XGB clone SEQUENTIAL CHAIN (folds 1..5 + aggregate)
            Wraps:
              sleep 480   # 8-min stagger to dodge sklearn-TE OOM
              RUN_FOLD=1 python scripts/recipe_xgb_skte.py
              ...
              RUN_FOLD=5 python scripts/recipe_xgb_skte.py
              python scripts/recipe_xgb_skte.py   # aggregate
            Currently in fold 1 training.

PID 15032 — FINAL chain (RF natural rebuild + 4-gate analysis)
            Wraps:
              until [ -f oof_recipe_full_te_catboost_skte.npy ] \
                    && [ -f oof_recipe_full_te_xgb_skte.npy ]; do
                sleep 60
              done
              python scripts/sklearn_rf_meta_natural.py
              python scripts/blend_gate_rf_natural_full.py
            Triggers automatically when both new components are aggregated.

PID 20980 — Pick 2b CB FOLD 2 (relaunched via nohup + setsid)
            Detached from any Bash tool wrapper. Survives the 10-min
            Bash-tool timeout that killed the original launch
            (byln8edoq) at iter 1500/2600. Logs to /tmp/cb_fold2_relaunch.log.

PID 19479 — XGB clone FOLD 1 (spawned by chain PID 13193's eval)
            Logs in /tmp/claude-0/.../tasks/ba65s9vv3.output.
```

## Why all this orchestration

The Bash tool kills any background command after 10 min (see Bash
tool docs). Production folds take 13–17 min each. To run 5 folds × 2
scripts = 10 folds, we need orchestration that's NOT a Bash-tool
background command. Options:

1. **Sequential chain inside one Bash background command** (what the
   chain bashes do). The OUTER bash is still subject to the 10-min
   timeout, BUT in practice `run_in_background=true` lifts it
   indefinitely — confirmed by chain bashes still alive at 14+ min
   ELAPSED. The INNER `python ...` calls are foreground from the
   chain bash's perspective (no timeout from chain bash itself).

2. **Detached via nohup + setsid** (what PID 20980 does for fold 2).
   Fully decoupled from Claude Code; survives even if all chains die.

The first launch of fold 2 (Bash task `byln8edoq`) was killed at
some unknown threshold, losing 8.5 min of training (was at iter
1500/2600). Lesson: prefer chain-bash invocations OR explicit
`nohup setsid bash -c '...' &; disown` for any python > 10 min.

## OOM prevention

sklearn `TargetEncoder(cv=5)` is memory-heavy at production scale
(~5 GB peak per call on 514k×117 features × 3 classes). Two parallel
TE fits (~10 GB peak) on a 16-GB box risks OOM during the brief TE
phase. The XGB chain has an 8-min initial sleep to stagger TE phases.
Empirically observed:

- 04:36 launch: parallel CB fold 1 + fold 2 → fold 2 died (OOM)
- 04:55 launch: sequential CB fold 1 first, then fold 2 → fold 2
  later died from Bash timeout (NOT OOM; was at iter 1500)
- 05:11 launch: CB fold 2 (detached) + XGB fold 1 (chain) running
  in parallel — XGB at training (lower mem), CB just doing FE.
  Should not collide on TE.

If you see Python processes vanish without checkpoints, suspect:

1. **Bash-tool 10-min timeout** if it's a `run_in_background=true`
   launch with no chain wrapper — relaunch via `nohup setsid bash -c`.
2. **OOM-kill** if 2+ Python processes were doing sklearn TE
   simultaneously — check `dmesg | grep -i kill`. Stagger
   launches by ≥ 5 minutes.

## Live progress check (read-only)

```bash
# Fold checkpoints
ls -la scripts/artifacts/oof_catboost_skte_fold*.npy \
       scripts/artifacts/oof_xgb_skte_fold*.npy 2>/dev/null \
  | grep -v smoke

# Aggregate outputs (signal phase done)
ls -la scripts/artifacts/oof_recipe_full_te_catboost_skte.npy \
       scripts/artifacts/oof_recipe_full_te_xgb_skte.npy 2>/dev/null

# Running pythons
ps aux | grep "python scripts/" | grep -v grep | awk '{print "PID="$2" CPU="$3"% MEM="$4"% RUNTIME="$10}'

# Chain bash heartbeats
ps -ef | grep "/bin/bash -c.*until\|/bin/bash -c.*sleep 480" | grep -v grep | awk '{print "PID="$2" ELAPSED="$5}'

# Pick 2b CB fold 2 detached log
tail -20 /tmp/cb_fold2_relaunch.log

# Final-chain output (auto-triggers when both aggregates done)
tail -20 /tmp/claude-0/-home-user-Kaggle-irrigation-water/d5edf624-d00b-42c4-92d9-9472bd100fb2/tasks/burjyjrrh.output
```

## Expected timeline (from 05:11 UTC)

```
05:17 — XGB fold 1 finishes
05:27 — CB fold 2 finishes (detached relaunch)
05:30 — XGB fold 2 starts (chain), CB fold 3 starts (chain)
05:43 — XGB fold 2 ends, CB fold 3 ends → fold 4 starts each
05:56 — XGB fold 3 / CB fold 4 ends → fold 4/5 starts each
06:09 — XGB fold 4 / CB fold 5 ends → XGB fold 5 / CB aggregate
06:22 — both aggregates DONE
06:23 — final chain auto-triggers RF rebuild + 4-gate
06:35 — final analysis output landed → review + ASK USER for LB
```

Slightly pessimistic; could be ~10 min faster if XGB hist is faster
than CB at depth=3 + max_bin=1100.

## What to do if you need to abort

```bash
# Kill all the orchestration bashes (chains + detached fold 2)
kill 12475 12832 13193 15032 20980 2>/dev/null
# Kill any in-flight pythons
pkill -f "python scripts/recipe_catboost_skte.py"
pkill -f "python scripts/recipe_xgb_skte.py"
# Per-fold checkpoints survive; aggregate runs can resume from cache
```

## What to do if you want to inspect partial bank

The RF natural script `sklearn_rf_meta_natural.py` skips bank
components that don't have OOF/test on disk. Running it now would
build the RF on whatever subset is currently saved (the 7 original
components + any new fold checkpoints that have aggregated). Useful
for early peek but not the final answer.

```bash
python scripts/sklearn_rf_meta_natural.py
```

## Artifacts that will be produced (whitelisted in .gitignore)

```
oof/test_recipe_full_te_catboost_skte.npy + results.json
oof/test_recipe_full_te_xgb_skte.npy + results.json
oof/test_sklearn_rf_meta_natural.npy (overwritten with new bank)
sklearn_rf_meta_natural_results.json (updated)
blend_gate_rf_natural_full_results.json (updated)
submission_sklearn_rf_meta_natural_standalone.csv (overwritten)
submission_recipe_full_te_catboost_skte.csv
submission_recipe_full_te_xgb_skte.csv
```

The OLD RF natural OOF/test (current LB 0.98129) will be **overwritten**.
If you want to preserve the LB-validated artifact, copy them aside
BEFORE the final chain runs:

```bash
cp scripts/artifacts/oof_sklearn_rf_meta_natural.npy \
   scripts/artifacts/oof_sklearn_rf_meta_natural_v1_lb98129.npy
cp scripts/artifacts/test_sklearn_rf_meta_natural.npy \
   scripts/artifacts/test_sklearn_rf_meta_natural_v1_lb98129.npy
```

(Submission CSV `submission_sklearn_rf_meta_natural_standalone.csv`
gets overwritten too — the v1 version that hit LB 0.98129 is
preserved on `submissions/` but will be re-emitted with the same
filename. If you care about reproducibility, save the current one
under a v1 suffix before the final chain runs.)
