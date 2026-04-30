# Kickoff — Day 1

Run this on the very first session of a new Kaggle comp.

## Step-by-step

### 1. Repo bootstrap

```bash
# clone the template (or copy from a previous comp, prune comp-specific scripts)
mkdir <comp-slug> && cd <comp-slug>
# copy the lift-list files (see Kaggle-irrigation-water/writeup/framework/13-repo-template.md)
# bootstrap.sh, scripts/lb_status.py, scripts/common.py, scripts/meta_common.py,
# tests/test_oof_invariants.py, .gitignore (artifact policy)
chmod +x bootstrap.sh
./bootstrap.sh   # installs deps + downloads competition data
```

### 2. Fill `comp-context.md`

Open the comp page on Kaggle. Fill out (and don't re-ask later):

```yaml
slug: <comp-slug>
url: https://www.kaggle.com/competitions/<comp-slug>
task: <binary | multiclass | regression | ...>
metric: <e.g., balanced accuracy, RMSE, mAP>
public_split_pct: <e.g., 20 | 30>
lb_stability: <stable | per-row-seeded | probe-once>
train_rows: <N>
test_rows: <N>
feature_count: { numeric: N, categorical: N }
class_priors: <if classification>
deadline: <YYYY-MM-DD>
team_size_limit: <typically 3 or 5>
submission_budget: 10/day
final_submissions: 2
data_license: <CC BY 4.0 | other>
external_data_allowed: <yes | no | conditional>
lb_best_at_kickoff: <score>
pack_score_at_rank_100: <score>
```

After Day 1, **never re-ask these facts**. They're settled.

### 3. EDA on a 50% stratified subsample

- Class priors, missingness, train/test categorical drift.
- Top-N feature signals (F-stat / chi²).
- Numeric distribution comparison train vs test.
- Save report to `plots/eda/report.html` (self-contained).

Don't read full data into the agent context. Summarize EDA into a
~30-line `eda-summary.md`.

### 4. `brief.md` — verbatim host material

Copy the comp description, evaluation, data description, and rules
verbatim from the comp page. Cap at 150 lines (host material is
usually short).

### 5. Baseline LGBM

```python
# scripts/baseline_lgbm.py
# 5-fold StratifiedKFold, seed=42
# tuned via prior-reweight + log-bias for the metric
```

Run, emit `oof_baseline_lgbm.npy` + `test_baseline_lgbm.npy`.
Build `submission_baseline_lgbm.csv`.

### 6. First submission

**Ask PI**:
> "Day-1 baseline: LGBM tuned, OOF <score>. Recommend submitting to
> calibrate OOF→LB gap. One slot of today's 10. Submit?"

Single-shot the submit on PI confirmation. Record the LB score.

### 7. Compute the OOF→LB calibration

```
gap = LB - OOF
fold_std = std(per-fold OOF scores)
```

If `|gap|` < `2 × fold_std`, OOF is calibrated and future deltas
can be trusted from OOF within ~1bp.

If `|gap|` > `2 × fold_std`, investigate before any further LB
probes:

- Public-LB might be a heavily skewed slice.
- Pipeline might leak across folds.
- Class imbalance might hit folds differently than test.

### 8. End of Day 1 — audit entry

Write `audit/<YYYY-MM-DD>-day-1-kickoff.md` with:

- `comp-context.md` summary (key facts, 5 lines).
- EDA highlights (5 lines).
- Baseline OOF and LB.
- Calibration verdict (calibrated / drift).
- 3 hypotheses queued for Day 2.

## Day-2 setup tail

Before closing Day 1, queue up Day 2's first 3 experiments:

1. **Domain hypothesis seeder** — read whatever exists about the
   problem domain. Capture in `DOMAIN.md`. Use as hypothesis seeder
   only; don't deeply engineer features yet.
2. **Heuristic baselines** — H1 (single-feature threshold), H2
   (hand-coded rule from domain reading).
3. **DGP archaeology** if the comp is synthetic — brute-force
   candidate rules on 5-6 candidate features.

## Anti-patterns on Day 1

- Skipping `comp-context.md` because "we'll figure it out later".
  You'll re-ask the same facts mid-comp. Don't.
- Loading CLAUDE.md before it has any content.
- Building a stack on Day 1 with no baseline calibration.
- Submitting a "tweaked" LGBM with hand-picked hyperparameters
  before submitting the plain baseline. The plain baseline is
  what calibrates OOF→LB.
- Doing FE before the DGP is understood. Physics-faithful FE on
  irrigation-water added Δ = −0.00052; we deleted the FE menu
  later. Wait until you know the DGP shape.
