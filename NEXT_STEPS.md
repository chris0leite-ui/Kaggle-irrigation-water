# Next steps

Ranked plan with expected OOF deltas, calibrated against the current
baseline (LGBM + tuned log-bias, 5-fold CV OOF = **0.97097**) and the
LB reference points (tied pack 0.98114, leader 0.98219). Ten days to
deadline, 10 LB submissions/day, 0 spent. Updated after the
heuristic / MNLogit / blend sweep on 2026-04-20.

## 1 · Burn one submission to calibrate CV ↔ LB ✓ done 2026-04-20

Submitted `submissions/submission_baseline_lgbm_tuned.csv` (LGBM +
tuned log-bias). **LB public = 0.96972** at rank 726 / 2357 (top 31 %).

- OOF 0.97097 vs LB 0.96972 → −0.00125, inside one fold-std (~0.002).
  CV is well-calibrated — experiment deltas from 5-fold OOF can be
  trusted going forward.
- The pack is *not* running raw argmax (that would have landed them
  near our 0.96 tier). Their 0.98114 comes from structural advantages:
  feature engineering, original dataset, seed bagging, better HPs, or
  all of the above.
- LB budget: 1 / 10 spent; 9 remaining today.

## 2 · Original Irrigation Prediction dataset ablation

`data/archive.zip` is already in the repo. Controlled ablation: fit
the same LGBM pipeline on (a) synthetic only vs (b) synthetic +
original concatenated with a sample-weight knob. Score both on the
synthetic-validation folds so the metric is apples-to-apples.

- Expected delta: **−0.005 to +0.005** (sign genuinely unknown — DGP
  may diverge).
- High info-per-hour: resolves an open question either way.

## 3 · Plug domain features into LGBM

Lift the engineered columns already validated by MNLogit-F2 and
heuristic-H3 straight into the LGBM feature matrix:

- `ET0_proxy = Temperature_C · (1 − Humidity/100) · Wind_Speed_kmh`
- `Kc_stage` (FAO-56 lookup by `Crop_Growth_Stage`)
- `ETc_proxy = ET0_proxy · Kc_stage · (1 − 0.30·is_mulched)`
- `Soil_deficit = max(0, capacity[Soil_Type] − Soil_Moisture)`
- `Crop_Type × Crop_Growth_Stage` (full Kc surface)
- `Season × Region` (climatic regime)
- `Is_Rainfed = (Irrigation_Type == "Rainfed")`

Retune log-bias after.

- Expected delta: **+0.001 to +0.003**. Trees often discover these
  interactions on their own but pre-built features help when splits
  are leaf-limited (our config hits 127 leaves).

## 4 · Seed-bag LGBM (3–5 seeds)

Average OOF probs + test probs across seeds, retune bias once on the
averaged OOF.

- Fold-level std on bal_acc is ~0.002; √5 reduction ≈ **+0.001** expected.
- Cheap — 3–5× runtime of `scripts/benchmark.py`, no new code.

## 5 · LGBM + XGBoost blend

XGB OOF already saved to `scripts/artifacts/oof_xgb_baseline.npy`.
Geometric mean, sweep mixing weight w, retune bias — same harness as
`scripts/blend_lgbm_mnlogit.py`.

- Expected delta: **+0.001 to +0.002** from model diversity.
- Unlike MNLogit (blend was a null), XGB is strong enough (argmax
  ~0.962) to plausibly contribute orthogonal splits.

## 6 · HP refresh + ordinal loss (only if still stuck)

- Quick Optuna on `(num_leaves, min_data_in_leaf, lr,
  feature_fraction, bagging_fraction, reg_alpha, reg_lambda)` — 50
  trials, one seed.
- LGBM with `multiclassova` + custom sample weights that upweight
  Medium↔High adjacency (ordinal-ish).
- Expected delta: **+0.001 each**, or nothing.

## 7 · Parked — DGP archaeology

Reverse-engineer the synthetic generator. Only if 1–6 leave us below
0.98 and there's still time.

## Suggested immediate action

Submit the tuned baseline **and** start step 2 in parallel. The
submission result is back within minutes; the ablation within an
hour. By end of day we'd know the LB gap and whether the external
dataset is even the right lever.
