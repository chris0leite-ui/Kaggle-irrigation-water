# Next steps

Ranked plan with expected OOF deltas, calibrated against the current
baseline (LGBM + tuned log-bias, 5-fold CV OOF = **0.97097**) and the
LB reference points (tied pack 0.98114, leader 0.98219). Ten days to
deadline, 10 LB submissions/day, 0 spent. Updated 2026-04-20 after
the FE null result (step 3 below is ruled out; step 4 is now the
next open bet).

## 1 · Burn one submission to calibrate CV ↔ LB (today)

Submit `submissions/baseline_lgbm_tuned.csv` as-is. Costs 1/100 of the
remaining budget; the only experiment that answers *"is the 0.98114
pack running argmax or already tuned?"*.

- If LB ≈ 0.98 → the pack is already tuned; our gap is real and has to
  come from features / diversity / the original dataset.
- If LB ≈ 0.971 → our CV is calibrated, and bias tuning is the trick
  separating us from the pack (implies the pack is running argmax; we
  may already be closer than we think).

Until this is done, every downstream decision is a guess about what
the pack is doing.

## 2 · Original Irrigation Prediction dataset ablation

`data/archive.zip` is already in the repo. Controlled ablation: fit
the same LGBM pipeline on (a) synthetic only vs (b) synthetic +
original concatenated with a sample-weight knob. Score both on the
synthetic-validation folds so the metric is apples-to-apples.

- Expected delta: **−0.005 to +0.005** (sign genuinely unknown — DGP
  may diverge).
- High info-per-hour: resolves an open question either way.

## 3 · ~~Plug domain features into LGBM~~ — ruled out 2026-04-20

Ran `scripts/benchmark_fe.py` with 8 engineered cols (`ET0_proxy`,
`Kc_stage`, `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`,
`Eff_Rainfall_active`, `Crop_x_Stage`, `Season_x_Region`). Tuned OOF
**0.97045** vs baseline 0.97097 (Δ = **−0.00052**, within the 0.00088
fold std). LGBM at 127 leaves was evidently not leaf-limited on this
dataset, so prebuilt interactions added no new splits. Parked — see
REPORT.md §5. Move to step 4.

## 4 · Seed-bag LGBM (3–5 seeds) — now the top open bet

Average OOF probs + test probs across seeds, retune bias once on the
averaged OOF.

- Fold-level std on bal_acc is ~0.00088 (measured on FE run);
  √5 reduction ≈ **+0.0005–0.001** expected. Smaller than originally
  estimated but still the cheapest remaining win.
- Cheap — 3–5× runtime of `scripts/benchmark.py`, no new code beyond
  a seed loop wrapper.

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
