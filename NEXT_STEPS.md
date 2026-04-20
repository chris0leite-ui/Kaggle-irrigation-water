# Next steps

Ranked plan with expected OOF deltas. Current best:
**LGBM+EXT tuned log-bias, 5-fold CV OOF = 0.97124** (synthetic + 10k
original concat). LB reference points: tied pack 0.98114, leader
0.98219. Ten days to deadline, 10 LB submissions/day, 1 spent (LB
0.96972). Updated 2026-04-20 after step 2 concat (+0.00027) and step
3 FE (null, ruled out).

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

## 2 · Original Irrigation Prediction dataset ablation ✓ done 2026-04-20

Ran `scripts/benchmark_external.py` — concat synthetic + 10k original
rows into each training fold, validate on synthetic-only folds.
**Tuned OOF 0.97124** vs 0.97097 baseline → **Δ = +0.00027**, smaller
than the 0.00068 fold std. Tiny free positive, not a silver bullet.

Also ran `scripts/transfer_check.py` (train on 8k original, eval on
630k synthetic): bal_acc **0.96278** — only 0.00819 below the
5-fold synthetic baseline despite 63× less training data. DGPs
overlap almost completely; the small concat delta reflects that 10k
is just 1.6 % of the training pool, not that the datasets diverge.

- Kept: new best OOF 0.97124; LGBM+EXT now the base model.
- Ruled out: bigger lift from this lever (sample-weight sweeps would
  marginally boost the original's weight, but the ceiling is bounded
  by the transfer gap — we'd need the ~0.008 gain the transfer proves
  is there in principle, and that requires either 10× more original
  rows or a recipe-change, not a reweighting knob).

## 3 · ~~Plug domain features into LGBM~~ — ruled out 2026-04-20

Ran `scripts/benchmark_fe.py` with 8 engineered cols (`ET0_proxy`,
`Kc_stage`, `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`,
`Eff_Rainfall_active`, `Crop_x_Stage`, `Season_x_Region`). Tuned OOF
**0.97045** vs baseline 0.97097 (Δ = **−0.00052**, within the 0.00088
fold std). LGBM at 127 leaves was evidently not leaf-limited on this
dataset, so prebuilt interactions added no new splits. Parked — see
REPORT.md §5. Move to step 4.

## 4 · Seed-bag LGBM+EXT (3–5 seeds) — top open bet

Average OOF probs + test probs across seeds on the **concat**
(synthetic + 10k original) pipeline, retune bias once on the averaged
OOF.

- Fold-level std on bal_acc is ~0.00068 (measured with EXT);
  √5 reduction ≈ **+0.0005–0.001** over the 0.97124 new base.
- Cheap — 3–5× runtime of `scripts/benchmark_external.py`, no new
  code beyond a seed loop wrapper.

## 5 · LGBM+EXT + XGBoost blend

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
