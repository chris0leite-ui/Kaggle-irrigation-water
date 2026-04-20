# Work Report

A structured readout of what was done, what was observed, and what was
concluded. Lives alongside `CLAUDE.md` (the running log),
`DOMAIN.md` (the agronomy primer) and `LEARNINGS.md` (the portable
patterns).

## 1. Problem summary

- **Task**: 3-class classification (`Low` / `Medium` / `High`) of
  `Irrigation_Need` on 19 tabular features (11 numeric, 8 categorical).
- **Metric**: balanced accuracy (macro-recall).
- **Data**: 630k train rows, 270k test rows, no missingness, no
  train/test categorical-vocabulary drift.
- **Priors**: Low 58.7% / Medium 37.9% / High 3.3% — badly imbalanced;
  under balanced accuracy, `High`-class recall drives the score.

## 2. Data observations

- **Missingness**: none in train or test.
- **Drift**: test shares train's categorical vocabulary exactly. A
  formal numeric-drift (KS) pass is still TODO.
- **Top numeric signals (F-stat on a stratified 50% EDA subsample)**:
  `Soil_Moisture` (41k) ≫ `Wind_Speed_kmh` (11k) ≈ `Temperature_C`
  (11k) > `Rainfall_mm` (3.5k). All other numerics are < 100.
- **Top categorical signals (chi²)**: `Crop_Growth_Stage` (97k) ≫
  `Mulching_Used` (28k) ≫ everything else (< 1k).
- So ~6 features carry the bulk of the signal; the rest look like
  either noise or near-uniform structure.
- EDA uses a stratified 50% subsample (seed=42); other 50% held out.
- EDA report: `plots/eda/report.html` (self-contained, base64 images).

## 3. Models tried

All 5-fold stratified CV, seed 42. OOF balanced accuracy.

| Tier                     | Model / rule                             | OOF bal_acc |      Δ vs prev |
|---                       |---                                       |         ---:|           ---: |
| Floor                    | Majority-class (all Low)                 |     0.33333 |             –  |
| Floor                    | Stratified random                        |     0.33384 |      +0.00001  |
| Heuristic                | H1 — Soil_Moisture alone                 |     0.62911 |      +0.29527  |
| Heuristic                | H2 — raw water balance, equal z-weights  |     0.60606 |      −0.02305  |
| Heuristic                | H3 — H2 + Kc + mulch + soil capacity     |     0.63041 |      +0.00130  |
| Linear (MNLogit)         | F1 — minimal balance, 4 feats, argmax    |     0.52337 |             –  |
| Linear (MNLogit)         | F1 — tuned log-bias                      |     0.64721 |      +0.12384  |
| Linear (MNLogit)         | F2 — balance + Kc + deficit + mgmt, 19f  |     0.66429 |             –  |
| Linear (MNLogit)         | F2 — tuned log-bias                      |     0.78074 |      +0.11645  |
| Linear (MNLogit)         | F3 — full structural, 48 feats           |     0.61680 |             –  |
| Linear (MNLogit)         | F3 — tuned log-bias                      |     0.73294 |      +0.11614  |
| Tree (LGBM)              | argmax                                   |     0.96135 |             –  |
| Tree (LGBM)              | prior-reweight argmax                    |     0.97065 |      +0.00930  |
| **Tree (LGBM)**          | **tuned log-bias**                       | **0.97097** |      +0.00032  |
| Tree (XGBoost)           | argmax (per-fold ~0.961–0.964)           |      ~0.962 |             –  |
| Tree (LGBM+FE)           | argmax (8 engineered cols)               |     0.96133 |      −0.00002  |
| Tree (LGBM+FE)           | tuned log-bias (8 engineered cols)       |     0.97045 |      −0.00052  |
| Blend                    | LGBM + MNLogit Fk, sweep w ∈ [0, 0.5]    |     0.97097 |   +0.00000 (null) |
| LB reference             | LB tied pack (~100 teams)                |     0.98114 |             –  |
| LB reference             | LB leader (Chris Deotte)                 |     0.98219 |             –  |

Key read-outs

- **Soil_Moisture is the single dominant feature.** H1 (just signed
  Soil_Moisture + 2 thresholds) already covers 2/3 of the distance from
  random (0.333) to competitive (0.971). H2 is worse than H1 because
  equal-weight z-scoring dilutes a dominant signal with noisier axes.
- **Nonlinear interactions are where the real lift lives.** H3 vs
  LGBM = +0.34 of bal_acc, on the same underlying physics features.
  That gap isn't "the equation is wrong"; it's "additive combinations
  of the equation's terms miss the Medium↔High decision surface".
- **Bias tuning is model-agnostic and big when the base model is
  uncalibrated.** +0.12 on each MNLogit formula, +0.010 on LGBM. The
  leaderboard pack almost certainly already applies some form of it;
  our LGBM-tuned 0.971 is not a surprise to them.
- **LGBM × MNLogit blending is a null.** Δ = 0 at every mixing weight.
  MNLogit at 0.78 tuned is simply too far below LGBM's 0.971 to
  contribute orthogonal signal.
- **Hand-engineered domain features add nothing to LGBM.** Injecting
  `ET0_proxy`, `Kc_stage`, `ETc_proxy`, `Soil_deficit`, `Is_Rainfed`,
  `Eff_Rainfall_active`, `Crop_x_Stage`, `Season_x_Region` (+8 cols,
  27 total) moved tuned OOF from 0.97097 → 0.97045 (Δ = −0.00052,
  smaller than the 0.00088 fold std). Trees already discover these
  interactions; prebuilt features were the hypothesised fallback for
  near-leaf-limit splits, and that hypothesis doesn't hold at the
  current leaf count. Bias solution was essentially unchanged
  (Low +0.23, Medium +0.57, High +3.40).
- **Confusion-matrix mass lives at Medium↔High.** LGBM tuned still
  flips ~4k Medium→High and ~875 High→Medium on OOF; the heuristic
  makes that error 50× more often. This is where any further gain must
  come from.

## 4. Strategy and next steps

Rough rule of thumb for the remaining 10 days: we need +0.010 bal_acc
to reach the tied pack, and +0.011 to reach rank 1. Our baseline
already includes the "threshold trick", so the remaining lift has to
come from feature engineering, model diversity, or external data.

Ranked by expected ROI / effort (post-FE-null update 2026-04-20):

1. **Seed-bag LGBM (now top of list).** 3–5 seeds, average OOF + test
   probs, retune bias. Fold-level std on bal_acc is ~0.00088
   (measured on FE run; earlier estimate of 0.002 was conservative),
   so the expected SE reduction is roughly √5 ≈ 2.2×. Estimate:
   +0.0005–0.001.
2. ~~**Feature engineering on LGBM.**~~ **Ruled out (2026-04-20).**
   Adding `ET0_proxy`, `Kc_stage`, `ETc_proxy`, `Soil_deficit`,
   `Is_Rainfed`, `Eff_Rainfall_active`, `Crop_x_Stage`,
   `Season_x_Region` moved tuned OOF from 0.97097 → 0.97045
   (Δ = −0.00052, within 1σ fold noise). Trees already discover these
   interactions. See §5.
3. **LGBM + XGBoost blend.** XGBoost OOF is already saved from the
   multi-model benchmark. Geometric mean with re-tuned bias. Estimate:
   +0.001–0.002 (model diversity, not raw strength).
4. **Original Irrigation Prediction dataset.** Explicitly allowed and
   not yet used. Steps: download, schema-align, add as either
   (a) concat with train, or (b) external train and synthetic as
   validation. Sign unknown — DGP may diverge. Estimate: −0.005 to
   +0.005.
5. **LGBM hyperparameter refresh.** Current config (lr=0.05, 127
   leaves, ~280 trees) was a gut estimate, not a search. One round of
   Optuna on (num_leaves, min_data_in_leaf, feature_fraction,
   bagging_fraction, lr, reg_alpha, reg_lambda). Estimate: +0.001.
6. **Ordinal-aware loss or threshold metric.** Errors cluster between
   adjacent classes; an ordinal objective may reduce Medium↔High
   confusion even though the metric itself is order-agnostic.
   Estimate: +0.001 or nothing.
7. **DGP archaeology (parked).** Reverse-engineer the synthetic
   generator. High effort, unclear payoff — revisit only if stuck
   above 0.9815 after the cheaper bets land.

Minimum-viable first submission: the current
`submissions/baseline_lgbm_tuned.csv` (LGBM + tuned log-bias, OOF
0.97097). Sending it is cheap information — it tells us the LB gap
more accurately than our guess. Decision pending: burn one of the 10
daily submissions to calibrate, or wait until features/blend are
ready.

## 5. Rejected ideas

- **Equal-weight z-score fusion of water-balance axes (H2).** Worse
  than using Soil_Moisture alone (H1) because it dilutes the dominant
  signal. If we come back to hand-weighted scores, weights must be
  proportional to per-axis informativeness (F-stat or similar), not
  uniform.
- **LGBM + MNLogit blending.** Zero contribution from the linear
  model at any mixing weight. Parked as a stacking option only (use
  MNLogit OOF probs as additional features inside LGBM) — and even
  that is low expected value given how much weaker the linear model
  is.
- **CatBoost standalone.** Fold 1 argmax 0.96000 ≈ LGBM/XGB — no model-
  level edge, 23 min/fold training cost. Killed after fold 1. Could
  revisit only for a full 4+ model blend late in the competition if
  compute budget allows.
- **Hand-engineered water-balance features inside LGBM.** Eight cols
  from MNLogit-F2 / heuristic-H3 (ET0_proxy, Kc_stage, ETc_proxy,
  Soil_deficit, Is_Rainfed, Eff_Rainfall_active, Crop_x_Stage,
  Season_x_Region) ran in `scripts/benchmark_fe.py` — tuned OOF
  0.97045 vs baseline 0.97097 (Δ = −0.00052, within 1σ fold noise of
  0.00088). LGBM at 127 leaves / 200 min_data_in_leaf was clearly not
  leaf-limited on this dataset, so prebuilt interactions add no new
  splits. Artefacts: `scripts/artifacts/oof_lgbm_fe.npy`,
  `scripts/artifacts/test_lgbm_fe.npy`,
  `submissions/submission_lgbm_fe_tuned.csv`. Could revisit only if
  we ever retrain with a much smaller leaf budget or a tiny subset.

## 6. Open questions

- Public LB calibration: is 0.98114 already tuned-log-bias, or raw
  argmax? Sending one submission would resolve this.
- Does the original Irrigation Prediction dataset improve CV?
- How much of the 0.00105 gap top↔pack is systematic vs noise? Our
  fold std is ~0.002 — the gap is within one seed's worth of variance.
- Is there an ordinal structure lurking in the synthetic DGP that an
  ordinal loss would exploit?
