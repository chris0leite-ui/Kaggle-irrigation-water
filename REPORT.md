# Work Report

A structured readout of what was done, what was observed, and what was
concluded. Lives alongside `CLAUDE.md` (the running log) and
`LEARNINGS.md` (the portable patterns).

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
- **Drift**: test shares train's categorical vocabulary exactly (no new
  levels). Numeric ranges match at a glance; a formal KS pass is still
  TODO.
- **Top numeric signals (F-stat on a stratified 50% EDA subsample)**:
  `Soil_Moisture` (41k) ≫ `Wind_Speed_kmh` (11k) ≈ `Temperature_C`
  (11k) > `Rainfall_mm` (3.5k). All other numerics are < 100.
- **Top categorical signals (chi²)**: `Crop_Growth_Stage` (97k) ≫
  `Mulching_Used` (28k) ≫ everything else (< 1k).
- So ~6 features carry the bulk of the signal; the rest look like
  either noise features or near-uniform structure.
- EDA uses a stratified 50% subsample with `seed=42` so that the
  analysis doesn't lock in decisions from patterns only visible at full
  sample size. The remaining 50% is a clean holdout.
- EDA report: `plots/eda/report.html` (self-contained, images
  base64-embedded).

## 3. Models tried

| Model                          | OOF bal_acc | LB      | Notes |
|---                             |        ---: |    ---: | --- |
| Majority-class (all Low)       |     0.33333 |       – | Metric floor. |
| Stratified random              |     0.33384 |       – | Same floor, confirms metric. |
| LGBM + argmax                  |     0.96135 |       – | 5-fold CV, 250–300 trees, lr 0.05, 127 leaves. |
| LGBM + prior-reweight argmax   |     0.97065 |       – | Divide probs by prior → equal effective prior at decision. |
| **LGBM + tuned log-bias**      | **0.97097** |       – | Coord-ascent on OOF; bias = Low 0.23, Med 0.67, High 3.40. |

- Confusion matrix (tuned log-bias): almost all `Low` rows correctly
  classified; `Medium ↔ High` is the dominant error pattern (~4k
  Medium→High, 874 High→Medium), which is expected given the tiny
  `High` prior.
- Single-seed, single-set-of-params run. No ensembling yet. No feature
  engineering yet. Still ~1 point below the tied pack at 0.98114.

## 4. Final approach

- Not yet selected. Current best candidate is the tuned-log-bias LGBM
  submission at `submissions/baseline_lgbm_tuned.csv`, but it's not
  submitted yet and needs more work before it's worth burning a slot.

## 5. Rejected ideas

- _none yet._

## 6. Open questions

- How much of the 0.98114 ceiling comes from better trees vs. better
  thresholds? The log-bias trick alone gave us +0.010 off argmax;
  public notebooks may already apply it.
- Does bagging across seeds / longer training close the gap to the
  tied pack?
- Would adding the optional real-world Irrigation Prediction dataset
  help, or does it hurt because of DGP mismatch with the synthetic
  train?
- Ordinal-aware loss (`Low < Medium < High`) — worth trying given that
  the errors cluster between adjacent classes, even though the metric
  itself doesn't reward ordering.
