# bagginglr_natural standalone — assumption-challenge probe result (48th saturation)

**Date**: 2026-04-30
**Submission**: `submission_bagginglr_natural_standalone.csv`
**LB public**: **0.98106**

## Hypothesis tested

Is v1's LB 0.98129 driven by the 7-component natural-cal bank, or by
RandomForest specifically as the L2 algorithm? The standalone CSV had
never been built — only the L3 (RF+ET+BagLR) mean shipped from
`scripts/n3_l3_bagging_metas.py`. We probed the assumption with one
slot since improvement was not expected.

## Construction

- Same exact 7-component bank as v1 (LB-validated 0.98129):
  rawashishsin_2600, realmlp, recipe_full_te, recipe_full_te_catboost,
  recipe_full_te_catboost_natural, xgb_corn, xgb_dist_digits.
- L2: `BaggingClassifier(LogisticRegression(C=0.1), n_estimators=100,
  max_features=0.7, max_samples=0.8)` instead of v1's RandomForest.
- Bias-tune on bagginglr OOF (single pass), argmax on test.
- Build script: `scripts/build_bagginglr_standalone.py`.

## Result

```
                  OOF        LB         gap
v1 (RF L2)     0.98063 -> 0.98129   -0.00066
bagginglr      0.98065 -> 0.98106   -0.00041
delta vs v1    +0.00002    -0.00023   +0.00025 (gap-attenuation)
```

- **456 test rows** differ from v1 standalone — real architectural
  diff, not noise.
- Per-class recall vs v1: L+0.00015, M-0.00059, H+0.00052
  (symmetric M↔H trade).

## Read-out

The assumption that "v1's lift is mechanism-driven" splits cleanly:
1. **Bank carries most of the lift.** LB 0.98106 is far above the
   standalone-natural-cal floor (~0.98000), so the 7-component
   bank IS the dominant lever.
2. **RF L2 buys an extra ~2.3 bp.** RF outscores LR-bagging on the
   same bank by 0.98129 - 0.98106 = 0.00023 LB. Not essential, but
   a real architectural advantage.
3. **Negative-gap pattern holds for L2-substituted meta-stackers**,
   just at attenuated magnitude (-0.00041 vs v1's -0.00066). LEARNINGS
   line 1147-1158 ("OOF can underestimate LB lift for meta-stackers
   built on noisy OOF banks") generalizes beyond the RF-specific case.

## Implication for final-selection

No new orthogonal HEDGE option emerged. Architecturally-distinct L2
(linear vs trees) underperforms RF by a real margin (0.00023 LB) on
this bank. Swapping HEDGE to bagginglr would cost public-LB without
buying meaningful private-orthogonality from PRIMARY.

Current finals stand:
- PRIMARY: `submission_idea4b_selective_override.csv` (LB 0.98150)
- HEDGE: `submission_sklearn_rf_meta_natural_standalone.csv` (LB 0.98129)

## Portable rule candidate

**L2-substitution attenuates the meta-stacker negative-gap.** When
swapping the L2 algorithm on a saturated natural-cal bank, expect:
- bal_acc OOF stays roughly tied (the bank dominates the bias-tuned
  argmax),
- but the negative OOF→LB gap shrinks by ~30-40% if the L2 has
  weaker per-fold variance reduction than RF/ET.

In this comp: linear bagging (LR) attenuated v1's -0.00066 gap to
-0.00041 (37% reduction). Mechanism: meta-stackers gain LB lift
above OOF because at test-time all components see unseen rows
simultaneously and noise cancels out (LEARNINGS L1147-1158); the
extent of cancellation depends on how aggressively the L2 averages
neighboring training rows. RF/ET aggregate via per-tree bootstraps
that span the full feature basis; linear bagging averages within
random feature subsets and so propagates more of the per-component
OOF noise into the final prediction.

48th saturation at LB 0.98129 floor for non-override standalones.
