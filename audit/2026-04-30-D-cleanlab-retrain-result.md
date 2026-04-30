# 2026-04-30 — D: cleanlab + retrain v1 result (45th saturation, no LB probe)

## Mechanism

Use cleanlab's `find_label_issues` (filter_by="prune_by_noise_rate") with
**v1 RF natural OOF probabilities as the teacher posterior** to flag
likely-mislabeled TRAIN rows. v1 is the LB-best standalone (LB 0.98129) so
its 5-fold OOF is the most-LB-aligned posterior for noise-transition
estimation.

Two retrain variants:
- **D-drop**:    remove flagged rows from training, retrain v1 on 628,369 rows.
- **D-relabel**: replace flagged rows' y with DGP rule_pred, retrain on 630k.

v1 architecture preserved exactly: sklearn RF n_est=500, max_depth=12,
class_weight=None, bootstrap=True, on the 11-component natural-cal bank +
14 distance/rule meta features (47-feature input matrix).

Distinct from prior cleanlab artifact (which used LB-best 2-way blend as
teacher, NOT v1) and from DROP_DETERMINISTIC (which removed boundary-
anchor rows, the OPPOSITE direction — removing model-confident rows).

## Cleanlab diagnostic (D1)

```
flagged: 1631 / 630000 (0.2589%)
99.6934% of flagged rows are y != rule_pred (host NN-flip signal)
concentrated at boundary scores 2-7 (the rule-flip band)
H-class flag rate 1.36% (highest); M 0.32%; L 0.15%
```

The flagged set is heavily enriched for host-NN-flipped rows (99.7% vs 1.6%
global) — cleanlab is finding the right targets.

## Production results

```
                          D-drop           D-relabel
fold 1 raw bal_acc        0.97985          0.98078
fold 2 raw bal_acc        0.97920          0.97881
fold 3 raw bal_acc        0.97873          0.97846
fold 4 raw bal_acc        0.97733          0.97938
fold 5 raw bal_acc        0.97910          0.97714
TUNED MACRO (orig y)      0.98066          0.98068
                          v1 baseline = 0.98063  (Δ: +3e-5 / +5e-5, sub-noise)
```

Fold-by-fold variance (~0.002 between folds) dwarfs the standalone delta.

## Audit gates (D3)

```
                                  D-drop          D-relabel       v1 baseline
G1 standalone tuned (orig y)      0.980632        0.980637        0.980646
   delta                          -0.000015       -0.000010

G4 per-class recall
   L                              0.99451         0.99450         0.99490
   M                              0.96947         0.96807         0.96751
   H                              0.97791         0.97934         0.97953
   delta vs v1: L/M/H             -0.0004/+0.0020/-0.0016    -0.0004/+0.0006/-0.0002

G2 Jaccard vs v1                  0.9996          0.9996
   Jaccard vs all 14 banks        0.9958-0.9996 (xgb_nonrule 0.6524 ablation artifact)

G3 best v1+D 2-comp meta          α=0.10 +6.3e-5  α=0.15 +3.6e-5

G5 test-side flips vs v1 test     25 / 270000     27 / 270000
                                  (0.009%)        (0.010%)
```

## Diagnosis

**Both modes essentially recover v1 itself.**

1. **Cleanlab flag set is too small** (1631 rows = 0.26% of TRAIN) to
   meaningfully alter the RF meta-stacker's split structure at
   max_depth=12. The RF's 500 trees average over enough randomized
   splits to absorb sub-1% data-distribution shifts.

2. **Test-side behavioral change is negligible**: only 25-27 of 270k
   test predictions flip vs v1 (0.01%). The retrain didn't move the
   decision boundary in any practically-meaningful way.

3. **Per-class recall trade is a wash**: D-drop trades M up (+0.0020)
   vs H down (-0.0016) — net ≈ 0. D-relabel is similar but smaller
   magnitude. The flagged H rows ARE host-NN-flips (rule says H, y is M),
   and removing them lifts M-recall while hurting H-recall on the
   genuinely-difficult boundary cases.

4. **The meta-stacker already has rule structure as input**: it
   consumes `dgp_score`, `rule_pred`, `score_dist_low_mid`,
   `score_dist_mid_high`, and per-axis distance features. The model
   has *already learned* how to handle boundary regions — explicit
   row removal/relabel doesn't add new information.

5. **Best v1+D meta lift is +6.3e-5** (D-drop α=0.10) — well below
   fold-noise (~0.002 std-fold) and within the documented stacking-
   inflation regime. Even if this transferred 1:1 to LB, it would
   put us at ~0.98135 — still below LB-best 0.98150.

## Connection to prior saturations

This adds to the saturation list as the **first cleanlab+retrain mechanism**
tested (distinct from `cleanlab_diagnose.py` which only diagnosed without
retraining, and from DROP_DETERMINISTIC which removed boundary-anchor rows
in the opposite direction).

The training-data-cleanliness mechanism class is now closed for v1:
- DROP_DETERMINISTIC: removed boundary-anchor rows → regression
- D-drop: removed cleanlab-flagged rows → +3e-5 (sub-noise)
- D-relabel: replaced flagged y with rule_pred → +5e-5 (sub-noise)

For a meaningfully different result, would need to either:
- Use a base-learner architecture more sensitive to small data shifts
  (e.g., kNN, SVM) — but those are on the saturation list
- Flag a much larger fraction (e.g., all 10,304 rows where y != rule_pred)
  — but then we'd be replacing TRAIN's noise structure, which is the
  same noise structure the host placed in TEST → systematically biased

## Saturation count: 45

## LB budget

No LB probe spent. Production wall: ~28 min D-drop + ~27 min D-relabel +
~1 min cleanlab + ~1 min audit = ~57 min CPU total.

## Implications for remaining-untried list

The training-data-cleanliness mechanism is closed. Genuinely-new untried
mechanisms shrink to:
1. Sonnet-grade LLM judge (~$10-20 API cost, not in this container)
2. NN inversion / DGP archaeology (host NN architecture unknown)

Both are outside this session's scope.
