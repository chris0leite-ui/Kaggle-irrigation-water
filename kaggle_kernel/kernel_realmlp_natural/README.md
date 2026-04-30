# kernel_realmlp_natural — first NN test of the calibration finding

Tests whether the 2026-04-29 RF natural meta-stacker calibration finding
(naturally-calibrated bagging meta hit LB 0.98129, +0.00035 over prior
4-stack) carries over to NN base learners. The 15+ prior NN nulls all
exhibited magnitude trap (errs +5-30% over anchor); hypothesis is that
class-balanced training over-pushes High predictions at depth-limited
capacity.

## Diff vs `kernel_realmlp/realmlp_pytabkit.py`

```
                          baseline (n_ens=1)        natural-cal probe
ORIG augmentation         none                       concat 10k @ sw=0.5
TargetEncoder cv          2                          5
class_weight              none (already)             none
n_ens                     1                          1 (kept; LB-positive)
n_epochs                  40                         40
```

## SMOKE-first

Per CLAUDE.md GPU 1h cap rule, push SMOKE first by flipping `IS_SMOKE = True`
at the top of `realmlp_natural.py` (line 86). Expected smoke wall ~5 min.
Then flip to `IS_SMOKE = False` and push production (~40-50 min).

## Outputs

```
oof_realmlp_natural.npy            (630_000, 3)
test_realmlp_natural.npy           (270_000, 3)
realmlp_natural_results.json       per-fold + drift diagnostic
submission_realmlp_natural_tuned.csv
```

## Decision gate (post-pull)

Run `python scripts/blend_realmlp_natural.py` after
`kaggle kernels output chrisleitescha/irrigation-realmlp-natural -p scripts/artifacts/`.

Reports:
- bias drift from -log(prior); natural-cal PASS if `max|drift| ≤ 0.3`
- comparison to baseline RealMLP n_ens=1 drift (was [0.70, 0.50, 0.00])
- standalone tuned OOF + errors at recipe bias
- Jaccard vs LB-best 4-stack + rawashishsin v3 + RF natural meta
- Fixed-bias α-sweep vs all three anchors
- Verdict: deploy as RF natural meta bank input (next stage) if
  `errs ≤ 1.05 × anchor` AND drift improved vs baseline
