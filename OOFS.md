# OOF / Test Artifact Manifest

Committed prediction arrays for cross-branch blending. Every file lives
at `scripts/artifacts/{name}.npy` and is a float64 numpy array.

## Fold & class conventions (identical across ALL committed artifacts)

- **CV split**: `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the competition train target.
- **OOF shape**: `(630_000, 3)` — one row per competition-train row, val-fold predictions only.
- **Test shape**: `(270_000, 3)` — one row per competition-test row, averaged across 5 fold predictions (unless noted).
- **Class index**: `0 = Low, 1 = Medium, 2 = High`.
- **Probabilities** sum to ~1.0 per row (softmax or log-blend renormalised).
- **Log-bias** is NOT baked into probs — it's applied at argmax time. Each blender should re-tune log-bias on their meta-blend using `coord-ascent on OOF`.

## Load & blend recipe

```python
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

y = pd.read_csv("data/train.csv")["Irrigation_Need"].map({"Low":0,"Medium":1,"High":2}).values
prior = np.bincount(y) / len(y)

oof = np.load("scripts/artifacts/oof_hybrid_lgbmxgb_blend.npy")
test = np.load("scripts/artifacts/test_hybrid_lgbmxgb_blend.npy")

# Tune log-bias
def tune(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior); best = balanced_accuracy_score(y, (lp + b).argmax(1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy(); sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]; best = sc[j]; imp = True
        if not imp: break
    return b, best

bias, tuned = tune(oof, y, prior)
final_pred = (np.log(np.clip(test, 1e-9, 1.0)) + bias).argmax(1)
```

To **blend** with another branch's OOFs in log-space (geometric mean):

```python
ours = np.load("scripts/artifacts/oof_hybrid_lgbmxgb_blend.npy")
theirs = np.load("...")  # their OOF, same shape, same fold convention
w = 0.5  # sweep
blend = np.exp(w * np.log(np.clip(ours, 1e-9, 1.0)) +
               (1-w) * np.log(np.clip(theirs, 1e-9, 1.0)))
blend /= blend.sum(1, keepdims=True)
bias, tuned_blend = tune(blend, y, prior)
```

## Committed artifacts

All files below live at `scripts/artifacts/`. Sizes: OOF = 15 MB, test = 6 MB.

### Current best (recommended for blending as a single contribution)

| File | OOF tuned bal_acc | LB | Notes |
|---|---|---|---|
| `oof_greedy_blend.npy` + `oof_xgb_nonrule.npy` | **0.97421** (log-blend α=0.15) | **0.97352** | **Current best (2026-04-21).** Greedy 3-way blend (`0.45 hybrid_v3 + 0.40 routed_v3 + 0.15 spec_678`) log-blended with non-rule-features-only XGB at α_nonrule=0.15, FIXED greedy bias [0.1324, 0.5689, 3.4008]. See `scripts/nonrule_features_only.py` for provenance. |
| `oof_greedy_blend.npy` / `test_greedy_blend.npy` | 0.97375 | 0.97296 | Greedy 3-way log-blend (reconstructed from components). See `scripts/greedy_binhigh_minimal.py`. |
| `oof_hybrid_binhigh.npy` / `test_hybrid_binhigh.npy` | 0.97398 | 0.97212 (**overfit**) | Binary-High head stacked on hybrid_lgbmxgb_blend via logit-add λ=+0.60, with bias retuned per blend. Selection-overfit; keep for reference, do not use as a blend leg. `scripts/binary_high_head.py`. |
| `oof_hybrid_lgbmxgb_blend.npy` / `test_hybrid_lgbmxgb_blend.npy` | 0.97362 | — | Log-blend: `0.75 × hybrid_v3 + 0.25 × (LGBM×0.45 + XGB×0.55)`. `scripts/blend_hybrid_lgbmxgb.py`. |

### Hybrid components (if the blender wants to construct its own hybrid variant)

| File | OOF tuned | Notes |
|---|---|---|
| `oof_xgb_dist_routed_v3.npy` / `test_xgb_dist_routed_v3.npy` | 0.97332 | Main routed XGB — trained without scores {0,1,2}; those scores routed to rule (Low) at inference. 43-feature dist set. `scripts/xgb_dist_routed_v3.py`. |
| `oof_xgb_spec_678.npy` / `test_xgb_spec_678.npy` | 0.97352 (in hybrid) | Specialist XGB trained only on `dgp_score ∈ {6,7,8}` (56 k rows, 69 % Med / 31 % High). Same 43-feature set; only populated at spec-domain rows, others zero. Hybrid uses it to override main on those rows. `scripts/xgb_specialist_678.py`. |

To reconstruct hybrid-v3: `hybrid[spec_rows] = spec[spec_rows]; hybrid[other_rows] = main[other_rows]` where `spec_rows` = rows with `dgp_score in {6,7,8}`. Tuned bal_acc = 0.97352.

### Base learners (for re-blending)

| File | OOF tuned | Notes |
|---|---|---|
| `oof_xgb_vanilla_dist.npy` / `test_xgb_vanilla_dist.npy` | 0.97304 | XGB-dist trained on all 630 k rows, no routing. 43-feature dist set. Emitted as by-product of `scripts/xgb_dist_routed_v7.py`. |
| `oof_lgbm_te_orig.npy` / `test_lgbm_te_orig.npy` | 0.97270 | LGBM-dist + TE from 10k original (null TE lift; proxy for vanilla LGBM-dist). `scripts/benchmark_te_orig.py`. |
| `oof_xgb_bin_high.npy` / `test_xgb_bin_high.npy` | AUC 0.9987 | 1-D binary 'is High?' head. Same 43-feature dist set, 5-fold stratified on 3-class y, `binary:logistic`. Shape `(N,)` not `(N, 3)` — use with the hybrid via logit-add or mix. `scripts/binary_high_head.py`. **Lever dead after fixed-bias falsification.** |
| `oof_xgb_nonrule.npy` / `test_xgb_nonrule.npy` | 0.42965 argmax / 0.56966 tuned standalone | 3-class XGB on 13 non-rule features only (`Soil_Type, Soil_pH, Organic_Carbon, Electrical_Conductivity, Humidity, Sunlight_Hours, Crop_Type, Season, Irrigation_Type, Water_Source, Field_Area_hectare, Previous_Irrigation_mm, Region`). Near-random standalone, but log-blended at α=0.15 with greedy lifts LB +0.00056. `scripts/nonrule_features_only.py`. |

## Other OOFs (NOT committed — regenerate with the listed script)

Per-experiment OOFs from the null results in `scripts/artifacts/`. Not
committed to keep repo size down. Each is reproducible in 5–60 min
depending on script:

| File | OOF tuned | Script |
|---|---|---|
| `oof_catboost_dist.npy` | 0.97128 | `legacy/null/benchmark_catboost_dist.py` |
| `oof_xgb_dist_routed_v6.npy` | 0.97320 | `legacy/null/xgb_dist_routed_v6.py` |
| `oof_xgb_dist_routed_v7.npy` | 0.97288 | `scripts/xgb_dist_routed_v7.py` |
| `oof_lgbm_te_oof.npy` | 0.97271 | `legacy/null/benchmark_te_oof.py` |
| `oof_lgbm_rule_distill.npy` | 0.97219 | `legacy/null/rule_distillation.py` |
| `oof_pseudo_hybrid_tau95.npy` | 0.97332 | `legacy/null/pseudo_label_hybrid.py` |
| `oof_xgb_spec_3.npy` | — (null) | `legacy/null/xgb_specialist_3.py` |
| `oof_xgb_spec_46.npy` | — (null) | `legacy/null/xgb_specialist_46.py` |
| `oof_xgb_spec_678_aug_{w10,w03}.npy` | — (null) | `legacy/null/xgb_specialist_678_aug.py` |
| `oof_per_cell_lr.npy` | 0.96280 | `legacy/null/per_cell_lr.py` |

## Regenerating committed artifacts

To regenerate all committed OOFs from scratch:

```bash
python scripts/benchmark_xgb_dist.py          # -> oof_xgb_vanilla_dist via routed_v7
python scripts/xgb_dist_routed_v3.py          # -> oof_xgb_dist_routed_v3
python scripts/xgb_specialist_678.py          # -> oof_xgb_spec_678
python scripts/benchmark_te_orig.py           # -> oof_lgbm_te_orig
python scripts/blend_hybrid_lgbmxgb.py        # -> oof_hybrid_lgbmxgb_blend (needs above)
```

Total wall-clock ~90–120 min on 8 cores.

## OOF→LB calibration

Current-best committed artifact (`oof_hybrid_lgbmxgb_blend.npy`) is
**not** yet validated on LB. Expected LB from prior OOF→LB gap ≈ 0.0008:
OOF 0.97362 − 0.0008 ≈ **~0.9728**, close to our current LB-best 0.97271.
