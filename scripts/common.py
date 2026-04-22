"""Shared utilities for the irrigation pipeline.

Extracted from the copy-pasted versions in benchmark_*, xgb_*, blend_*.
New scripts should import from here; existing scripts are left untouched
to avoid regenerating committed OOFs.

Pinned conventions (matches OOFS.md):
    5-fold StratifiedKFold, shuffle=True, random_state=42
    classes: Low=0, Medium=1, High=2
    OOF shape: (630_000, 3)  (val-fold only; rows sum to 1)
    test shape: (270_000, 3) (averaged across 5 folds; rows sum to 1)

Sparse-carrier exception: oof_xgb_spec_678.npy is zero-filled on the
~91% of rows outside dgp_score ∈ {6,7,8}; only the in-domain rows
carry signal. Consumers must index with `np.isin(dgp_score, (6,7,8))`
before reading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

SEED = 42
N_FOLDS = 5
CLASSES = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
DGP_THRESHOLDS = dict(sm=25.0, rf=300.0, tc=30.0, ws=10.0)


def fast_bal_acc(y: np.ndarray, pred: np.ndarray, n_class: int = 3,
                 class_counts: np.ndarray | None = None) -> float:
    """Vectorised macro-recall. ~30x faster than sklearn on 630k rows."""
    if class_counts is None:
        class_counts = np.bincount(y, minlength=n_class)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(n_class)], dtype=np.int64)
    return float((hit / np.maximum(class_counts, 1)).mean())


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """43-feature dist set used across benchmark_dist, xgb_specialist_678, routed_v3.

    Every output is a deterministic function of the raw input columns — safe
    to compute pre-fold. DGP score formula: 2*(dry+norain) + hot+windy+nomulch + Kc.
    """
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).to_numpy()
    rf = out["Rainfall_mm"].astype(float).to_numpy()
    tc = out["Temperature_C"].astype(float).to_numpy()
    ws = out["Wind_Speed_kmh"].astype(float).to_numpy()

    dry = (sm < DGP_THRESHOLDS["sm"]).astype(np.int8)
    norain = (rf < DGP_THRESHOLDS["rf"]).astype(np.int8)
    hot = (tc > DGP_THRESHOLDS["tc"]).astype(np.int8)
    windy = (ws > DGP_THRESHOLDS["ws"]).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).to_numpy() == "No").astype(np.int8)
    stage = out["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - DGP_THRESHOLDS["sm"]).astype(np.float32)
    out["rf_dist"] = (rf - DGP_THRESHOLDS["rf"]).astype(np.float32)
    out["tc_dist"] = (tc - DGP_THRESHOLDS["tc"]).astype(np.float32)
    out["ws_dist"] = (ws - DGP_THRESHOLDS["ws"]).astype(np.float32)
    for col in ("sm_dist", "rf_dist", "tc_dist", "ws_dist"):
        out[col.replace("_dist", "_abs")] = np.abs(out[col].to_numpy()).astype(np.float32)

    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].to_numpy()),
        np.abs(out["score_dist_mid_high"].to_numpy()),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].to_numpy(), out["rf_abs"].to_numpy(),
         out["tc_abs"].to_numpy(), out["ws_abs"].to_numpy()]
    ).astype(np.float32)

    out["sm_x_rf"] = (out["sm_dist"].to_numpy() * out["rf_dist"].to_numpy()).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].to_numpy() * out["ws_dist"].to_numpy()).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].to_numpy() * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].to_numpy() * kc.astype(np.float32)).astype(np.float32)

    return out


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
                  high_grid_wide: bool = True, coarse: bool = False,
                  eps: float = 1e-9):
    """Coord-ascent per-class log-bias on full OOF (val-fold only, no leakage).

    Grid `-3..+6` for High when `high_grid_wide=True` — empirical optimum is
    ~+3.4 under severe class imbalance (see LEARNINGS.md).

    Returns (bias, best_tuned_bal_acc).
    """
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)
    best = fast_bal_acc(y, (log_oof + bias).argmax(1), class_counts=cc)
    if coarse:
        grid_default = np.linspace(-2.0, 2.0, 21)
        grid_high = np.linspace(-1.0, 5.0, 25) if high_grid_wide else grid_default
        max_iter = 6
    else:
        grid_default = np.linspace(-3.0, 3.0, 61)
        grid_high = np.linspace(-3.0, 6.0, 91) if high_grid_wide else grid_default
        max_iter = 25
    for _ in range(max_iter):
        improved = False
        for k in range(3):
            grid = grid_high if k == 2 else grid_default
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(1), class_counts=cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend(oofs: list[np.ndarray], weights: np.ndarray,
              eps: float = 1e-9) -> np.ndarray:
    """Weighted geometric mean in probability space."""
    w = weights / weights.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def load_oof_pair(name: str, art_dir: str = "scripts/artifacts"):
    """Load (oof, test) pair by base name (without prefix/extension).

    Raises on shape mismatch against the pinned convention. Sparse-carrier
    artefacts (e.g., xgb_spec_678) are returned as-is; use the dgp_score
    mask to read only in-domain rows.
    """
    from pathlib import Path
    art = Path(art_dir)
    oof = np.load(art / f"oof_{name}.npy")
    test = np.load(art / f"test_{name}.npy")
    if oof.shape != (630_000, 3):
        raise ValueError(f"oof_{name}.npy shape {oof.shape}, expected (630000, 3)")
    if test.shape != (270_000, 3):
        raise ValueError(f"test_{name}.npy shape {test.shape}, expected (270000, 3)")
    return oof, test
