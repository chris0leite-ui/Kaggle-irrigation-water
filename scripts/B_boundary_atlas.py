"""B — Decision-boundary atlas (v2: per-cell 7-feature LR for P(deviate)).

Goal: extract row-level "P(model deviates from cell-majority class)" feature
that captures within-cell multi-axis flip signal.

Mechanism:
  1. Reconstruct LB-best 4-stack primary (OOF + test).
  2. Compute 6-bit cell_id from rule features.
  3. For each cell with sufficient deviations (n_dev >= 10):
       Target = I[primary_argmax != cell_majority]
       7-feature LR (StandardScaled non-rule numerics) → P(deviate)
       Per train row in cell: LR.predict_proba(row)[:, 1] = feature value
       Per test row in cell: same.
       Cells with insufficient signal default to 0.0 (no contribution).
  4. Save oof_boundary_atlas.npy (n_train, 1) and test_boundary_atlas.npy.
  5. Optional: also save per-axis 1D version (unused if v2 is the lever).

Outputs:
  scripts/artifacts/oof_boundary_atlas.npy  (n_train, 1)  P(deviate)
  scripts/artifacts/test_boundary_atlas.npy (n_test, 1)  P(deviate)
  scripts/artifacts/B_boundary_atlas_results.json

SMOKE=1: train=20k subsample (~30s wall).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    build_lbbest_stack, iso_cal, load_y, normed, ART, BIAS,
)

SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_smoke" if SMOKE else ""
SEED = 42

NON_RULE_NUMS = [
    "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
    "Soil_pH", "Field_Area_hectare", "Sunlight_Hours", "Organic_Carbon",
]


def encode_cell(df: pd.DataFrame) -> np.ndarray:
    """Returns 6-bit cell_id ∈ [0, 64) per row."""
    dry = (df["Soil_Moisture"].values < 25).astype(np.int8)
    norain = (df["Rainfall_mm"].values < 300).astype(np.int8)
    hot = (df["Temperature_C"].values > 30).astype(np.int8)
    windy = (df["Wind_Speed_kmh"].values > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    kc = np.isin(df["Crop_Growth_Stage"].astype(str).values,
                 ("Flowering", "Vegetative")).astype(np.int8)
    cell = (dry | (norain << 1) | (hot << 2) | (windy << 3)
            | (nomulch << 4) | (kc << 5)).astype(np.int8)
    return cell


def log(msg: str) -> None:
    print(f"[B {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    log("loading train + test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = load_y()
    log(f"train={len(train):,}  test={len(test):,}")

    log("reconstructing LB-best 4-stack primary (OOF + test)")
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t], np.array([0.7, 0.3]))
    primary_o_argmax = (np.log(np.clip(primary_o, 1e-12, 1.0)) + BIAS).argmax(1)
    primary_t_argmax = (np.log(np.clip(primary_t, 1e-12, 1.0)) + BIAS).argmax(1)
    from sklearn.metrics import balanced_accuracy_score
    log(f"  primary OOF tuned bal_acc = {balanced_accuracy_score(y, primary_o_argmax):.5f}  "
        "(should be ~0.98084)")

    log("computing 6-bit cell_id")
    train_cells = encode_cell(train)
    test_cells = encode_cell(test)

    log("fitting per-cell 7-feature LR for P(deviate from cell-majority)")
    train_feat = np.zeros((len(train),), dtype=np.float32)
    test_feat = np.zeros((len(test),), dtype=np.float32)

    cell_majority = {}
    for c in range(64):
        in_cell = train_cells == c
        if in_cell.sum() < 30:
            continue
        cell_majority[c] = int(np.bincount(y[in_cell], minlength=3).argmax())

    fitted_cells = []
    skipped_cells = []
    for c in cell_majority:
        tr_in = train_cells == c
        te_in = test_cells == c
        cell_maj = cell_majority[c]
        deviates = (primary_o_argmax[tr_in] != cell_maj).astype(np.int8)
        n_dev = int(deviates.sum())
        n_clean = int((deviates == 0).sum())
        if n_dev < 10 or n_clean < 10:
            skipped_cells.append({"cell": c, "n_dev": n_dev, "n_clean": n_clean})
            continue

        X_tr = train.iloc[np.where(tr_in)[0]][NON_RULE_NUMS].to_numpy(dtype=np.float64)
        X_te = test.iloc[np.where(te_in)[0]][NON_RULE_NUMS].to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.empty((0, 7))

        lr = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs",
                                class_weight="balanced", random_state=SEED)
        try:
            lr.fit(X_tr_s, deviates)
        except Exception as e:
            skipped_cells.append({"cell": c, "n_dev": n_dev, "err": str(e)[:60]})
            continue
        # Per-row P(deviate). For train rows in this cell: in-sample fit (slight
        # leak; mitigated by primary OOF being already leak-free per row).
        train_feat[tr_in] = lr.predict_proba(X_tr_s)[:, 1].astype(np.float32)
        if len(X_te) > 0:
            test_feat[te_in] = lr.predict_proba(X_te_s)[:, 1].astype(np.float32)

        fitted_cells.append({
            "cell": c, "n_in": int(tr_in.sum()), "n_dev": n_dev,
            "deviation_rate": float(n_dev / tr_in.sum()),
            "auc_train": float(_auc(deviates, lr.predict_proba(X_tr_s)[:, 1])),
        })

    log(f"  fitted={len(fitted_cells)} cells; skipped={len(skipped_cells)} cells")

    # Reshape to (n, 1) so feature can be appended as a single numeric col.
    train_out = train_feat.reshape(-1, 1)
    test_out = test_feat.reshape(-1, 1)

    log("saving features")
    np.save(ART / f"oof_boundary_atlas{SUFFIX}.npy", train_out)
    np.save(ART / f"test_boundary_atlas{SUFFIX}.npy", test_out)

    out = {
        "smoke": SMOKE,
        "n_cells_with_majority": len(cell_majority),
        "n_cells_fitted": len(fitted_cells),
        "n_cells_skipped": len(skipped_cells),
        "fitted_cells": fitted_cells,
        "skipped_cells_count": len(skipped_cells),
        "feature_summary": {
            "train_nonzero_rate": float((train_feat > 0).mean()),
            "train_mean": float(train_feat.mean()),
            "train_std": float(train_feat.std()),
            "train_p99": float(np.percentile(train_feat, 99)),
            "test_nonzero_rate": float((test_feat > 0).mean()),
            "test_mean": float(test_feat.mean()),
        },
        "elapsed_seconds": time.time() - t0,
    }
    out_path = ART / f"B_boundary_atlas{SUFFIX}_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log(f"saved {out_path.name}")

    log("=" * 60)
    log(f"FITTED CELLS: {len(fitted_cells)} / {len(cell_majority)}")
    if fitted_cells:
        aucs = [c["auc_train"] for c in fitted_cells]
        log(f"  in-sample AUC range: {min(aucs):.3f} - {max(aucs):.3f}, mean {np.mean(aucs):.3f}")
        # Top 5 cells by deviation rate
        top5 = sorted(fitted_cells, key=lambda c: c["deviation_rate"], reverse=True)[:5]
        log("  top-5 cells by deviation rate:")
        for c in top5:
            log(f"    cell {c['cell']:2d}  n_in={c['n_in']:6d}  "
                f"dev_rate={c['deviation_rate']:.3f}  auc_train={c['auc_train']:.3f}")
    log(f"feature: train_nonzero={int((train_feat > 0).sum()):,} ({(train_feat > 0).mean()*100:.1f}%)  "
        f"test_nonzero={int((test_feat > 0).sum()):,} ({(test_feat > 0).mean()*100:.1f}%)")
    log(f"feature mean train={train_feat.mean():.3f}  std={train_feat.std():.3f}  "
        f"p99={np.percentile(train_feat, 99):.3f}")
    log(f"total wall = {(time.time()-t0):.1f}s")


def _auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


if __name__ == "__main__":
    main()
