"""T1 — Score-conditional per-bucket log-bias on LB-best 4-stack.

Currently the LB-best 4-stack uses a single global log-bias [1.43, 1.47, 3.40].
Per_bin_blend.py at 5 buckets × 30 params overfit (in-sample +0.00009 → nested
−0.00031). Per-fold log-bias search at 5 bins is too high-DoF for the per-bin
signal density.

This T1 splits into ONLY 2 buckets:
  A: dgp_score ∈ {3, 6, 7, 8}   ~31% of rows, ~83% of error mass
  B: dgp_score ∈ {others}        ~69% of rows, ~17% of error mass

6 free params (3 classes × 2 buckets) — between global underfit (3 params)
and per-bin overfit (30 params). Nested 5-fold CV reports honest gain.

Outputs:
  oof_t1_per_bucket_bias.npy      (LB-4 with per-bucket bias applied)
  test_t1_per_bucket_bias.npy
  t1_per_bucket_bias_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, build_lbbest_stack, iso_cal, load_y, normed,
)

DATA = Path("data")
SEED = 42
N_FOLDS = 5
BUCKET_A_SCORES = {3, 6, 7, 8}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values,
                          ("Flowering", "Vegetative")), 2, 0)
    return 2 * (dry + norain) + (hot + windy + nomulch) + kc


def per_bucket_tune(p_oof, y, bucket_mask):
    """Tune log-bias on bucket_mask rows only. Returns 3-vector."""
    prior = np.bincount(y[bucket_mask], minlength=3).astype(np.float32) / max(int(bucket_mask.sum()), 1)
    bias, _ = tune_log_bias(p_oof[bucket_mask], y[bucket_mask], prior)
    return bias


def apply_per_bucket(p, bias_a, bias_b, bucket_mask):
    log_p = np.log(np.clip(p, 1e-12, 1))
    out = log_p.copy()
    out[bucket_mask] += bias_a
    out[~bucket_mask] += bias_b
    return out.argmax(1)


def main():
    y = load_y()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    score_tr = dgp_score(train)
    score_te = dgp_score(test)
    bucket_tr = np.isin(score_tr, list(BUCKET_A_SCORES))
    bucket_te = np.isin(score_te, list(BUCKET_A_SCORES))
    log(f"bucket A train: {bucket_tr.sum():,} ({bucket_tr.mean():.1%})")
    log(f"bucket A test:  {bucket_te.sum():,} ({bucket_te.mean():.1%})")

    # Build LB-best 4-stack OOF + test.
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    w4 = np.array([0.70, 0.30])
    lb4_o = normed(log_blend([lb3_o, meta_o_iso], w4))
    lb4_t = normed(log_blend([lb3_t, meta_t_iso], w4))
    log(f"LB4 OOF (global bias) = {balanced_accuracy_score(y, (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)):.5f}")

    # In-sample per-bucket bias (full OOF — optimistic, should be HIGHER than nested).
    bias_a_full = per_bucket_tune(lb4_o, y, bucket_tr)
    bias_b_full = per_bucket_tune(lb4_o, y, ~bucket_tr)
    pred_in = apply_per_bucket(lb4_o, bias_a_full, bias_b_full, bucket_tr)
    bal_in = balanced_accuracy_score(y, pred_in)
    log(f"in-sample per-bucket OOF = {bal_in:.5f}  "
        f"bias_A={bias_a_full.round(4).tolist()}  bias_B={bias_b_full.round(4).tolist()}")

    # NESTED 5-fold: tune bias on outer-train fold rows, apply to outer-val fold.
    log("running nested 5-fold...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    nested_pred = np.zeros(len(y), dtype=np.int32)
    fold_biases = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        # Tune bias on outer-train rows only.
        b_a = per_bucket_tune(lb4_o[tr_idx], y[tr_idx], bucket_tr[tr_idx])
        b_b = per_bucket_tune(lb4_o[tr_idx], y[tr_idx], ~bucket_tr[tr_idx])
        fold_biases.append((b_a.tolist(), b_b.tolist()))
        # Apply to outer-val.
        log_p = np.log(np.clip(lb4_o[va_idx], 1e-12, 1))
        for i, idx in enumerate(va_idx):
            log_p[i] += b_a if bucket_tr[idx] else b_b
        nested_pred[va_idx] = log_p.argmax(1)
    bal_nested = balanced_accuracy_score(y, nested_pred)
    log(f"NESTED per-bucket OOF = {bal_nested:.5f}")
    log(f"  Δ vs LB4 global = {bal_nested - 0.98084:+.5f}")
    log(f"  in-sample inflation = {bal_in - bal_nested:+.5f}")

    # Per-class recall delta.
    pred_global = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    for c, name in enumerate(("Low", "Medium", "High")):
        m = (y == c)
        rg = (pred_global[m] == c).mean()
        rn = (nested_pred[m] == c).mean()
        log(f"  recall {name}: global={rg:.4f}  nested={rn:.4f}  Δ={rn-rg:+.4f}")

    # Apply best per-bucket bias to test.
    log_t = np.log(np.clip(lb4_t, 1e-12, 1))
    test_pred_idx = np.zeros(len(lb4_t), dtype=np.int32)
    for i in range(len(lb4_t)):
        test_pred_idx[i] = (log_t[i] + (bias_a_full if bucket_te[i] else bias_b_full)).argmax()
    log(f"test class dist (bucketed bias): {np.bincount(test_pred_idx)}")
    log(f"test class dist (global bias):   {np.bincount((log_t + BIAS).argmax(1))}")

    # Save.
    np.save(ART / "oof_t1_per_bucket_bias_lb4.npy", lb4_o)  # base for reference
    np.save(ART / "test_t1_per_bucket_bias_lb4.npy", lb4_t)
    res = {
        "lb4_global_bal_acc": float(balanced_accuracy_score(
            y, (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1))),
        "in_sample_bal_acc": float(bal_in),
        "nested_bal_acc": float(bal_nested),
        "delta_vs_global_nested": float(bal_nested - 0.98084),
        "in_sample_inflation": float(bal_in - bal_nested),
        "bias_A_full": bias_a_full.tolist(),
        "bias_B_full": bias_b_full.tolist(),
        "fold_biases": fold_biases,
        "bucket_A_scores": list(BUCKET_A_SCORES),
        "bucket_A_train_pct": float(bucket_tr.mean()),
        "bucket_A_test_pct": float(bucket_te.mean()),
    }
    (ART / "t1_per_bucket_bias_results.json").write_text(json.dumps(res, indent=2))
    log("saved t1_per_bucket_bias_results.json")


if __name__ == "__main__":
    main()
