"""Audit-#3: Per-row blend weight calibration via AUC-0.63 residual head.

Different delivery mechanism for the 10k-anchor signal that bypasses the
meta-stacker bank entirely. The AUC-0.63 residual head (from
n5b_followup_residual_auc.py) predicts P(primary_error | x) per row.
Use this scalar to MODIFY the LB-best 4-stack's blend weight per row:

  alpha_row = clip(0.30 + lambda * (P_err_row - mean(P_err)), 0.05, 0.70)
  pred = argmax( log_blend(lb3, meta_iso, [1-alpha_row, alpha_row]) + bias )

When P(primary_error|x) is high, increase weight on meta_iso (trust the
meta MORE). When P_err low, decrease meta weight (trust primary more).
The MEAN-CENTER ensures the blend is anchored at alpha=0.30 baseline.

Steps:
  1) Train binary XGB residual head on FULL train (the OOF version saved
     as oof_n5b_residual_auc.npy is per-fold; for test we need full-fit).
  2) Compute P_err_test for each test row.
  3) Sweep lambda in {0.05, 0.10, 0.20, 0.30, 0.50, 1.00}; for each,
     compute OOF macro + per-class recall.
  4) If any lambda passes +0.0003 + per-class guardrail, emit submission.

Wall: ~5 min CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
SEED = 42
N_FOLDS = 5
LABELS = ["Low", "Medium", "High"]


def main() -> None:
    print("[1] Loading PRIMARY components + N5b features...")
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)

    # Reuse saved OOF residual prediction (already 5-fold leak-free)
    p_err_o = np.load(ART / "oof_n5b_residual_auc.npy").astype(np.float32)
    print(f"    OOF P_err: mean={p_err_o.mean():.4f} std={p_err_o.std():.4f}")

    # Build features for the binary head (matches n5b_followup_residual_auc.py)
    ood = np.load(ART / "oof_ood3_train.npy").astype(np.float32)
    knn = np.load(ART / "oof_knn10k_train.npy").astype(np.float32)
    X_tr = np.concatenate([ood, knn], axis=1)
    ood_te = np.load(ART / "test_ood3.npy").astype(np.float32)
    knn_te = np.load(ART / "test_knn10k.npy").astype(np.float32)
    X_te = np.concatenate([ood_te, knn_te], axis=1)

    # Compute primary argmax to define residual target
    BIAS_ = BIAS  # alias
    def lb(p, b=BIAS_): return np.log(np.clip(p, 1e-12, 1)) + b
    def macro(p, b=BIAS_): return balanced_accuracy_score(y, lb(p, b).argmax(1))
    def rec(p, b=BIAS_): return recall_score(y, lb(p, b).argmax(1), average=None)

    p_primary_o = np.exp(lb(s3_o) * 0.7 + lb(ms_o_iso) * 0.3)
    p_primary_o = p_primary_o / p_primary_o.sum(1, keepdims=True)
    pred_p = (lb(p_primary_o)).argmax(1)
    residual = (y != pred_p).astype(np.int32)
    print(f"    PRIMARY error count={residual.sum()} ({residual.mean()*100:.2f}%)")

    print("[2] Train binary XGB residual head on FULL train (for test inference)...")
    t0 = time.time()
    dtrain = xgb.DMatrix(X_tr, label=residual)
    xgb_params = dict(
        objective="binary:logistic", eval_metric="auc",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.0, reg_lambda=1.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    # Use median best_iter across folds from earlier OOF
    booster = xgb.train(xgb_params, dtrain, num_boost_round=80, verbose_eval=0)
    p_err_t = booster.predict(xgb.DMatrix(X_te)).astype(np.float32)
    print(f"    test P_err: mean={p_err_t.mean():.4f} std={p_err_t.std():.4f}  ({time.time()-t0:.1f}s)")

    print("\n[3] Per-row weight calibration sweep")
    print(f"    baseline alpha=0.30 fixed: OOF={macro(p_primary_o):.5f}")
    p_err_o_centered = p_err_o - p_err_o.mean()
    p_err_t_centered = p_err_t - p_err_t.mean()
    base_macro = macro(p_primary_o)
    base_rec = rec(p_primary_o)

    def per_row_blend(s3, ms, alpha_row):
        # alpha_row shape (N,)
        a = alpha_row[:, None]  # (N, 1)
        log_blended = (1 - a) * np.log(np.clip(s3, 1e-12, 1)) + a * np.log(np.clip(ms, 1e-12, 1))
        e = np.exp(log_blended - log_blended.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    out = {"baseline_oof": float(base_macro),
            "baseline_rec": base_rec.tolist(), "lambdas": []}
    best_emit = None
    for lam in [-0.50, -0.30, -0.20, -0.10, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
        alpha_row_o = np.clip(0.30 + lam * p_err_o_centered, 0.05, 0.70).astype(np.float32)
        blend_o = per_row_blend(s3_o, ms_o_iso, alpha_row_o)
        m = macro(blend_o); r = rec(blend_o)
        d = m - base_macro; drec = (r - base_rec).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"  λ={lam:+.2f}  α_row range=[{alpha_row_o.min():.3f}, {alpha_row_o.max():.3f}]  "
              f"OOF={m:.5f}  Δ={d:+.5f}  drec={drec.tolist()}  "
              f"{'PASS' if guard else 'FAIL'}{marker}")
        out["lambdas"].append({"lambda": lam, "oof": float(m), "delta": float(d),
                                "drec": drec.tolist(), "guard": guard, "emit": emit})
        if emit and (best_emit is None or d > best_emit["d"]):
            best_emit = {"lam": lam, "d": d}

    if best_emit is not None:
        lam = best_emit["lam"]
        alpha_row_t = np.clip(0.30 + lam * p_err_t_centered, 0.05, 0.70).astype(np.float32)
        blend_t = per_row_blend(s3_t, ms_t_iso, alpha_row_t)
        pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
        # diff vs PRIMARY
        p_v1_t = np.exp(lb(s3_t) * 0.7 + lb(ms_t_iso) * 0.3)
        p_v1_t = p_v1_t / p_v1_t.sum(1, keepdims=True)
        pred_v1 = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_test != pred_v1).sum())
        test_df = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test_df["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_test]})
        fname = f"submission_audit3_perrow_lam{int(lam*100):+03d}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"\n  test_diff_vs_PRIMARY={n_diff}")
        print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")
        out["best_emit"] = {"lambda": lam, "submission": fname}
    else:
        print("\n  No lambda passes both gates; no submission emitted.")

    out_path = ART / "audit3_per_row_calib_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    np.save(ART / "test_n5b_residual_auc.npy", p_err_t)
    print(f"\nSaved -> {out_path} + test_n5b_residual_auc.npy")


if __name__ == "__main__":
    main()
