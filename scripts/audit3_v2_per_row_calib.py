"""Audit-#3 v2: per-row weight calibration via AUC-0.63 head — RANK-NORMALIZED.

v1 NULL diagnosis: P_err mean=0.0149 std=0.0060 → mean-centered modifier
is bounded by ±0.02, even at λ=1.0 alpha_row range stays in [0.28, 0.32].
Too narrow to produce meaningful blend variation.

v2 fixes:
  A. Rank-normalize P_err to uniform [0, 1] (std-independent variation).
  B. Add HARD ROUTING variants: top-K rows override to meta_iso argmax.
  C. Test BOTH directions of lambda (maybe meta IS the wrong delivery on
     hard rows; primary should win there).
  D. Test the 3-class residual head approach (mode E): retrain a 3-class
     XGB on train rows where y != primary_argmax, override on top-K
     P_err rows with this head's argmax.

All variants use the FIXED LB-validated bias [1.43, 1.47, 3.40].
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
SEED = 42
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def macro(p, y, b=BIAS):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1))


def per_class_rec(p, y, b=BIAS):
    return recall_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1), average=None)


def per_row_blend(s3, ms, alpha_row):
    a = alpha_row[:, None]
    log_blended = (1 - a) * np.log(np.clip(s3, 1e-12, 1)) + a * np.log(np.clip(ms, 1e-12, 1))
    return log_blended  # return log-probs; bias added at scoring


def score(log_probs, y, b=BIAS):
    pred = (log_probs + b).argmax(1)
    m = balanced_accuracy_score(y, pred)
    r = recall_score(y, pred, average=None)
    return m, r, pred


def main() -> None:
    print("[1] Loading PRIMARY components + N5b features...")
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)

    p_err_o = np.load(ART / "oof_n5b_residual_auc.npy").astype(np.float32)
    # Already saved in v1: scripts/artifacts/test_n5b_residual_auc.npy
    p_err_t = np.load(ART / "test_n5b_residual_auc.npy").astype(np.float32)
    print(f"    OOF P_err: mean={p_err_o.mean():.4f} std={p_err_o.std():.4f}")

    # CORRECT baseline: log-blend at fixed alpha=0.30, BIAS once
    base_log = per_row_blend(s3_o, ms_o_iso, np.full(len(y), 0.30, dtype=np.float32))
    base_macro, base_rec, base_pred = score(base_log, y)
    base_errs = int((base_pred != y).sum())
    print(f"    baseline PRIMARY: OOF={base_macro:.5f}  rec={base_rec.round(5)}  errs={base_errs}")

    # Rank-normalize P_err to uniform [0, 1]
    p_err_rank_o = (rankdata(p_err_o) - 1) / (len(p_err_o) - 1)
    p_err_rank_t = (rankdata(p_err_t) - 1) / (len(p_err_t) - 1)
    p_err_rank_o = (p_err_rank_o - 0.5).astype(np.float32)  # center to [-0.5, +0.5]
    p_err_rank_t = (p_err_rank_t - 0.5).astype(np.float32)

    out = {"baseline_oof": float(base_macro),
            "baseline_rec": base_rec.tolist(),
            "p_err_oof_stats": {"mean": float(p_err_o.mean()),
                                "std": float(p_err_o.std())},
            "results": []}

    print("\n[2] MODE A: rank-normalized continuous lambda sweep")
    print(f"    alpha_row = clip(0.30 + lambda * rank(P_err), 0.05, 0.95)")
    for lam in [-0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40, 0.60]:
        alpha_o = np.clip(0.30 + lam * p_err_rank_o, 0.05, 0.95).astype(np.float32)
        log_o = per_row_blend(s3_o, ms_o_iso, alpha_o)
        m, r, _ = score(log_o, y)
        d = m - base_macro; drec = (r - base_rec).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"  λ={lam:+.2f}  α_row range=[{alpha_o.min():.3f}, {alpha_o.max():.3f}]  "
              f"OOF={m:.5f}  Δ={d:+.5f}  drec={drec.tolist()}  "
              f"{'PASS' if guard else 'FAIL'}{marker}")
        out["results"].append({"mode": "A_rank", "lambda": lam, "oof": float(m),
                                "delta": float(d), "drec": drec.tolist(),
                                "guard": guard, "emit": emit})

    print("\n[3] MODE B: hard-route top-K rows by P_err to alpha=0.7 or 1.0")
    for K_pct in [1, 2, 5, 10, 20]:
        for new_alpha in [0.5, 0.7, 1.0]:
            K = int(len(y) * K_pct / 100)
            top_idx = np.argsort(p_err_o)[-K:]
            alpha_o = np.full(len(y), 0.30, dtype=np.float32)
            alpha_o[top_idx] = new_alpha
            log_o = per_row_blend(s3_o, ms_o_iso, alpha_o)
            m, r, _ = score(log_o, y)
            d = m - base_macro; drec = (r - base_rec).round(6)
            guard = bool((drec >= -5e-4).all())
            emit = guard and d >= 3e-4
            marker = "  ← EMIT" if emit else ""
            print(f"  K={K_pct}% (n={K})  alpha_top={new_alpha}  "
                  f"OOF={m:.5f}  Δ={d:+.5f}  {'PASS' if guard else 'FAIL'}{marker}")
            out["results"].append({"mode": f"B_topK", "K_pct": K_pct,
                                    "alpha_top": new_alpha, "oof": float(m),
                                    "delta": float(d), "drec": drec.tolist(),
                                    "guard": guard, "emit": emit})

    print("\n[4] MODE C: hard-route top-K to PRIMARY-argmax = MORE primary (less meta)")
    for K_pct in [1, 2, 5, 10, 20]:
        K = int(len(y) * K_pct / 100)
        top_idx = np.argsort(p_err_o)[-K:]
        alpha_o = np.full(len(y), 0.30, dtype=np.float32)
        alpha_o[top_idx] = 0.0  # pure lb3
        log_o = per_row_blend(s3_o, ms_o_iso, alpha_o)
        m, r, _ = score(log_o, y)
        d = m - base_macro; drec = (r - base_rec).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"  K={K_pct}% (n={K})  alpha_top=0.0 (lb3-only)  "
              f"OOF={m:.5f}  Δ={d:+.5f}  {'PASS' if guard else 'FAIL'}{marker}")
        out["results"].append({"mode": f"C_topK_lb3only", "K_pct": K_pct,
                                "oof": float(m), "delta": float(d),
                                "drec": drec.tolist(), "guard": guard, "emit": emit})

    print("\n[5] MODE D: train a 3-class RESIDUAL CORRECTION head, override top-K")
    # Train 3-class XGB on rows where y != primary_argmax, target=y, features=N5b
    err_mask = base_pred != y
    n_err = err_mask.sum()
    print(f"    Training set: {n_err} primary-error rows")
    if n_err < 1000:
        print(f"    Too few errors; skipping mode D")
    else:
        ood = np.load(ART / "oof_ood3_train.npy").astype(np.float32)
        knn = np.load(ART / "oof_knn10k_train.npy").astype(np.float32)
        X_tr = np.concatenate([ood, knn], axis=1)
        ood_te = np.load(ART / "test_ood3.npy").astype(np.float32)
        knn_te = np.load(ART / "test_knn10k.npy").astype(np.float32)
        X_te = np.concatenate([ood_te, knn_te], axis=1)
        # 5-fold OOF for the residual-correction head, leak-safe
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        residual_oof = np.zeros((len(y), 3), dtype=np.float32)
        xgb_params = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            learning_rate=0.05, max_depth=4, min_child_weight=5,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=1.0, reg_lambda=1.0,
            tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
        )
        t0 = time.time()
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
            err_tr_idx = tr_idx[err_mask[tr_idx]]
            if len(err_tr_idx) < 100:
                continue
            dtr = xgb.DMatrix(X_tr[err_tr_idx], label=y[err_tr_idx])
            dva = xgb.DMatrix(X_tr[va_idx])
            booster = xgb.train(xgb_params, dtr, num_boost_round=200)
            residual_oof[va_idx] = booster.predict(dva)
        # Full-fit for test
        err_idx = np.where(err_mask)[0]
        dtr_full = xgb.DMatrix(X_tr[err_idx], label=y[err_idx])
        booster_full = xgb.train(xgb_params, dtr_full, num_boost_round=200)
        residual_test = booster_full.predict(xgb.DMatrix(X_te)).astype(np.float32)
        print(f"    residual head trained in {time.time()-t0:.1f}s")
        np.save(ART / "oof_n5b_residual_3class.npy", residual_oof)
        np.save(ART / "test_n5b_residual_3class.npy", residual_test)

        for K_pct in [1, 2, 5, 10]:
            K = int(len(y) * K_pct / 100)
            top_idx = np.argsort(p_err_o)[-K:]
            # Override top-K predictions with residual head argmax
            base_log_copy = base_log.copy()
            override_log = np.log(np.clip(residual_oof[top_idx], 1e-12, 1))
            base_log_copy[top_idx] = override_log
            m, r, _ = score(base_log_copy, y)
            d = m - base_macro; drec = (r - base_rec).round(6)
            guard = bool((drec >= -5e-4).all())
            emit = guard and d >= 3e-4
            marker = "  ← EMIT" if emit else ""
            print(f"  K={K_pct}% (n={K})  override w/ residual_3class  "
                  f"OOF={m:.5f}  Δ={d:+.5f}  {'PASS' if guard else 'FAIL'}{marker}")
            out["results"].append({"mode": f"D_topK_residual3class", "K_pct": K_pct,
                                    "oof": float(m), "delta": float(d),
                                    "drec": drec.tolist(), "guard": guard, "emit": emit})

    # Find best emit candidate
    emits = [r for r in out["results"] if r.get("emit")]
    print(f"\n[6] {len(emits)} emit candidates")
    if emits:
        best = max(emits, key=lambda r: r["delta"])
        print(f"  best: {best['mode']} Δ={best['delta']:+.5f}")

    out_path = ART / "audit3_v2_per_row_calib_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
