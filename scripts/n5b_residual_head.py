"""N5b option #3: residual head — different delivery mechanism than bank-add.

Train a 3-class XGB on (PRIMARY OOF probs as logits + 11 N5b features
+ optional 9 expanded features) → predict y. The residual head sees
PRIMARY's posterior PLUS the 10k-anchor signal, and learns to correct
PRIMARY's specific errors (NOT through meta-stacker bank-add).

Then blend: final = log_blend([primary, residual], [w, 1-w]) for small w.

Different from bank-add (which retrains the entire xgb_metastack on a
60+ component bank). Different from direct recipe FE (which puts N5b
features in the recipe matrix). This trains a SMALL second-stage model
that consumes primary's output + anchor features.

Hypothesis: if the meta-stacker's −2.5x OOF→LB carryover is an
architectural property of the saturated bank (not the signal), a
fresh small XGB on (primary_probs + 11 anchor feats) may transfer
differently because it isn't part of the meta-stacker bank.

Wall: ~10 min CPU.
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


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def macro(p, y, b=BIAS):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1))


def main() -> None:
    print("[1] Loading components...")
    y = load_y()

    # Build PRIMARY OOF + test
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)
    p_primary_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    p_primary_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))

    # 11 N5b features
    ood3_o = np.load(ART / "oof_ood3_train.npy").astype(np.float32)
    ood3_t = np.load(ART / "test_ood3.npy").astype(np.float32)
    knn10_o = np.load(ART / "oof_knn10k_train.npy").astype(np.float32)
    knn10_t = np.load(ART / "test_knn10k.npy").astype(np.float32)

    # 9 expanded features (if available)
    use_expanded = (ART / "oof_ood9_train.npy").exists()
    if use_expanded:
        ood9_o = np.load(ART / "oof_ood9_train.npy").astype(np.float32)
        ood9_t = np.load(ART / "test_ood9.npy").astype(np.float32)
        print(f"    expanded 9 features available -> 20 total N5b features")
    else:
        ood9_o = np.zeros((len(y), 0), dtype=np.float32)
        ood9_t = np.zeros((len(p_primary_t), 0), dtype=np.float32)
        print(f"    expanded features NOT yet available -> 11 N5b features only")

    # Feature matrix: log(primary_probs)[3] + 11 (or 20) N5b features
    p_log_o = np.log(np.clip(p_primary_o, 1e-12, 1))
    p_log_t = np.log(np.clip(p_primary_t, 1e-12, 1))
    X_tr = np.concatenate([p_log_o, ood3_o, knn10_o, ood9_o], axis=1).astype(np.float32)
    X_te = np.concatenate([p_log_t, ood3_t, knn10_t, ood9_t], axis=1).astype(np.float32)
    print(f"    X_tr shape={X_tr.shape}, y shape={y.shape}")

    base_macro = macro(p_primary_o, y)
    base_rec = recall_score(y, (np.log(np.clip(p_primary_o, 1e-12, 1)) + BIAS).argmax(1), average=None)
    print(f"    PRIMARY OOF macro={base_macro:.5f}  rec={base_rec.round(5)}")

    print("\n[2] 5-fold residual XGB head...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds = []
    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=2.0, reg_lambda=2.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X_tr[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_te)
        booster = xgb.train(xgb_params, dtr, num_boost_round=2000,
                             evals=[(dva, "val")], early_stopping_rounds=100,
                             verbose_eval=0)
        bi = booster.best_iteration
        vp = booster.predict(dva, iteration_range=(0, bi + 1))
        oof[va_idx] = vp
        tp = booster.predict(dte, iteration_range=(0, bi + 1))
        test_folds.append(tp)
        m_va = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        print(f"  fold {fold+1}/5 best_it={bi} val_argmax={m_va:.5f} wall={time.time()-t0:.1f}s")

    test_pred = np.mean(test_folds, axis=0).astype(np.float32)
    np.save(ART / "oof_n5b_residual_head.npy", oof)
    np.save(ART / "test_n5b_residual_head.npy", test_pred)

    # Standalone tuned macro
    print("\n[3] Residual head standalone:")
    head_macro = macro(oof, y)
    head_rec = recall_score(y, (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1), average=None)
    head_errs = int(((np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1) != y).sum())
    primary_errs = int(((np.log(np.clip(p_primary_o, 1e-12, 1)) + BIAS).argmax(1) != y).sum())
    print(f"  macro@bias={head_macro:.5f}  rec={head_rec.round(5)}  errs={head_errs}")
    print(f"  primary errs={primary_errs}, head errs={head_errs}, diff={head_errs - primary_errs}")

    # Jaccard residual vs primary
    head_pred = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    pri_pred = (np.log(np.clip(p_primary_o, 1e-12, 1)) + BIAS).argmax(1)
    inter = int(((head_pred != y) & (pri_pred != y)).sum())
    union = int(((head_pred != y) | (pri_pred != y)).sum())
    jacc = inter / max(1, union)
    print(f"  Jaccard(head_errs, primary_errs) = {jacc:.4f}")

    print("\n[4] Blend gate vs PRIMARY (fixed BIAS, alpha sweep)...")
    rows = []
    best = None
    for a in [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        blend = log_blend([p_primary_o, oof], np.array([1 - a, a]))
        m = macro(blend, y)
        r = recall_score(y, (np.log(np.clip(blend, 1e-12, 1)) + BIAS).argmax(1), average=None)
        d = m - base_macro
        drec = (r - base_rec).round(6)
        guard = bool((drec >= -5e-4).all())
        row = {"alpha": a, "oof": float(m), "d": float(d),
               "drec": drec.tolist(), "guard": guard}
        rows.append(row)
        marker = " <- best" if guard and (best is None or d > best["d"]) else ""
        if guard and (best is None or d > best["d"]):
            best = row
        print(f"  a={a:.3f}  OOF={m:.5f}  d={d:+.5f}  drec={drec.tolist()}  "
              f"{'PASS' if guard else 'FAIL'}{marker}")

    print("\n[5] Best under guardrail:")
    if best:
        print(f"  alpha={best['alpha']}  d={best['d']:+.5f}")
        if best["d"] >= 2e-4:
            # Build & save submission
            blend_t = log_blend([p_primary_t, test_pred], np.array([1 - best["alpha"], best["alpha"]]))
            pred_t = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
            primary_pred_t = (np.log(np.clip(p_primary_t, 1e-12, 1)) + BIAS).argmax(1)
            n_diff = int((pred_t != primary_pred_t).sum())
            test_df = pd.read_csv("data/test.csv")
            sub = pd.DataFrame({"id": test_df["id"].values,
                                 "Irrigation_Need": [LABELS[i] for i in pred_t]})
            tag = f"a{int(best['alpha']*1000):03d}"
            n_feats = X_tr.shape[1]
            fname = f"submission_n5b_residual_head_f{n_feats}_{tag}.csv"
            sub.to_csv(SUB / fname, index=False)
            print(f"  test diff vs PRIMARY: {n_diff}")
            print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")

    out = {
        "n_features": int(X_tr.shape[1]),
        "use_expanded": bool(use_expanded),
        "primary_oof": float(base_macro),
        "head_oof": float(head_macro),
        "head_errs": int(head_errs),
        "primary_errs": int(primary_errs),
        "jaccard": float(jacc),
        "rows": rows,
        "best": best,
    }
    out_path = ART / "n5b_residual_head_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
