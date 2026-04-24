"""Option 3: disagreement-feature meta-stacker.

Train a shallow XGB on features derived from teacher-vs-candidate disagreement
MAGNITUDES (not raw probs), plus a few cell/score features. Target = y.

Rationale: prior LR / greedy meta-stackers on per-class probs have been null
because inputs are ~collinear with log-bias tune. Disagreement features encode
"which candidate disagrees with teacher, by how much, on which class" — a
signal no α-blend weight can represent.

Features per row (no raw probs):
    diff_cand_c  = P_teacher(c) - P_cand(c)        (3 per candidate; 7*3=21)
    teacher_conf = max_c P_teacher(c)              (1)
    teacher_entropy                                 (1)
    teacher_argmax                                  (1 int)
    disagree_mask_c = (teacher_argmax != cand_argmax)  (1 per candidate; 7)
    dgp_score                                       (1 int)
    signed distances sm_dist, rf_dist, tc_dist, ws_dist  (4)

Total ~35 features. 5-fold StratifiedKFold(seed=42) aligned with base OOFs.

Gate (no LB spend):
    standalone OOF > teacher (0.98029) by >+0.00020  AND
    blend vs teacher (fixed recipe bias) passes     OR
    standalone LB-forecast via gap-ladder > 0.98005
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from common import fast_bal_acc, tune_log_bias
from meta_common import (
    ART, CANDIDATES, EPS, argmax_int8, build_teacher, entropy, get_folds,
    load_candidates, load_y_and_features, log_blend, recipe_bias,
)

OUT_OOF = ART / "oof_disagree_meta.npy"
OUT_TEST = ART / "test_disagree_meta.npy"
OUT_JSON = ART / "disagree_meta_results.json"


def build_features(teacher: np.ndarray, cand_probs: dict) -> np.ndarray:
    """Stack per-row disagreement features (teacher shape (N,3))."""
    feats = []
    t_conf = teacher.max(axis=1, keepdims=True).astype(np.float32)
    t_ent = entropy(teacher).reshape(-1, 1)
    t_arg = argmax_int8(teacher).reshape(-1, 1).astype(np.float32)

    feats.append(t_conf)
    feats.append(t_ent)
    feats.append(t_arg)

    for name in CANDIDATES:
        p = cand_probs[name]
        diff = (teacher - p).astype(np.float32)          # (N, 3)
        cand_arg = argmax_int8(p).reshape(-1, 1).astype(np.float32)
        disagree = (t_arg != cand_arg).astype(np.float32)
        feats.extend([diff, disagree])

    return np.concatenate(feats, axis=1).astype(np.float32)


def run(smoke: bool = False) -> dict:
    t0 = time.time()
    print("[meta] loading y + dist features ...")
    y, tr_score, tr_dist, te_score, te_dist = load_y_and_features()
    print("[meta] building teacher + candidate OOFs ...")
    oof_teacher, test_teacher = build_teacher()
    cand_oof, cand_test = load_candidates()

    print("[meta] stacking features ...")
    X_oof = np.concatenate([
        build_features(oof_teacher, cand_oof),
        tr_score.reshape(-1, 1).astype(np.float32),
        tr_dist,
    ], axis=1)
    X_test = np.concatenate([
        build_features(test_teacher, cand_test),
        te_score.reshape(-1, 1).astype(np.float32),
        te_dist,
    ], axis=1)
    print(f"[meta] feature matrix oof={X_oof.shape} test={X_test.shape}")

    folds = get_folds(y)
    oof_pred = np.zeros((len(y), 3), dtype=np.float32)
    test_preds = []

    xgb_params = dict(
        objective="multi:softprob", num_class=3, tree_method="hist",
        max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=2.0, min_child_weight=20,
        eval_metric="mlogloss", verbosity=0,
    )
    if smoke:
        xgb_params["learning_rate"] = 0.2
        n_rounds = 80
        folds = folds[:1]
    else:
        n_rounds = 1500
    early_stop = 80

    for i, (tr_idx, va_idx) in enumerate(folds):
        print(f"[meta] fold {i+1}/{len(folds)} ...")
        dtr = xgb.DMatrix(X_oof[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_oof[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_test)
        bst = xgb.train(xgb_params, dtr, num_boost_round=n_rounds,
                        evals=[(dva, "va")],
                        early_stopping_rounds=early_stop, verbose_eval=False)
        best_it = bst.best_iteration + 1
        pv = bst.predict(dva, iteration_range=(0, best_it))
        pt = bst.predict(dte, iteration_range=(0, best_it))
        oof_pred[va_idx] = pv
        test_preds.append(pt)
        print(f"[meta]   best_iter={best_it}  fold argmax bal_acc="
              f"{fast_bal_acc(y[va_idx], pv.argmax(1)):.5f}")

    test_pred = np.mean(test_preds, axis=0).astype(np.float32) if test_preds \
        else np.zeros((len(te_score), 3), dtype=np.float32)

    # Teacher baseline at recipe fixed bias
    bias = recipe_bias()
    prior = np.bincount(y, minlength=3) / len(y)

    teacher_fixed = fast_bal_acc(y, (np.log(np.clip(oof_teacher, EPS, 1.0)) + bias).argmax(1))
    meta_fixed = fast_bal_acc(y, (np.log(np.clip(oof_pred, EPS, 1.0)) + bias).argmax(1))
    tuned_b, meta_tuned = tune_log_bias(oof_pred, y, prior, high_grid_wide=True)
    print(f"[meta] teacher @ recipe bias = {teacher_fixed:.5f}")
    print(f"[meta] meta    @ recipe bias = {meta_fixed:.5f}   "
          f"(Δ={meta_fixed - teacher_fixed:+.5f})")
    print(f"[meta] meta tuned = {meta_tuned:.5f}   bias={tuned_b.tolist()}")

    # Fixed-bias blend sweep teacher × meta (NO bias retune)
    sweep = {}
    for a in np.linspace(0.0, 0.6, 13):
        b = log_blend([oof_teacher, oof_pred], [1 - a, a])
        ba = fast_bal_acc(y, (np.log(np.clip(b, EPS, 1.0)) + bias).argmax(1))
        sweep[f"{a:.3f}"] = float(ba)
    peak_a = max(sweep, key=sweep.get)
    peak_v = sweep[peak_a]
    print(f"[meta] blend sweep peak α={peak_a} → {peak_v:.5f}  "
          f"Δ={peak_v - teacher_fixed:+.5f}")

    if not smoke:
        np.save(OUT_OOF, oof_pred)
        np.save(OUT_TEST, test_pred)
    summary = dict(
        n_features=int(X_oof.shape[1]),
        teacher_fixed_bias_ba=float(teacher_fixed),
        meta_fixed_bias_ba=float(meta_fixed),
        meta_tuned_ba=float(meta_tuned),
        meta_tuned_bias=tuned_b.tolist(),
        blend_sweep=sweep,
        peak_alpha=peak_a,
        peak_blend_ba=float(peak_v),
        delta_vs_teacher=float(peak_v - teacher_fixed),
        smoke=bool(smoke),
        wall_seconds=round(time.time() - t0, 1),
    )
    if not smoke:
        OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[meta] done in {summary['wall_seconds']}s")
    return summary


if __name__ == "__main__":
    import os
    smoke = os.environ.get("SMOKE", "0") == "1"
    run(smoke=smoke)
