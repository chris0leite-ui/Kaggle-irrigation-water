"""Option 1: per-row selective routing (not global log-blend).

For each row:
    if teacher_conf >= τ: predict teacher argmax  (keep LB-best)
    else:                 predict router-chosen argmax

The router is a shallow XGB classifier trained ONLY on low-conf rows to
predict y, using argmax/conf features from teacher and every candidate.
Rationale: global α-blend averages probs on confident-correct rows —
wastes orthogonality and risks the "magnitude trap" (candidate's extra
errors leak into the blend).

5-fold StratifiedKFold(seed=42) aligned with base OOFs so router OOF
predictions can be log-blended / gated.

Sweep:
    τ ∈ {0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99}
Report teacher_argmax OOF vs router OOF at each τ.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from common import fast_bal_acc, tune_log_bias
from meta_common import (
    ART, CANDIDATES, EPS, argmax_int8, build_teacher, get_folds,
    load_candidates, load_y_and_features, recipe_bias,
)

OUT_OOF = ART / "oof_selective_router.npy"
OUT_TEST = ART / "test_selective_router.npy"
OUT_JSON = ART / "selective_router_results.json"

TAUS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]


def build_router_features(teacher: np.ndarray, cand_probs: dict) -> np.ndarray:
    """Per-row features for the router: argmaxes + confidences + disagreements + score/dist."""
    feats = []
    t_conf = teacher.max(axis=1, keepdims=True).astype(np.float32)
    t_arg = argmax_int8(teacher).reshape(-1, 1).astype(np.float32)
    feats.extend([t_conf, t_arg])
    for name in CANDIDATES:
        p = cand_probs[name]
        c_conf = p.max(axis=1, keepdims=True).astype(np.float32)
        c_arg = argmax_int8(p).reshape(-1, 1).astype(np.float32)
        disagree = (c_arg != t_arg).astype(np.float32)
        feats.extend([c_conf, c_arg, disagree])
    return np.concatenate(feats, axis=1).astype(np.float32)


def run(smoke: bool = False) -> dict:
    t0 = time.time()
    print("[router] loading y + dist features ...")
    y, tr_score, tr_dist, te_score, te_dist = load_y_and_features()
    print("[router] building teacher + candidate bank ...")
    oof_teacher, test_teacher = build_teacher()
    cand_oof, cand_test = load_candidates()

    # Router features (same on train + test)
    X_oof = np.concatenate([
        build_router_features(oof_teacher, cand_oof),
        tr_score.reshape(-1, 1).astype(np.float32),
        tr_dist,
    ], axis=1)
    X_test = np.concatenate([
        build_router_features(test_teacher, cand_test),
        te_score.reshape(-1, 1).astype(np.float32),
        te_dist,
    ], axis=1)
    print(f"[router] X_oof={X_oof.shape} X_test={X_test.shape}")

    folds = get_folds(y)
    if smoke:
        folds = folds[:1]

    # Train router on ALL rows (the τ gate is applied at inference time).
    # This way router OOF is defined for every row, making the sweep apples-to-apples.
    oof_router = np.zeros((len(y), 3), dtype=np.float32)
    test_preds = []
    xgb_params = dict(
        objective="multi:softprob", num_class=3, tree_method="hist",
        max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, min_child_weight=10,
        eval_metric="mlogloss", verbosity=0,
    )
    n_rounds = 80 if smoke else 1500

    for i, (tr_idx, va_idx) in enumerate(folds):
        print(f"[router] fold {i+1}/{len(folds)} ...")
        dtr = xgb.DMatrix(X_oof[tr_idx], label=y[tr_idx])
        dva = xgb.DMatrix(X_oof[va_idx], label=y[va_idx])
        dte = xgb.DMatrix(X_test)
        bst = xgb.train(xgb_params, dtr, num_boost_round=n_rounds,
                        evals=[(dva, "va")],
                        early_stopping_rounds=80, verbose_eval=False)
        best_it = bst.best_iteration + 1
        oof_router[va_idx] = bst.predict(dva, iteration_range=(0, best_it))
        test_preds.append(bst.predict(dte, iteration_range=(0, best_it)))
        print(f"[router]   best_iter={best_it}")

    test_router = (np.mean(test_preds, axis=0).astype(np.float32)
                   if test_preds else np.zeros((len(te_score), 3), dtype=np.float32))

    # Teacher baseline at recipe bias
    bias = recipe_bias()
    log_t = np.log(np.clip(oof_teacher, EPS, 1.0))
    t_argmax_biased = (log_t + bias).argmax(axis=1)
    t_conf = oof_teacher.max(axis=1)
    teacher_ba = fast_bal_acc(y, t_argmax_biased)
    print(f"[router] teacher @ recipe bias = {teacher_ba:.5f}")

    # Router-alone (standalone OOF @ its own tuned bias, diagnostic)
    prior = np.bincount(y, minlength=3) / len(y)
    r_tuned_bias, r_tuned_ba = tune_log_bias(oof_router, y, prior, high_grid_wide=True)
    print(f"[router] router tuned standalone = {r_tuned_ba:.5f} bias={r_tuned_bias.tolist()}")
    # IMPORTANT: use tuned-bias argmax so router lives on the same macro-recall
    # operating point as teacher. Raw argmax collapses to majority class.
    r_argmax = (np.log(np.clip(oof_router, EPS, 1.0)) + r_tuned_bias).argmax(axis=1)

    # τ sweep: route only rows where teacher_conf < τ
    sweep = {}
    for tau in TAUS:
        use_router = t_conf < tau
        pred = np.where(use_router, r_argmax, t_argmax_biased)
        ba = fast_bal_acc(y, pred)
        n_routed = int(use_router.sum())
        routed_correct = int(((r_argmax == y) & use_router).sum())
        would_be_correct = int(((t_argmax_biased == y) & use_router).sum())
        sweep[f"tau={tau:.2f}"] = dict(
            bal_acc=float(ba),
            n_routed=n_routed,
            frac_routed=round(n_routed / len(y), 4),
            router_right_on_routed=routed_correct,
            teacher_would_right_on_routed=would_be_correct,
            net_wins=routed_correct - would_be_correct,
        )
        print(f"[router] τ={tau:.2f}  routed={n_routed:>7d} ({100*n_routed/len(y):5.2f}%)  "
              f"bal_acc={ba:.5f}  Δ={ba - teacher_ba:+.5f}  "
              f"net_wins={routed_correct - would_be_correct:+d}")

    peak_tau = max(sweep, key=lambda k: sweep[k]["bal_acc"])
    peak_val = sweep[peak_tau]["bal_acc"]

    # Also: class-specific confidence gate — route only where teacher
    # predicts rare class (High) with low confidence. That's where blend
    # history shows the biggest correction potential.
    class_gate = {}
    for tau in (0.85, 0.90, 0.95):
        use_router = (t_argmax_biased == 2) & (t_conf < tau)  # High, low-conf
        pred = np.where(use_router, r_argmax, t_argmax_biased)
        ba = fast_bal_acc(y, pred)
        n = int(use_router.sum())
        class_gate[f"high_only_tau={tau:.2f}"] = dict(
            bal_acc=float(ba),
            n_routed=n,
            delta=float(ba - teacher_ba),
        )

    if not smoke:
        np.save(OUT_OOF, oof_router)
        np.save(OUT_TEST, test_router)

    summary = dict(
        n_features=int(X_oof.shape[1]),
        teacher_ba=float(teacher_ba),
        router_standalone_tuned_ba=float(r_tuned_ba),
        router_standalone_tuned_bias=r_tuned_bias.tolist(),
        tau_sweep=sweep,
        peak_tau=peak_tau,
        peak_bal_acc=float(peak_val),
        delta_vs_teacher=float(peak_val - teacher_ba),
        high_only_gate=class_gate,
        smoke=bool(smoke),
        wall_seconds=round(time.time() - t0, 1),
    )
    if not smoke:
        OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[router] done in {summary['wall_seconds']}s  peak={peak_val:.5f} at {peak_tau}")
    return summary


if __name__ == "__main__":
    import os
    smoke = os.environ.get("SMOKE", "0") == "1"
    run(smoke=smoke)
