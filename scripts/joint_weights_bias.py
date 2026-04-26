"""Joint optimization of blend weights + per-class bias (#4 from menu).

Why this is genuinely new on this competition:
  - Every prior tuning held BIAS fixed at [1.4324, 1.4689, 3.4008] to avoid
    the binhigh-trap (post-hoc bias retune inflated OOF +0.00084 → LB -0.00084).
  - This script jointly optimises blend weights AND bias with strong L2 toward
    the LB-best operating point, so the bias can NUDGE toward the simplex
    weights' optimum but cannot drift far from the LB-validated bias.
  - Single-stage optimization avoids the compounding overfit of "tune weights,
    then retune bias" (which caused the binhigh +0.00084 → -0.00084 swap).

Surrogate: smoothed macro-recall via temperature-relaxed argmax. T=0.3 keeps
the surrogate close to true argmax while remaining differentiable.

Honest 5-fold CV: optimize on tr_idx, score on va_idx. Test gets full-OOF-fit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                            load_y, normed)


ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
EPS = 1e-12
T = 0.3   # softmax temperature for the relaxed-argmax surrogate
L2_BIAS = 5.0   # strong pull toward LB-validated bias
L2_W = 1e-3     # mild weight regularization


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def softmax_w(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def soft_macro_recall_loss(theta, oofs_arr, y, K, T=T,
                           l2_bias=L2_BIAS, l2_w=L2_W):
    """Negated soft macro-recall + L2 toward LB-validated bias.

    P[i,c] = exp(sum_k w_k log E_kic) / Σ ... at row i (log-blend).
    z[i,c] = log P[i,c] + bias[c]
    p_soft[i,c] = exp(z[i,c]/T) / Σ_c'  (smooth argmax, T → 0 is hard argmax)
    macro_recall_soft = mean_c (sum_{i:y_i=c} p_soft[i,c]) / N_c
    objective = -macro_recall_soft + 0.5*l2_bias*||bias - BIAS||² + 0.5*l2_w*||z_w||²
    """
    z_w = theta[:K]
    bias = theta[K:K + 3]
    w = softmax_w(z_w)
    # log-blend
    log_oofs = np.log(np.clip(oofs_arr, EPS, 1.0))
    logits = (w[:, None, None] * log_oofs).sum(0)   # (N, 3)
    z = logits + bias[None, :]
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    p_soft = e / e.sum(1, keepdims=True)            # (N, 3)
    # Per-class soft recall
    cc = np.bincount(y, minlength=3).astype(np.float32)
    soft_recall = np.zeros(3)
    for c in range(3):
        mask = (y == c)
        if mask.sum() > 0:
            soft_recall[c] = p_soft[mask, c].sum() / cc[c]
    obj = -soft_recall.mean()
    # Regularization
    obj += 0.5 * l2_bias * float(((bias - BIAS) ** 2).sum())
    obj += 0.5 * l2_w * float((z_w ** 2).sum())
    return obj


def fit_one_fold(oofs_tr, y_tr, K, sigma0=0.1, maxiter=100):
    theta0 = np.concatenate([np.zeros(K), BIAS.copy()])
    oofs_arr = np.stack(oofs_tr, axis=0)
    res = minimize(
        soft_macro_recall_loss, theta0,
        args=(oofs_arr, y_tr.astype(np.int32), K),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-6},
    )
    return res.x


def true_bal_acc(P, y, bias):
    pred = (np.log(np.clip(P, EPS, 1.0)) + bias).argmax(1)
    return balanced_accuracy_score(y, pred)


def main():
    log("Loading components")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    realmlp_o, realmlp_t = L("realmlp")
    nr_raw_o, nr_raw_t = L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr_raw_o, nr_raw_t, y)
    leaf_o, leaf_t = L("leaf_ote_meta_v2")
    dig_o, dig_t = L("xgb_dist_digits")

    components = [
        ("lb_best_3stack", lb3_o, lb3_t),
        ("xgb_metastack_iso", meta_iso_o, meta_iso_t),
        ("realmlp", realmlp_o, realmlp_t),
        ("xgb_nonrule_iso", nr_o, nr_t),
        ("leaf_ote_meta_v2", leaf_o, leaf_t),
        ("xgb_dist_digits", dig_o, dig_t),
    ]
    names = [c[0] for c in components]
    K = len(components)
    oofs = [c[1] for c in components]
    tests = [c[2] for c in components]
    log(f"K={K}: {names}")

    # In-sample full fit
    log("Phase 1: in-sample joint full-fit")
    theta_full = fit_one_fold(oofs, y, K, maxiter=200)
    w_full = softmax_w(theta_full[:K])
    bias_full = theta_full[K:K + 3]
    P_full = log_blend(oofs, w_full)
    bal_full = true_bal_acc(P_full, y, bias_full)
    log(f"  In-sample bal_acc={bal_full:.6f}")
    log(f"  Weights: {dict(zip(names, [round(float(x), 4) for x in w_full]))}")
    log(f"  Bias: {bias_full.round(4).tolist()} (vs LB {BIAS.tolist()})")

    # Honest CV
    log("Phase 2: nested 5-fold CV")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.zeros((len(y), 3), dtype=np.float32)
    bias_records = []
    test_acc = np.zeros(tests[0].shape, dtype=np.float32)
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        oofs_tr = [o[tr] for o in oofs]
        theta = fit_one_fold(oofs_tr, y[tr], K, maxiter=100)
        w = softmax_w(theta[:K])
        bias = theta[K:K + 3]
        # Predict on val
        oofs_va = [o[va] for o in oofs]
        Pv = log_blend(oofs_va, w)
        oof_blend[va] = Pv
        # Test
        Pte = log_blend(tests, w)
        test_acc += Pte / N_FOLDS
        bias_records.append({
            "fold": fi + 1,
            "weights": [float(x) for x in w],
            "bias": [float(x) for x in bias],
            "wall_s": round(time.time() - t0, 2),
        })
        log(f"  fold {fi+1}: w={[round(float(x),3) for x in w]} "
            f"bias={[round(float(x),3) for x in bias]} wall={time.time()-t0:.1f}s")

    # CV scoring uses the per-fold tuned bias for that fold's val rows
    # (honest: we don't peek at the held-out rows when fitting bias)
    cv_pred = np.zeros(len(y), dtype=np.int8)
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        bias = np.array(bias_records[fi]["bias"])
        cv_pred[va] = (np.log(np.clip(oof_blend[va], EPS, 1.0)) + bias).argmax(1)
    bal_cv = balanced_accuracy_score(y, cv_pred)
    log(f"Nested CV bal_acc (per-fold tuned bias) = {bal_cv:.6f}")

    # Also score at FIXED LB bias for direct comparability
    bal_cv_fixed = true_bal_acc(oof_blend, y, BIAS)
    log(f"Nested CV bal_acc (FIXED LB bias)      = {bal_cv_fixed:.6f}")

    # Save artifacts
    np.save(ART / "oof_joint_blend.npy", oof_blend)
    np.save(ART / "test_joint_blend.npy", test_acc)
    out = {
        "components": names,
        "K": K,
        "in_sample_bal_acc": bal_full,
        "in_sample_weights": dict(zip(names, [float(x) for x in w_full])),
        "in_sample_bias": [float(x) for x in bias_full],
        "cv_bal_acc_per_fold_bias": bal_cv,
        "cv_bal_acc_fixed_lb_bias": bal_cv_fixed,
        "fold_records": bias_records,
        "T": T,
        "L2_BIAS": L2_BIAS,
        "L2_W": L2_W,
    }
    (ART / "joint_weights_bias_results.json").write_text(json.dumps(out, indent=2))
    log("Saved oof_joint_blend.npy + test + results JSON")


if __name__ == "__main__":
    main()
