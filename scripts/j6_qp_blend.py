"""J6: constraint-aware QP for blend weights.

Greedy + LR are myopic local-optimizers on the simplex; a global
log-blend solver might surface a config greedy missed.

Convex objective: macro-balanced cross-entropy on log-blend probs
  L(w) = (1/3) Σ_k (1/N_k) Σ_{i:y_i=k} (logsumexp(z_i) − z_{i,k})
  where z_i = Σ_c w_c · log p_c[i] + bias

Constraints: w ≥ 0, Σw = 1 (simplex).

Per-class recall is non-convex; we evaluate it post-hoc on the QP
solution and emit only if the LB-best 4-stack guardrail holds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tier1b_helpers import (
    ART, BIAS, build_lbbest_stack, iso_cal, load_y, normed,
)

# Candidate pool: strong + diverse, all proven (some LB-validated as
# components; iso variants normalized in-place).
POOL = [
    "recipe_full_te",
    "recipe_pseudolabel",                # pseudo_s1 (in LB-best)
    "recipe_pseudolabel_seed7labeler",   # pseudo_s7 (in LB-best)
    "realmlp",                           # in LB-best 3-stack (α=0.20)
    "xgb_nonrule",                       # iso form is in LB-best 4-stack
    "xgb_metastack",                     # iso form is meta-stacker leg
    "xgb_metastack_v3",                  # cross-poll meta-stacker
    "lgbm_te_orig",
    "xgb_corn",
    "xgb_dist_digits",
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return recall_score(y_true, y_pred, labels=[0, 1, 2], average=None)


def main() -> None:
    log("loading y + LB-best 4-stack anchor")
    y = load_y()

    lb3o, lb3t = build_lbbest_stack(y)
    # LB-best 4-stack = LB-best 3-stack + xgb_metastack_iso (α=0.30)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_oi, meta_ti = iso_cal(meta_o, meta_t, y)
    log_lb4_o = 0.70 * np.log(np.clip(lb3o, 1e-12, 1)) + 0.30 * np.log(np.clip(meta_oi, 1e-12, 1))
    lb4_o = normed(np.exp(log_lb4_o - log_lb4_o.max(1, keepdims=True)))
    lb4_pred = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    lb4_bal = balanced_accuracy_score(y, lb4_pred)
    lb4_pcr = per_class_recall(y, lb4_pred)
    log(f"LB-best 4-stack OOF bal={lb4_bal:.5f}  per-class recall={lb4_pcr}")

    log("loading candidate pool (iso-cal'd where indicated)")
    candidates: list[tuple[str, np.ndarray]] = []
    candidates_test: list[np.ndarray] = []
    for name in POOL:
        o = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        t = normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        if name in ("xgb_nonrule", "xgb_metastack", "xgb_metastack_v3", "lgbm_te_orig"):
            o, t = iso_cal(o, t, y)
        candidates.append((name, o))
        candidates_test.append(t)
        std_pred = (np.log(np.clip(o, 1e-12, 1)) + BIAS).argmax(1)
        std_bal = balanced_accuracy_score(y, std_pred)
        log(f"  {name:<35s} standalone bal={std_bal:.5f}")

    C = len(candidates)
    N = len(y)
    logp = np.stack([np.log(np.clip(o, 1e-12, 1)) for _, o in candidates])  # (C, N, 3)
    logp_t = np.stack([np.log(np.clip(t, 1e-12, 1)) for t in candidates_test])  # (C, M, 3)

    # Per-class row counts for balanced CE
    counts = np.bincount(y, minlength=3).astype(np.float64)
    log(f"class counts: {counts.tolist()}")

    def objective_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = (w[:, None, None] * logp).sum(0) + BIAS  # (N, 3)
        zmax = z.max(1, keepdims=True)
        Z = np.exp(z - zmax)
        Z_sum = Z.sum(1, keepdims=True)
        log_softmax = z - (zmax + np.log(Z_sum))      # (N, 3)
        nll_per_row = -log_softmax[np.arange(N), y]   # (N,)
        # Class-balanced: weight each row by 1/(3 * N_class)
        row_w = 1.0 / (3.0 * counts[y])
        loss = float((row_w * nll_per_row).sum())
        # Gradient: dL/dw_c = sum_i row_w_i * (softmax_i - onehot_i) · logp_c[i]
        sm = np.exp(log_softmax)                      # (N, 3)
        sm_minus_y = sm.copy()
        sm_minus_y[np.arange(N), y] -= 1.0
        # Weighted by row_w
        weighted = sm_minus_y * row_w[:, None]        # (N, 3)
        grad = (logp * weighted[None, :, :]).sum((1, 2))  # (C,)
        return loss, grad

    log(f"solving QP: minimize macro-balanced NLL on {C}-simplex")
    t0 = time.time()
    w0 = np.full(C, 1.0 / C)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * C
    res = minimize(
        lambda w: objective_and_grad(w),
        w0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    log(f"QP solved in {time.time()-t0:.1f}s, success={res.success}, nit={res.nit}")
    w_qp = res.x.copy()
    w_qp[w_qp < 1e-4] = 0.0
    w_qp = w_qp / w_qp.sum()
    log(f"QP weights:")
    for (name, _), wi in sorted(zip(candidates, w_qp), key=lambda x: -x[1]):
        if wi > 1e-4:
            log(f"    {wi:.4f}  {name}")

    # Evaluate on OOF
    z_qp = (w_qp[:, None, None] * logp).sum(0)
    blend_p = normed(np.exp(z_qp - z_qp.max(1, keepdims=True)))
    pred_qp = (np.log(np.clip(blend_p, 1e-12, 1)) + BIAS).argmax(1)
    qp_bal = balanced_accuracy_score(y, pred_qp)
    qp_pcr = per_class_recall(y, pred_qp)
    log(f"QP OOF bal={qp_bal:.5f}  per-class recall={qp_pcr}")

    delta = qp_bal - lb4_bal
    pcr_delta = qp_pcr - lb4_pcr
    guardrail_ok = (pcr_delta >= -5e-4).all()
    log(f"Δ vs LB-best 4-stack = {delta:+.5f}")
    log(f"per-class Δ          = {pcr_delta}")
    log(f"guardrail (Δ ≥ -5e-4 each class): {guardrail_ok}")

    # Test-side blend
    z_qp_t = (w_qp[:, None, None] * logp_t).sum(0)
    blend_t = normed(np.exp(z_qp_t - z_qp_t.max(1, keepdims=True)))
    np.save(ART / "oof_j6_qp_blend.npy", blend_p.astype(np.float32))
    np.save(ART / "test_j6_qp_blend.npy", blend_t.astype(np.float32))

    res_d = dict(
        pool=POOL,
        weights={name: float(w) for (name, _), w in zip(candidates, w_qp)},
        lb4_bal=float(lb4_bal),
        lb4_per_class_recall=lb4_pcr.tolist(),
        qp_bal=float(qp_bal),
        qp_per_class_recall=qp_pcr.tolist(),
        delta_bal=float(delta),
        per_class_delta=pcr_delta.tolist(),
        guardrail_ok=bool(guardrail_ok),
        emit_threshold=2e-4,
        emit=bool(guardrail_ok and delta >= 2e-4),
    )
    with open(ART / "j6_qp_blend_results.json", "w") as f:
        json.dump(res_d, f, indent=2)
    log(f"saved oof_j6_qp_blend.npy, test_j6_qp_blend.npy, j6_qp_blend_results.json")
    log(f"VERDICT: emit={res_d['emit']}  (Δ {delta:+.5f}, guardrail {guardrail_ok})")


if __name__ == "__main__":
    main()
