"""Fixed-bias blend sweep: monotone XGB onto greedy and LB-best baselines.

Baselines:
  greedy           = oof_greedy_blend.npy / test_greedy_blend.npy
                     (hybrid_v3 0.45 + routed_v3 0.40 + spec_678 0.15)
  greedy+nonrule   = log_blend(greedy, xgb_nonrule; w=[0.85, 0.15])
                     (this is the current LB best at 0.97352)

For each α in a coarse grid, log-blend monotone OOF with the baseline
at weight α (monotone) / 1-α (baseline), tune log-bias, report.

The "fixed-bias" protocol matters: the bias is tuned ONCE per blend
configuration, not retuned per component added. Previous blend
experiments proved that retuning bias manufactures OOF lift that
does NOT transfer to LB (binhigh example).

Also: Jaccard-of-errors between monotone and each baseline gives a
pure architectural-orthogonality diagnostic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "scripts/artifacts"
OUT = ROOT / "submissions"

CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ID = "id"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_blend(oofs, weights, eps=1e-9):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def tune_log_bias(oof, y, prior, grid=None, n_rounds=25, tol=1e-6):
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 61)
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    for _ in range(n_rounds):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + tol:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def jaccard_err(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> float:
    a = set(np.where(pred_a != y)[0].tolist())
    b = set(np.where(pred_b != y)[0].tolist())
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def main():
    log("loading train labels")
    tr = pd.read_csv(ROOT / "data/train.csv")
    y_str = tr["Irrigation_Need"].astype(str).values
    y = np.array([CLS2IDX[s] for s in y_str], dtype=np.int32)
    prior = np.bincount(y) / len(y)

    log("loading OOFs + test probs")
    oof_m = np.load(ART / "oof_xgb_dist_monotone.npy")
    test_m = np.load(ART / "test_xgb_dist_monotone.npy")
    oof_g = np.load(ART / "oof_greedy_blend.npy")
    test_g = np.load(ART / "test_greedy_blend.npy")
    oof_n = np.load(ART / "oof_xgb_nonrule.npy")
    test_n = np.load(ART / "test_xgb_nonrule.npy")

    # LB-best = greedy + nonrule log-blend at α_nr=0.15
    oof_lb = log_blend([oof_g, oof_n], [0.85, 0.15])
    test_lb = log_blend([test_g, test_n], [0.85, 0.15])

    # Sanity: check the LB-best OOF bal_acc matches the logged 0.97421
    bias_lb, tuned_lb = tune_log_bias(oof_lb, y, prior)
    bias_g, tuned_g = tune_log_bias(oof_g, y, prior)
    bias_m, tuned_m = tune_log_bias(oof_m, y, prior)
    log(f"baseline OOF tuned: greedy={tuned_g:.5f}  LB-best(greedy+nonrule)={tuned_lb:.5f}  "
        f"monotone_standalone={tuned_m:.5f}")

    # Error Jaccards for orthogonality diagnostic
    pred_m = (np.log(np.clip(oof_m, 1e-9, 1.0)) + bias_m).argmax(axis=1)
    pred_g = (np.log(np.clip(oof_g, 1e-9, 1.0)) + bias_g).argmax(axis=1)
    pred_lb = (np.log(np.clip(oof_lb, 1e-9, 1.0)) + bias_lb).argmax(axis=1)
    jac_mg = jaccard_err(pred_m, pred_g, y)
    jac_ml = jaccard_err(pred_m, pred_lb, y)
    err_m = int((pred_m != y).sum())
    err_g = int((pred_g != y).sum())
    err_lb = int((pred_lb != y).sum())
    log(f"Jaccard errs: monotone vs greedy = {jac_mg:.4f}  (m={err_m}  g={err_g})")
    log(f"Jaccard errs: monotone vs LB-best = {jac_ml:.4f}  (m={err_m}  lb={err_lb})")

    # Sweep α for monotone onto each baseline at FIXED baseline bias.
    # "Fixed bias" means: use bias_g (resp. bias_lb) throughout the sweep.
    # This is the honest selection-free test.
    alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    results = {"standalone": {"greedy": tuned_g, "lb_best": tuned_lb,
                              "monotone": tuned_m},
               "jaccard": {"vs_greedy": jac_mg, "vs_lb_best": jac_ml,
                           "errs_m": err_m, "errs_g": err_g, "errs_lb": err_lb}}

    for label, oof_base, test_base, bias_base, base_tuned in [
        ("greedy", oof_g, test_g, bias_g, tuned_g),
        ("lb_best", oof_lb, test_lb, bias_lb, tuned_lb),
    ]:
        log(f"\nSweep: monotone onto {label}  (fixed bias)")
        log(f"  baseline tuned OOF = {base_tuned:.5f}")
        sweep = []
        for a in alphas:
            if a == 0.0:
                blend_oof = oof_base
                blend_test = test_base
            elif a == 1.0:
                blend_oof = oof_m
                blend_test = test_m
            else:
                blend_oof = log_blend([oof_m, oof_base], [a, 1 - a])
                blend_test = log_blend([test_m, test_base], [a, 1 - a])
            log_blend_oof = np.log(np.clip(blend_oof, 1e-9, 1.0))
            pred = (log_blend_oof + bias_base).argmax(axis=1)
            tuned = balanced_accuracy_score(y, pred)
            sweep.append({"alpha": float(a), "tuned_bal_acc": float(tuned),
                          "delta_vs_base": float(tuned - base_tuned)})
            log(f"  alpha={a:.3f}  tuned_bal_acc={tuned:.5f}  Δ={tuned-base_tuned:+.5f}")
        results[f"sweep_{label}"] = sweep
        best = max(sweep, key=lambda r: r["tuned_bal_acc"])
        log(f"  best: α={best['alpha']}  tuned={best['tuned_bal_acc']:.5f}  "
            f"Δ={best['delta_vs_base']:+.5f}")
        results[f"best_{label}"] = best

        # If best beats base by ≥ 0.0005, emit a submission candidate
        if best["delta_vs_base"] >= 0.0005 and 0 < best["alpha"] < 1:
            a = best["alpha"]
            blend_test = log_blend([test_m, test_base], [a, 1 - a])
            pred_test = (np.log(np.clip(blend_test, 1e-9, 1.0)) + bias_base).argmax(axis=1)
            te_ids = pd.read_csv(ROOT / "data/test.csv", usecols=[ID])[ID].values
            sub = pd.DataFrame({ID: te_ids, "Irrigation_Need":
                                [CLASSES[i] for i in pred_test]})
            fn = f"submission_monotone_plus_{label}_alpha{int(a*1000):03d}.csv"
            sub.to_csv(OUT / fn, index=False)
            log(f"  LB candidate emitted: {fn}")

    with open(ART / "blend_monotone_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nSaved -> {ART}/blend_monotone_results.json")


if __name__ == "__main__":
    main()
