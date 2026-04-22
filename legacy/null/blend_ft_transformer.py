"""Fixed-bias blend sweep: FT-Transformer OOF into greedy + greedy+nonrule.

FT-Transformer standalone tuned 0.96780 (Jaccard 0.587 vs greedy on
fold 1 — lowest Jaccard of any NN we've run). Same fixed-bias
protocol as all other NN blend tests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            bals = []
            for g in grid:
                base[k] = bias[k] + g
                bals.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(bals))
            if bals[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = bals[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def main():
    log("loading OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_ftt = np.load(ART / "oof_ft_transformer.npy")
    test_ftt = np.load(ART / "test_ft_transformer.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior = np.bincount(y, minlength=3) / len(y)

    ft_argmax = balanced_accuracy_score(y, oof_ftt.argmax(axis=1))
    _, ft_tuned = tune_log_bias(oof_ftt, y, prior)
    log(f"FT-Transformer standalone: argmax={ft_argmax:.5f}  tuned={ft_tuned:.5f}")

    # Reconstruct LB-best: greedy + nonrule @ alpha=0.15
    oof_base = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_base = log_blend2(test_nonrule, test_greedy, 0.15)
    base_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_base, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    greedy_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_greedy, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"greedy baseline: {greedy_ba:.5f}  (ref)")
    log(f"LB-best ref (greedy+nonrule@0.15): {base_ba:.5f}")

    # Diagnostics
    ft_pred = oof_ftt.argmax(axis=1)
    gr_pred = oof_greedy.argmax(axis=1)
    base_pred = (np.log(np.clip(oof_base, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    e_ft = set(np.where(ft_pred != y)[0])
    e_gr = set(np.where(gr_pred != y)[0])
    e_base = set(np.where(base_pred != y)[0])
    jac_gr = len(e_ft & e_gr) / (len(e_ft | e_gr) or 1)
    jac_base = len(e_ft & e_base) / (len(e_ft | e_base) or 1)
    log(f"errors: ft={len(e_ft)}  greedy={len(e_gr)}  base(lb)={len(e_base)}")
    log(f"Jaccard ft vs greedy: {jac_gr:.4f}")
    log(f"Jaccard ft vs lb-best: {jac_base:.4f}")

    results = {
        "ft_standalone_argmax": float(ft_argmax),
        "ft_standalone_tuned": float(ft_tuned),
        "greedy_tuned_oof": float(greedy_ba),
        "lbbest_tuned_oof": float(base_ba),
        "greedy_bias": bias_greedy.tolist(),
        "ft_errors": len(e_ft),
        "greedy_errors": len(e_gr),
        "lbbest_errors": len(e_base),
        "jaccard_ft_vs_greedy": float(jac_gr),
        "jaccard_ft_vs_lbbest": float(jac_base),
        "sweep_vs_greedy": [],
        "sweep_vs_lbbest": [],
    }

    log("sweep 1: greedy + FT-T at fixed greedy bias")
    grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    best_g = {"alpha": 0.0, "oof": greedy_ba, "delta": 0.0}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_greedy
        elif alpha == 1.0:
            blend_oof = oof_ftt
        else:
            blend_oof = log_blend2(oof_ftt, oof_greedy, alpha)
        ba = balanced_accuracy_score(y,
            (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
        delta = ba - greedy_ba
        marker = ""
        if ba > best_g["oof"]:
            best_g = {"alpha": alpha, "oof": float(ba), "delta": float(delta)}
            marker = "  <- best"
        results["sweep_vs_greedy"].append({"alpha": alpha, "oof": float(ba),
                                            "delta_vs_greedy": float(delta)})
        log(f"  alpha_ftt={alpha:.2f}  OOF={ba:.5f}  Δ_greedy={delta:+.5f}{marker}")

    log("sweep 2: lb-best + FT-T at fixed greedy bias")
    best_b = {"alpha": 0.0, "oof": base_ba, "delta": 0.0}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_base
        elif alpha == 1.0:
            blend_oof = oof_ftt
        else:
            blend_oof = log_blend2(oof_ftt, oof_base, alpha)
        ba = balanced_accuracy_score(y,
            (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
        delta = ba - base_ba
        marker = ""
        if ba > best_b["oof"]:
            best_b = {"alpha": alpha, "oof": float(ba), "delta": float(delta)}
            marker = "  <- best"
        results["sweep_vs_lbbest"].append({"alpha": alpha, "oof": float(ba),
                                            "delta_vs_lbbest": float(delta)})
        log(f"  alpha_ftt={alpha:.2f}  OOF={ba:.5f}  Δ_lbbest={delta:+.5f}{marker}")

    results["best_vs_greedy"] = best_g
    results["best_vs_lbbest"] = best_b

    log(f"best vs greedy: alpha={best_g['alpha']}  OOF={best_g['oof']:.5f}  "
        f"Δ={best_g['delta']:+.5f}")
    log(f"best vs lb-best: alpha={best_b['alpha']}  OOF={best_b['oof']:.5f}  "
        f"Δ={best_b['delta']:+.5f}")

    # Decision: only consider LB probe if vs LB-best Δ >= +0.0003
    if best_b["delta"] < 1e-5:
        log("no lift vs LB-best — null")
        results["action"] = "no_submission"
    elif best_b["delta"] < 3e-4:
        log(f"lift vs LB-best Δ={best_b['delta']:+.5f} below +0.0003 — borderline")
        a = best_b["alpha"]
        test_blend = log_blend2(test_ftt, test_base, a) if 0 < a < 1 else (
            test_base if a == 0 else test_ftt)
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_lbbest_ftt_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"borderline submission {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best_b["alpha"]
        test_blend = log_blend2(test_ftt, test_base, a) if 0 < a < 1 else (
            test_base if a == 0 else test_ftt)
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_lbbest_ftt_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "blend_ft_transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/blend_ft_transformer_results.json")


if __name__ == "__main__":
    main()
