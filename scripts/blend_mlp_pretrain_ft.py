"""Blend pretrain-finetune MLP OOF into greedy with FIXED greedy bias.

Mirror of blend_nn_orig_greedy.py, targeting the idea-2 OOF.
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
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main():
    log("loading greedy + pretrain-ft OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = float(greedy_res["greedy_tuned_oof"])
    log(f"  greedy tuned OOF = {tuned_greedy:.5f}  bias={bias_greedy.round(4).tolist()}")

    oof_ft = np.load(ART / "oof_mlp_pretrain_ft.npy")
    test_ft = np.load(ART / "test_mlp_pretrain_ft.npy")
    log(f"  greedy shape {oof_greedy.shape}  ft shape {oof_ft.shape}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    prior = np.bincount(y, minlength=3) / len(y)

    ft_argmax = balanced_accuracy_score(y, oof_ft.argmax(axis=1))
    _, ft_tuned = tune_log_bias(oof_ft, y, prior)
    log(f"  pretrain-ft standalone OOF: argmax={ft_argmax:.5f}  tuned={ft_tuned:.5f}")

    ft_pred = oof_ft.argmax(axis=1)
    gr_pred = oof_greedy.argmax(axis=1)
    agreement = (ft_pred == gr_pred).mean()
    e_ft = set(np.where(ft_pred != y)[0])
    e_gr = set(np.where(gr_pred != y)[0])
    jac = len(e_ft & e_gr) / (len(e_ft | e_gr) or 1)
    log(f"  argmax agreement ft vs greedy = {agreement:.4f}")
    log(f"  error Jaccard ft vs greedy = {jac:.4f}  "
        f"(ft errs={len(e_ft)}, greedy errs={len(e_gr)})")

    results = {
        "ft_standalone_argmax": float(ft_argmax),
        "ft_standalone_tuned":  float(ft_tuned),
        "greedy_tuned_oof": tuned_greedy,
        "greedy_bias": bias_greedy.tolist(),
        "argmax_agreement_ft_vs_greedy": float(agreement),
        "error_jaccard_ft_vs_greedy": float(jac),
        "sweep_log_blend": [],
    }

    log("fixed-bias log-blend sweep over alpha (ft weight)")
    grid_a = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
              0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
    best = {"alpha": 0.0, "oof": tuned_greedy, "delta_vs_greedy": 0.0}
    for alpha in grid_a:
        if alpha == 0.0:
            blend_oof = oof_greedy
        elif alpha == 1.0:
            blend_oof = oof_ft
        else:
            blend_oof = log_blend2(oof_ft, oof_greedy, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - tuned_greedy
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)}
            marker = "  <- best"
        results["sweep_log_blend"].append({
            "alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)
        })
        log(f"  alpha_ft={alpha:.2f}  OOF (fixed bias) = {ba:.5f}  Δ = {delta:+.5f}{marker}")

    results["best"] = best
    log(f"best alpha={best['alpha']}  OOF={best['oof']:.5f}  Δ={best['delta_vs_greedy']:+.5f}")

    # Also try onto the current LB-best (greedy + nonrule):
    # reconstruct: 0.85 * greedy + 0.15 * nonrule (we don't have a saved
    # oof_greedy_nonrule; compose it with log_blend2).
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_base = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_base = log_blend2(test_nonrule, test_greedy, 0.15)
    base_ba = balanced_accuracy_score(y,
        (np.log(np.clip(oof_base, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"  greedy+nonrule (LB-best composition) OOF = {base_ba:.5f}")

    sweep_onto_base = []
    best_b = {"alpha": 0.0, "oof": float(base_ba), "delta_vs_base": 0.0}
    for alpha in grid_a:
        if alpha == 0.0:
            blend_oof = oof_base
        else:
            blend_oof = log_blend2(oof_ft, oof_base, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - base_ba
        marker = ""
        if ba > best_b["oof"]:
            best_b = {"alpha": alpha, "oof": float(ba), "delta_vs_base": float(delta)}
            marker = "  <- best"
        sweep_onto_base.append({"alpha": alpha, "oof": float(ba),
                                "delta_vs_base": float(delta)})
        log(f"  alpha_ft (onto greedy+nonrule)={alpha:.2f}  "
            f"OOF={ba:.5f}  Δ={delta:+.5f}{marker}")

    results["sweep_onto_greedy_nonrule"] = sweep_onto_base
    results["best_onto_greedy_nonrule"] = best_b

    # Decision: primary track vs greedy
    if best["alpha"] == 0.0 or best["delta_vs_greedy"] < 1e-5:
        log("vs greedy: no OOF lift — null")
        results["action_vs_greedy"] = "no_submission"
    elif best["delta_vs_greedy"] < 5e-4:
        log(f"vs greedy: lift {best['delta_vs_greedy']:+.5f} below threshold — borderline")
        results["action_vs_greedy"] = "borderline"
    else:
        a = best["alpha"]
        bl = log_blend2(test_ft, test_greedy, a) if 0 < a < 1 else (
            test_greedy if a == 0 else test_ft)
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_mlp_pretrain_ft_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action_vs_greedy"] = "ready_to_submit"
        results["submission_path_vs_greedy"] = str(sub)

    # Secondary: onto greedy+nonrule (LB best)
    if best_b["alpha"] == 0.0 or best_b["delta_vs_base"] < 1e-5:
        log("vs greedy+nonrule: no OOF lift — null")
        results["action_vs_base"] = "no_submission"
    elif best_b["delta_vs_base"] < 5e-4:
        log(f"vs greedy+nonrule: lift {best_b['delta_vs_base']:+.5f} below threshold — borderline")
        a = best_b["alpha"]
        bl = log_blend2(test_ft, test_base, a) if 0 < a < 1 else (
            test_base if a == 0 else test_ft)
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_nonrulebest_mlp_pretrain_ft_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote borderline {sub}")
        results["action_vs_base"] = "borderline_no_submit"
        results["submission_path_vs_base"] = str(sub)
    else:
        a = best_b["alpha"]
        bl = log_blend2(test_ft, test_base, a) if 0 < a < 1 else (
            test_base if a == 0 else test_ft)
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_nonrulebest_mlp_pretrain_ft_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action_vs_base"] = "ready_to_submit"
        results["submission_path_vs_base"] = str(sub)

    with open(ART / "blend_mlp_pretrain_ft_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/blend_mlp_pretrain_ft_results.json")


if __name__ == "__main__":
    main()
