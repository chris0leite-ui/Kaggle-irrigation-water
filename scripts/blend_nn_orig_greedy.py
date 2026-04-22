"""Blend NN-on-original ensemble OOF into greedy with FIXED greedy bias.

Exact mirror of the nonrule_features_only blend protocol:
  1. Load greedy blend OOF + its fitted bias.
  2. Log-space blend sweep over alpha (NN-orig weight).
  3. Fixed bias, no retune — mandatory post-binhigh rule.
  4. Only consider LB probe if fixed-bias OOF lifts >= +0.0005.

Inputs (must exist):
  scripts/artifacts/oof_greedy_blend.npy
  scripts/artifacts/test_greedy_blend.npy
  scripts/artifacts/greedy_binhigh_minimal_results.json  (bias source)
  scripts/artifacts/oof_nn_orig_ens.npy
  scripts/artifacts/test_nn_orig_ens.npy

Outputs:
  scripts/artifacts/blend_nn_orig_greedy_results.json
  submissions/submission_greedy_nn_orig_blend.csv  (if lift > 5e-4)
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


def main():
    log("loading greedy + NN-orig OOFs")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = float(greedy_res["greedy_tuned_oof"])
    log(f"  greedy baseline tuned OOF = {tuned_greedy:.5f}  "
        f"bias = {bias_greedy.round(4).tolist()}")

    oof_nn = np.load(ART / "oof_nn_orig_ens.npy")
    test_nn = np.load(ART / "test_nn_orig_ens.npy")
    log(f"  greedy shape {oof_greedy.shape}  nn shape {oof_nn.shape}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)

    # Standalone diagnostics for NN-orig
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

    prior = np.bincount(y, minlength=3) / len(y)
    nn_argmax = balanced_accuracy_score(y, oof_nn.argmax(axis=1))
    _, nn_tuned = tune_log_bias(oof_nn, y, prior)
    log(f"  NN-orig standalone OOF: argmax={nn_argmax:.5f}  tuned={nn_tuned:.5f}")

    # Agreement / overlap diagnostics
    ens_pred = oof_nn.argmax(axis=1)
    gr_pred = oof_greedy.argmax(axis=1)
    agreement = (ens_pred == gr_pred).mean()
    e_ens = set(np.where(ens_pred != y)[0])
    e_gr = set(np.where(gr_pred != y)[0])
    jac = len(e_ens & e_gr) / (len(e_ens | e_gr) or 1)
    log(f"  argmax agreement NN vs greedy = {agreement:.4f}")
    log(f"  error Jaccard NN vs greedy = {jac:.4f}  "
        f"(nn errs={len(e_ens)}, greedy errs={len(e_gr)})")

    results = {
        "nn_standalone_argmax": float(nn_argmax),
        "nn_standalone_tuned":  float(nn_tuned),
        "greedy_tuned_oof": tuned_greedy,
        "greedy_bias": bias_greedy.tolist(),
        "argmax_agreement_nn_vs_greedy": float(agreement),
        "error_jaccard_nn_vs_greedy": float(jac),
        "sweep_log_blend": [],
    }

    log("fixed-bias log-blend sweep over alpha (NN weight)")
    grid = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18,
            0.20, 0.25, 0.30, 0.40, 0.50]
    best = {"alpha": 0.0, "oof": tuned_greedy, "delta_vs_greedy": 0.0}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_greedy
        else:
            blend_oof = log_blend2(oof_nn, oof_greedy, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - tuned_greedy
        results["sweep_log_blend"].append({
            "alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)
        })
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)}
            marker = "  <- best"
        log(f"  alpha_nn={alpha:.2f}  OOF (fixed bias) = {ba:.5f}  "
            f"Δ = {delta:+.5f}{marker}")

    results["best"] = best
    log(f"best alpha={best['alpha']}  OOF={best['oof']:.5f}  Δ={best['delta_vs_greedy']:+.5f}")

    # Decision
    if best["alpha"] == 0.0 or best["delta_vs_greedy"] < 1e-5:
        log("no OOF lift from NN-orig blend — null result, no submission")
        results["action"] = "no_submission"
    elif best["delta_vs_greedy"] < 5e-4:
        log(f"OOF lift {best['delta_vs_greedy']:+.5f} is below the +0.0005 LB-probe"
            " threshold. Borderline: emit submission but do not auto-submit.")
        a = best["alpha"]
        test_blend = log_blend2(test_nn, test_greedy, a)
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_nn_orig_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"wrote borderline submission to {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best["alpha"]
        blend_oof = log_blend2(oof_nn, oof_greedy, a)
        cm = confusion_matrix(
            y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        log(f"OOF confusion matrix at best alpha={a}:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
        test_blend = log_blend2(test_nn, test_greedy, a)
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_nn_orig_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "blend_nn_orig_greedy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/blend_nn_orig_greedy_results.json")


if __name__ == "__main__":
    main()
