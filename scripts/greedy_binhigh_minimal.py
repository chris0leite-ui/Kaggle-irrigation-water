"""Stack binhigh into the greedy 3-way log-blend with minimal tuning.

The 2026-04-21 binhigh experiment added +0.00036 OOF but lost 0.00084 LB
vs the greedy baseline. Diagnosis: 75-point blend sweep + log-bias
retune compounded selection bias. This script tests the same lever
with ONE extra parameter (logit-add lam on binhigh's High prob) and
the GREEDY'S fitted log-bias reused as-is.

Greedy recipe (from LB-0.97296 submission):
  log-blend  0.45 * hybrid_v3 + 0.40 * routed_v3 + 0.15 * spec_678

hybrid_v3[i] = spec_678[i] if dgp_score(i) in {6,7,8} else routed_v3[i]

Artefacts:
  scripts/artifacts/oof_greedy_blend.npy
  scripts/artifacts/test_greedy_blend.npy
  scripts/artifacts/greedy_binhigh_minimal_results.json
  submissions/submission_greedy_binhigh_minimal.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from benchmark_xgb_dist import (
    CLASSES, CLS2IDX, IDX2CLS, ID, TARGET,
    add_distance_features, tune_log_bias,
)


SEED = 42
N_FOLDS = 5
ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_hybrid_v3(routed: np.ndarray, spec: np.ndarray, score: np.ndarray) -> np.ndarray:
    spec_mask = np.isin(score, (6, 7, 8))
    out = routed.copy()
    out[spec_mask] = spec[spec_mask]
    return out


def log_blend3(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray,
               w_a: float, w_b: float, w_c: float) -> np.ndarray:
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = w_b * np.log(np.clip(p_b, 1e-9, 1.0))
    lc = w_c * np.log(np.clip(p_c, 1e-9, 1.0))
    logs = la + lb + lc
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def add_high_logit(p: np.ndarray, phigh: np.ndarray, lam: float) -> np.ndarray:
    logp = np.log(np.clip(p, 1e-9, 1.0))
    lg = np.log(np.clip(phigh, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - phigh, 1e-9, 1.0))
    logp[:, 2] += lam * lg
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def tuned_bal_acc_with_fixed_bias(p: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    lp = np.log(np.clip(p, 1e-9, 1.0))
    return balanced_accuracy_score(y, (lp + bias).argmax(axis=1))


def main() -> None:
    log("loading train/test for dgp_score")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    score_tr = tr["dgp_score"].values.astype(np.int8)
    score_te = te["dgp_score"].values.astype(np.int8)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"N_train={len(y)}  prior={prior.round(4).tolist()}")

    log("loading component OOFs + test probs")
    oof_routed = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_routed = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")

    log("reconstructing hybrid_v3 (spec override on {6,7,8})")
    oof_hyb = build_hybrid_v3(oof_routed, oof_spec, score_tr)
    test_hyb = build_hybrid_v3(test_routed, test_spec, score_te)

    log("building greedy 3-way log-blend (0.45 hybrid + 0.40 routed + 0.15 spec)")
    oof_greedy = log_blend3(oof_hyb, oof_routed, oof_spec, 0.45, 0.40, 0.15)
    test_greedy = log_blend3(test_hyb, test_routed, test_spec, 0.45, 0.40, 0.15)

    bias_greedy, tuned_greedy = tune_log_bias(oof_greedy, y, prior)
    log(f"greedy OOF tuned = {tuned_greedy:.5f}  (target: 0.97375)")
    log(f"greedy bias = {bias_greedy.round(4).tolist()}")

    np.save(ART / "oof_greedy_blend.npy", oof_greedy)
    np.save(ART / "test_greedy_blend.npy", test_greedy)

    log("loading binary-High head (binhigh)")
    oof_binhigh = np.load(ART / "oof_xgb_bin_high.npy")
    test_binhigh = np.load(ART / "test_xgb_bin_high.npy")

    log("coarse lambda sweep on greedy + logit-add binhigh (FIXED greedy bias)")
    results = {
        "greedy_tuned_oof": float(tuned_greedy),
        "greedy_bias": bias_greedy.tolist(),
        "sweep": [],
    }

    # Fine-grained sweep over a small range first to find the peak region.
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for lam in grid:
        blend_oof = add_high_logit(oof_greedy, oof_binhigh, lam)
        ba = tuned_bal_acc_with_fixed_bias(blend_oof, y, bias_greedy)
        results["sweep"].append({"lam": lam, "oof_bal_acc_fixed_bias": float(ba)})
        log(f"  lam={lam:.2f}  OOF (fixed bias) = {ba:.5f}  Δ = {ba - tuned_greedy:+.5f}")

    sweep = results["sweep"]
    best = max(sweep, key=lambda d: d["oof_bal_acc_fixed_bias"])
    best_lam = best["lam"]
    best_oof = best["oof_bal_acc_fixed_bias"]

    if best_lam == 0.0 or best_oof - tuned_greedy < 1e-5:
        log("no OOF lift from binhigh with fixed bias — abort submission")
        results["action"] = "no_submission"
        results["best_lam"] = best_lam
        results["best_oof"] = float(best_oof)
    else:
        log(f"best lam={best_lam}  OOF={best_oof:.5f}  Δ={best_oof - tuned_greedy:+.5f}")
        blend_test = add_high_logit(test_greedy, test_binhigh, best_lam)
        lp_test = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp_test + bias_greedy).argmax(axis=1)
        sub = pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]})
        sub_path = OUT / "submission_greedy_binhigh_minimal.csv"
        sub.to_csv(sub_path, index=False)
        log(f"wrote {sub_path}")
        results["action"] = "submit"
        results["best_lam"] = best_lam
        results["best_oof"] = float(best_oof)
        results["submission_path"] = str(sub_path)

        blend_oof = add_high_logit(oof_greedy, oof_binhigh, best_lam)
        cm = confusion_matrix(y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
        log(f"OOF confusion matrix:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    with open(ART / "greedy_binhigh_minimal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/greedy_binhigh_minimal_results.json")


if __name__ == "__main__":
    main()
