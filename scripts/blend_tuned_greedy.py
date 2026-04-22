"""Rebuild greedy log-blend + nonrule α-sweep from TUNED OOFs.

Two comparisons:
  A. Rebuild at production weights (greedy 0.45/0.40/0.15, alpha 0.15).
     - If only log-bias retunes, this is the honest HP-tuned equivalent
       of the current LB-best submission.
  B. Nested-CV blend-weight search: 5 outer folds, inner weight fit on
     4 folds, evaluated on held-out fold. Reports the honest OOF and
     a sensitivity sweep.

Inputs (from refit_best_hp.py):
  scripts/artifacts/oof_xgb_dist_routed_v3_tuned.npy
  scripts/artifacts/test_xgb_dist_routed_v3_tuned.npy
  scripts/artifacts/oof_xgb_spec_678_tuned.npy
  scripts/artifacts/test_xgb_spec_678_tuned.npy
  scripts/artifacts/oof_xgb_nonrule_tuned.npy
  scripts/artifacts/test_xgb_nonrule_tuned.npy

Outputs:
  scripts/artifacts/oof_greedy_blend_tuned.npy
  scripts/artifacts/test_greedy_blend_tuned.npy
  scripts/artifacts/blend_tuned_greedy_results.json
  submissions/submission_greedy_nonrule_blend_tuned.csv (only if lift >= +0.0005)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from hp_common import (
    ART_DIR, CLASSES, CLS2IDX, IDX2CLS, OUT_DIR, SEED, TARGET,
    add_distance_features, log, tune_log_bias,
)

N_FOLDS = 5
# Production greedy weights (validated on LB at 0.97296).
GREEDY_W = (0.45, 0.40, 0.15)   # hybrid, routed, spec
# Production nonrule alpha (LB 0.97352).
NONRULE_ALPHA = 0.15
# Thresholds for action.
FOLD_STD = 0.0009               # ~1 sigma on 5-fold OOF
LB_PROBE_THRESHOLD = 5e-4       # OOF lift needed to justify LB probe

ID = "id"


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


def log_blend2(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    logs = la + lb
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def ba_fixed_bias(p: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    lp = np.log(np.clip(p, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(axis=1)))


def nested_cv_blend(
    oof_hyb: np.ndarray, oof_routed: np.ndarray, oof_spec: np.ndarray,
    oof_nonrule: np.ndarray, y: np.ndarray, prior: np.ndarray,
    use_tuned: str = "tuned",
) -> dict:
    """Nested CV over (w_hyb, w_routed, w_spec, alpha_nonrule).

    For each outer fold:
      - Use 4 folds as 'inner': find best weights + bias on inner OOF.
      - Evaluate the fitted weights + bias on the held-out outer fold.
    Returns nested-CV bal_acc (mean + fold std) and chosen weights per fold.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Greedy weight simplex grid (coarse) + alpha grid.
    w_grid = [(0.3, 0.5, 0.2), (0.35, 0.45, 0.2), (0.4, 0.4, 0.2),
              (0.45, 0.40, 0.15), (0.5, 0.35, 0.15),
              (0.4, 0.45, 0.15), (0.5, 0.4, 0.1),
              (0.3, 0.55, 0.15), (0.25, 0.6, 0.15),
              (0.45, 0.45, 0.10), (0.45, 0.35, 0.20), (0.45, 0.50, 0.05),
              (0.40, 0.50, 0.10), (0.55, 0.35, 0.10)]
    alpha_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    fold_scores: list[float] = []
    fold_choices: list[dict] = []
    for fi, (tr_idx, va_idx) in enumerate(skf.split(oof_hyb, y)):
        # Use tr_idx as 'inner' for weight selection, va_idx for evaluation.
        best_score = -1.0
        best_choice = None
        # For efficiency: fit bias once per weight combo on inner.
        for w_hyb, w_routed, w_spec in w_grid:
            greedy_inner = log_blend3(
                oof_hyb[tr_idx], oof_routed[tr_idx], oof_spec[tr_idx],
                w_hyb, w_routed, w_spec,
            )
            bias_inner, _ = tune_log_bias(greedy_inner, y[tr_idx], prior)
            for alpha in alpha_grid:
                blend_inner = log_blend2(oof_nonrule[tr_idx], greedy_inner, alpha)
                score_inner = ba_fixed_bias(blend_inner, y[tr_idx], bias_inner)
                if score_inner > best_score:
                    best_score = score_inner
                    best_choice = (w_hyb, w_routed, w_spec, alpha, bias_inner)
        w_hyb, w_routed, w_spec, alpha, bias_inner = best_choice
        # Apply on held-out outer fold.
        greedy_outer = log_blend3(
            oof_hyb[va_idx], oof_routed[va_idx], oof_spec[va_idx],
            w_hyb, w_routed, w_spec,
        )
        blend_outer = log_blend2(oof_nonrule[va_idx], greedy_outer, alpha)
        outer_score = ba_fixed_bias(blend_outer, y[va_idx], bias_inner)
        fold_scores.append(outer_score)
        fold_choices.append({
            "fold": fi,
            "w_hyb": w_hyb, "w_routed": w_routed, "w_spec": w_spec,
            "alpha": alpha,
            "bias": bias_inner.tolist(),
            "inner_score": float(best_score),
            "outer_score": float(outer_score),
        })
        log(f"nested fold {fi+1}/{N_FOLDS}  "
            f"w=({w_hyb},{w_routed},{w_spec})  α={alpha}  "
            f"inner={best_score:.5f}  outer={outer_score:.5f}")
    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores, ddof=1)),
        "fold_scores": fold_scores,
        "fold_choices": fold_choices,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nested", action="store_true",
                    help="Skip the nested-CV weight search (slow).")
    ap.add_argument("--suffix", default="_tuned",
                    help="Suffix for input arrays (default _tuned). "
                    "Use '' for sanity-check on original artefacts.")
    args = ap.parse_args()

    log("loading train/test (for dgp_score + submission ids)")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr = add_distance_features(tr)
    te = add_distance_features(te)
    score_tr = tr["dgp_score"].values.astype(np.int8)
    score_te = te["dgp_score"].values.astype(np.int8)
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"N_train={len(y)}  prior={prior.round(4).tolist()}")

    suffix = args.suffix
    log(f"loading component OOFs + test probs (suffix={suffix!r})")
    def load_pair(stub: str) -> tuple[np.ndarray, np.ndarray]:
        return (np.load(ART_DIR / f"oof_{stub}{suffix}.npy"),
                np.load(ART_DIR / f"test_{stub}{suffix}.npy"))
    oof_routed, test_routed = load_pair("xgb_dist_routed_v3")
    oof_spec, test_spec = load_pair("xgb_spec_678")
    oof_nonrule, test_nonrule = load_pair("xgb_nonrule")

    log("reconstructing hybrid_v3 (spec override on {6,7,8})")
    oof_hyb = build_hybrid_v3(oof_routed, oof_spec, score_tr)
    test_hyb = build_hybrid_v3(test_routed, test_spec, score_te)

    log(f"building greedy 3-way log-blend at production weights {GREEDY_W}")
    w_h, w_r, w_s = GREEDY_W
    oof_greedy = log_blend3(oof_hyb, oof_routed, oof_spec, w_h, w_r, w_s)
    test_greedy = log_blend3(test_hyb, test_routed, test_spec, w_h, w_r, w_s)

    bias_greedy, tuned_greedy = tune_log_bias(oof_greedy, y, prior)
    log(f"greedy OOF tuned = {tuned_greedy:.5f}  "
        f"bias = {bias_greedy.round(4).tolist()}")

    np.save(ART_DIR / f"oof_greedy_blend{suffix}.npy", oof_greedy)
    np.save(ART_DIR / f"test_greedy_blend{suffix}.npy", test_greedy)

    # Fixed-bias sweep of nonrule α (matches production honest methodology).
    log("fixed greedy-bias sweep over nonrule α")
    sweep = []
    for alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend_oof = log_blend2(oof_nonrule, oof_greedy, alpha)
        ba = ba_fixed_bias(blend_oof, y, bias_greedy)
        sweep.append({"alpha": alpha, "oof": float(ba),
                      "delta_vs_greedy": float(ba - tuned_greedy)})
        log(f"  α={alpha:.2f}  OOF(fixed-bias) = {ba:.5f}  "
            f"Δ vs greedy = {ba - tuned_greedy:+.5f}")
    best_alpha = max(sweep, key=lambda d: d["oof"])
    log(f"best α={best_alpha['alpha']}  OOF={best_alpha['oof']:.5f}  "
        f"Δ={best_alpha['delta_vs_greedy']:+.5f}")

    # Also compute OOF at production α=0.15 (fixed) for direct comparison.
    blend_prod = log_blend2(oof_nonrule, oof_greedy, NONRULE_ALPHA)
    prod_bal = ba_fixed_bias(blend_prod, y, bias_greedy)
    log(f"OOF at production α={NONRULE_ALPHA} = {prod_bal:.5f}")

    # Compare to current best OOF 0.97421 (production greedy+nonrule).
    current_best = 0.97421
    delta = prod_bal - current_best
    log(f"Δ vs current best 0.97421 = {delta:+.5f}  "
        f"(fold-std = {FOLD_STD:.4f})")

    results = {
        "suffix": suffix,
        "production_greedy_weights": list(GREEDY_W),
        "production_nonrule_alpha": NONRULE_ALPHA,
        "greedy_tuned_oof": float(tuned_greedy),
        "greedy_bias": bias_greedy.tolist(),
        "alpha_sweep_fixed_bias": sweep,
        "best_alpha": best_alpha,
        "oof_at_production_alpha": float(prod_bal),
        "current_best_oof": current_best,
        "delta_vs_current_best": float(delta),
    }

    # Nested-CV blend weights (optional).
    if not args.skip_nested:
        log("starting nested-CV blend-weight search...")
        t0 = time.time()
        nested = nested_cv_blend(
            oof_hyb, oof_routed, oof_spec, oof_nonrule, y, prior,
        )
        results["nested_cv"] = nested
        log(f"nested CV: mean OOF = {nested['mean']:.5f} "
            f"(std {nested['std']:.5f})  [{time.time()-t0:.1f}s]")
    else:
        log("skipping nested-CV weight search")

    # Write submission only if OOF lift is meaningful AND uses production α.
    if prod_bal >= current_best + LB_PROBE_THRESHOLD:
        blend_test = log_blend2(test_nonrule, test_greedy, NONRULE_ALPHA)
        lp_test = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp_test + bias_greedy).argmax(axis=1)
        sub_path = OUT_DIR / f"submission_greedy_nonrule_blend{suffix}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote submission candidate {sub_path}  "
            f"(DO NOT AUTO-SUBMIT — requires user approval per CLAUDE.md)")
        results["submission"] = {"path": str(sub_path),
                                  "oof": float(prod_bal),
                                  "delta_vs_current_best": float(delta)}
    else:
        log(f"OOF lift {delta:+.5f} below LB-probe threshold "
            f"{LB_PROBE_THRESHOLD:.4f} — no submission written")
        results["submission"] = None

    # Confusion matrix at production α.
    cm = confusion_matrix(
        y, (np.log(np.clip(blend_prod, 1e-9, 1.0)) + bias_greedy).argmax(axis=1),
    )
    results["confusion_matrix_at_prod_alpha"] = cm.tolist()
    log(f"OOF confusion matrix (α={NONRULE_ALPHA}):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    with open(ART_DIR / "blend_tuned_greedy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved {ART_DIR}/blend_tuned_greedy_results.json")


if __name__ == "__main__":
    main()
