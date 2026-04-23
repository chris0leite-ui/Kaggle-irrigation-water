"""Greedy forward-selection log-blend over the full OOF bank.

Current LB-best is a 2-way log-blend: digits-OTE (0.4) × digit-XGB (0.6).
This script extends to greedy multi-way selection over all saved OOFs,
using the proven 2026-04-21 procedure:

  1. Evaluate each candidate standalone under digit-XGB's log-bias
     (fixed, no retune per candidate).
  2. Pick the best standalone as anchor.
  3. For each candidate NOT in the blend, find the α ∈ [0, 0.5] that
     maximises tuned log-bias OOF of `α × candidate + (1-α) × current_blend`.
     Pick the one with the largest improvement.
  4. Stop when no addition improves OOF by >= +1e-5, OR the component
     reduces tuned OOF under fixed bias.

Key principle from CLAUDE.md: use FIXED bias (no retune per candidate)
to avoid the selection-overfit that killed the binhigh experiment. The
bias is the digit-XGB tuned bias since that's the LB-calibrated anchor.

Output: best blend weights, expected LB estimate, emit submission if
delta vs LB-best >= +0.0001 (smaller than the +0.0005 threshold since
multi-way blend on saved OOFs is LOW-cost selection risk).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Reasonable candidates on disk. Skip:
#   - xgb_bin_high / hybrid_binhigh: known-overfit (-0.00084 LB previously)
#   - xgb_spec_678: domain-restricted, not global probabilities
#   - lb_best_fs7 / fs123: fold-seed bag components, overfit on OOF
CANDIDATES = [
    ("digit_xgb",        "oof_xgb_dist_digits.npy",              "test_xgb_dist_digits.npy"),
    ("digits_ote",       "oof_xgb_dist_digits_ote_digits.npy",   "test_xgb_dist_digits_ote_digits.npy"),
    ("digits_pairs",     "oof_xgb_dist_digits_ote_digits_pairs.npy", "test_xgb_dist_digits_ote_digits_pairs.npy"),
    ("digits_light_ote", "oof_xgb_dist_digits_ote_digits_light.npy", "test_xgb_dist_digits_ote_digits_light.npy"),
    ("cat_ote",          "oof_xgb_dist_digits_ote.npy",          "test_xgb_dist_digits_ote.npy"),
    ("cat_ote_light",    "oof_xgb_dist_digits_ote_light.npy",    "test_xgb_dist_digits_ote_light.npy"),
    ("lgbm_digit",       "oof_lgbm_dist_digits.npy",             "test_lgbm_dist_digits.npy"),
    ("xgb_nonrule",      "oof_xgb_nonrule.npy",                  "test_xgb_nonrule.npy"),
    ("xgb_vanilla_dist", "oof_xgb_vanilla_dist.npy",             "test_xgb_vanilla_dist.npy"),
    ("xgb_routed_v3",    "oof_xgb_dist_routed_v3.npy",           "test_xgb_dist_routed_v3.npy"),
    ("hybrid_lgbmxgb",   "oof_hybrid_lgbmxgb_blend.npy",         "test_hybrid_lgbmxgb_blend.npy"),
    ("xgb_corn",         "oof_xgb_corn.npy",                     "test_xgb_corn.npy"),
    ("lgbm_te_orig",     "oof_lgbm_te_orig.npy",                 "test_lgbm_te_orig.npy"),
    ("tabpfn",           "oof_tabpfn.npy",                       "test_tabpfn.npy"),
    ("greedy_blend",     "oof_greedy_blend.npy",                 "test_greedy_blend.npy"),
]


def load_components():
    comps = {}
    for name, oof_name, test_name in CANDIDATES:
        op = ART / oof_name
        tp = ART / test_name
        if not op.exists() or not tp.exists():
            log(f"  skip {name}: missing {op.name if not op.exists() else tp.name}")
            continue
        comps[name] = {
            "oof": np.load(op),
            "test": np.load(tp),
        }
    return comps


def log_blend_list(probs_list, weights) -> np.ndarray:
    """Weighted geometric mean (log-space blend) of 3-class probs."""
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def fixed_bias_bal_acc(oof, y, bias):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(axis=1)))


def main() -> None:
    log("loading components")
    comps = load_components()
    log(f"loaded {len(comps)} components: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Anchor bias = digit-XGB's tuned bias (LB-calibrated).
    digit_res = json.loads((ART / "xgb_dist_digits_results.json").read_text())
    bias = np.array(digit_res["log_bias"])
    log(f"anchor bias (digit-XGB's tuned) = {bias.round(4).tolist()}")

    # Standalone OOF under fixed bias.
    log("--- standalone OOF at fixed digit-XGB bias ---")
    standalone = {}
    for n, v in comps.items():
        ba = fixed_bias_bal_acc(v["oof"], y, bias)
        standalone[n] = ba
        log(f"  {n:25s}  OOF = {ba:.5f}")

    # Baseline reference: digits-OTE × digit-XGB at α=0.40 (current LB-best blend).
    lb_best_oof = log_blend_list(
        [comps["digits_ote"]["oof"], comps["digit_xgb"]["oof"]], [0.4, 0.6]
    )
    lb_best_ba = fixed_bias_bal_acc(lb_best_oof, y, bias)
    log(f"reference LB-best blend (digits_ote 0.4 / digit_xgb 0.6) = {lb_best_ba:.5f}")

    # Greedy forward selection.
    # Start from the best standalone component.
    current_blend = None
    current_weights = []  # weights sum to 1 in log space
    current_names = []
    current_ba = 0.0

    best_start = max(standalone, key=lambda n: standalone[n])
    current_names = [best_start]
    current_weights = [1.0]
    current_blend = comps[best_start]["oof"]
    current_ba = standalone[best_start]
    log(f"\n--- greedy forward selection ---")
    log(f"start with best standalone: {best_start}  OOF={current_ba:.5f}")

    alpha_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    MIN_IMPROVEMENT = 1e-5

    while True:
        best_add = None
        best_add_delta = 0.0
        best_add_alpha = 0.0
        for cand in comps:
            if cand in current_names:
                continue
            # Sweep α: new_blend = α × cand + (1-α) × current_blend (log-blend weight renorm).
            # In log-space the α acts as a normalised weight on top of the current blend's composite log-prob.
            best_alpha = 0.0
            best_alpha_ba = current_ba
            for a in alpha_grid:
                cand_blend = log_blend_list(
                    [comps[cand]["oof"], current_blend], [a, 1 - a]
                )
                ba = fixed_bias_bal_acc(cand_blend, y, bias)
                if ba > best_alpha_ba:
                    best_alpha_ba = ba
                    best_alpha = a
            delta = best_alpha_ba - current_ba
            if delta > best_add_delta:
                best_add_delta = delta
                best_add = cand
                best_add_alpha = best_alpha
        if best_add is None or best_add_delta < MIN_IMPROVEMENT:
            log(f"  no candidate improves by >= {MIN_IMPROVEMENT}; stop.")
            break

        # Apply the winning add. Renormalise weights in log-space:
        # new_log = α × log(cand) + (1-α) × current_log
        #         = α × log(cand) + (1-α) × Σ_i w_i × log(model_i)
        new_weights = [(1 - best_add_alpha) * w for w in current_weights] + [best_add_alpha]
        new_names = current_names + [best_add]
        new_blend = log_blend_list(
            [comps[n]["oof"] for n in new_names], new_weights
        )
        current_blend = new_blend
        current_weights = new_weights
        current_names = new_names
        current_ba = fixed_bias_bal_acc(current_blend, y, bias)
        log(f"  + {best_add:25s}  α={best_add_alpha:.3f}  OOF={current_ba:.5f}  "
            f"Δ={best_add_delta:+.5f}")

    log(f"\n--- final greedy blend ---")
    log(f"components & weights (log-space):")
    for n, w in zip(current_names, current_weights):
        log(f"  {w:.4f}  {n}")
    log(f"OOF (fixed digit-XGB bias) = {current_ba:.5f}")
    log(f"vs LB-best 2-way reference  = {lb_best_ba:.5f}   Δ = {current_ba - lb_best_ba:+.5f}")

    # Also tune log-bias on the greedy blend — DIAGNOSTIC ONLY.
    # Submission uses the fixed digit-XGB bias to avoid selection overfit.
    _, tuned_ba = tune_log_bias(current_blend, y, prior)
    log(f"(diagnostic) tuned-bias OOF = {tuned_ba:.5f}")

    # Build test blend.
    test_blend = log_blend_list(
        [comps[n]["test"] for n in current_names], current_weights
    )

    # Confusion matrix at final blend.
    cm = confusion_matrix(
        y, (np.log(np.clip(current_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"\nOOF confusion matrix at final blend:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Error Jaccard vs LB-best reference.
    new_err = (
        np.log(np.clip(current_blend, 1e-9, 1.0)) + bias
    ).argmax(axis=1) != y
    lb_err = (
        np.log(np.clip(lb_best_oof, 1e-9, 1.0)) + bias
    ).argmax(axis=1) != y
    jacc = (new_err & lb_err).sum() / max(1, (new_err | lb_err).sum())
    log(f"\nerror count:  greedy={new_err.sum()}  LB-best={lb_err.sum()}")
    log(f"Jaccard vs LB-best = {jacc:.4f}")

    # Emit submission iff Δ OOF > +1e-4 (looser than +5e-4 since this is
    # a no-retraining selection on pre-computed OOFs).
    action = "no_submission"
    delta_vs_lbbest = current_ba - lb_best_ba
    if delta_vs_lbbest >= 1e-4:
        preds = (np.log(np.clip(test_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub_path = SUB / "submission_greedy_full_bank.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"\nwrote {sub_path}  OOF Δ = {delta_vs_lbbest:+.5f}")
        action = "submission_ready" if delta_vs_lbbest >= 5e-4 else "submission_borderline"
    else:
        log(f"\nno submission: OOF Δ vs LB-best = {delta_vs_lbbest:+.5f} below +1e-4 gate")

    # Dump results.
    out = {
        "anchor_bias": bias.tolist(),
        "standalone_at_fixed_bias": standalone,
        "lb_best_2way_oof": lb_best_ba,
        "greedy": {
            "components": current_names,
            "weights_log_space": current_weights,
            "oof_fixed_bias": current_ba,
            "oof_tuned_bias_diagnostic": tuned_ba,
            "delta_vs_lb_best": delta_vs_lbbest,
            "error_count": int(new_err.sum()),
            "jaccard_vs_lb_best": float(jacc),
        },
        "action": action,
    }
    with open(ART / "greedy_full_bank_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {ART}/greedy_full_bank_results.json")


if __name__ == "__main__":
    main()
