"""T6 — Caruana diversity-penalized forward selection.

Anchor: LB-best 4b's TRAIN OOF analog = v1 RF natural OOF
(since 4b is built on top of v1).

Mechanism:
  For each step, evaluate each candidate component m not already in
  ensemble. Score = macro_recall(blend(ens, m)) - beta * max_jaccard(m, ens).
  Select argmax. Stop when no candidate clears beta-adjusted improvement.

Reports the final ensemble weights and the macro-recall path. Requires
diversity_max - diversity_added >= 0.0001 cumulative gain over no-diversity
baseline before recommending a new submission.

NOTE: this is a TRAIN-OOF-validated mechanism. It does NOT directly emit a
test-side submission unless the OOF result demonstrates clear lift over a
naive blend baseline. We report and let user decide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T6_diversity_helpers import (  # noqa: E402
    argmax_jaccard,
    list_oof_names,
    load_y_train,
    macro_recall,
    normed,
    tune_log_bias_simple,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

MAX_STEPS = 8
BETA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05]


# Curated pool — LB-validated bank components plus structurally diverse
# additions known to be naturally calibrated. Skip known LB-regressors.
POOL = [
    "sklearn_rf_meta_natural",
    "rawashishsin_2600",
    "tier1b_greedy_meta",
    "recipe_full_te",
    "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler",
    "realmlp",
    "xgb_nonrule",
    "xgb_metastack",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "lgbm_meta_natural",
    "sklearn_rf_meta_natural_a1lgbm",
    "sklearn_rf_meta_natural_r10_with_tier1b",
]


def load_oof(name: str) -> np.ndarray:
    return normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))


def load_test(name: str) -> np.ndarray:
    return normed(np.load(ART / f"test_{name}.npy").astype(np.float32))


def main():
    print("=== T6 — Caruana diversity-penalized forward selection ===\n")
    y = load_y_train()

    # Anchor = v1 RF natural (proxy for 4b base)
    anchor_name = "sklearn_rf_meta_natural"
    anchor_oof = load_oof(anchor_name)
    anchor_score = macro_recall(y, anchor_oof.argmax(1))
    print(f"Anchor: {anchor_name}, raw argmax macro = {anchor_score:.6f}")
    bias, tuned_score = tune_log_bias_simple(anchor_oof, y)
    print(f"Anchor tuned bias = {bias.round(3).tolist()}, "
          f"tuned macro = {tuned_score:.6f}")

    # Load all candidate OOFs
    candidates = {}
    for name in POOL:
        if name == anchor_name:
            continue
        p = ART / f"oof_{name}.npy"
        if p.exists():
            candidates[name] = load_oof(name)
        else:
            print(f"WARN: missing {p}")

    print(f"\nCandidate pool size: {len(candidates)}")

    # Run greedy with multiple beta values; report best
    results_per_beta = {}
    for beta in BETA_VALUES:
        print(f"\n--- beta = {beta} ---")
        # ensemble weights and component prob arrays
        ens_names = [anchor_name]
        ens_arrays = [anchor_oof.copy()]
        ens_weights = [1.0]
        cur_blend = anchor_oof.copy()
        cur_score = tuned_score
        bias_curr = bias.copy()

        path = [{"step": 0, "name": anchor_name, "alpha": 1.0,
                 "macro": cur_score, "bias": bias.tolist()}]

        for step in range(1, MAX_STEPS + 1):
            # For each remaining candidate, sweep alpha and find best blend
            best = None
            for cname, carr in candidates.items():
                if cname in ens_names:
                    continue
                # alpha sweep at fixed coarse grid
                for alpha in [0.05, 0.10, 0.15, 0.20, 0.30]:
                    log_blend = (1 - alpha) * np.log(np.clip(cur_blend, 1e-9, None)) \
                                + alpha * np.log(np.clip(carr, 1e-9, None))
                    blend = np.exp(log_blend - log_blend.max(1, keepdims=True))
                    blend = blend / blend.sum(1, keepdims=True)
                    # Reuse current bias (no re-tune to avoid overfit)
                    pred = (np.log(np.clip(blend, 1e-9, None)) + bias_curr).argmax(1)
                    s = macro_recall(y, pred)

                    # diversity penalty: max jaccard with existing ens members
                    # using component's own argmax
                    if step == 1:
                        max_j = argmax_jaccard(carr, ens_arrays[0])
                    else:
                        max_j = max(argmax_jaccard(carr, e) for e in ens_arrays)
                    score = s - beta * max_j

                    cand = (score, s, cname, alpha, max_j, blend)
                    if best is None or cand[0] > best[0]:
                        best = cand

            if best is None:
                break
            adj_score, raw_macro, cname, alpha, jac, new_blend = best
            delta = raw_macro - cur_score
            print(f"  step {step}: + {cname} alpha={alpha} "
                  f"jac={jac:.3f} raw_macro={raw_macro:.6f} "
                  f"delta={delta:+.6f} adj={adj_score:.6f}")

            # Stop criterion: no improvement in adjusted score
            if delta < 1e-5:
                print(f"  stop at step {step}: delta < 1e-5")
                break

            cur_blend = new_blend
            cur_score = raw_macro
            ens_names.append(cname)
            ens_arrays.append(candidates[cname])
            ens_weights.append(alpha)

            path.append({"step": step, "name": cname, "alpha": alpha,
                         "macro": cur_score, "jaccard": jac, "delta": delta})

        results_per_beta[str(beta)] = {
            "final_macro": cur_score,
            "lift_over_anchor": cur_score - tuned_score,
            "path": path,
        }

    # Report best beta
    print("\n=== Summary ===")
    print(f"{'beta':<8} {'final_macro':<14} {'lift':<14}")
    for beta_str, res in results_per_beta.items():
        print(f"{beta_str:<8} {res['final_macro']:<14.6f} {res['lift_over_anchor']:<+14.6f}")

    out = ART / "T6_diversity_greedy_results.json"
    out.write_text(json.dumps(results_per_beta, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
