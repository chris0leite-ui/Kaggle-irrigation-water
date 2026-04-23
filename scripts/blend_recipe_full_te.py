"""Fixed-bias log-blend sweep: recipe_full_te × each architecturally-distinct
component on disk.

Purpose: test whether the new LB-best pipeline (recipe_full_te, OOF 0.97967 /
LB 0.97939) can be lifted by blending with components whose decision-surface
is structurally distinct — xgb_nonrule (13 non-rule features only), xgb_corn
(Frank-Hall ordinal decomposition), hybrid_lgbmxgb_blend (model-family
diversity anchor), digit_xgb (narrower FE on same digit lever), and
greedy_blend (specialist + routed XGB stack).

Protocol (from greedy_full_bank.py + CLAUDE.md binhigh lesson):
  - Anchor bias = recipe_full_te's tuned log-bias (LB-calibrated at 0.97939).
  - For each candidate, sweep α ∈ {0.025 .. 0.50} of
        log P_blend = α × log P_cand + (1-α) × log P_recipe
    and evaluate bal_acc under the FIXED anchor bias.
  - Report per-candidate peak OOF + Δ vs recipe OOF 0.97967.
  - Also report diagnostic tuned-bias OOF at peak α (upper bound; do NOT
    submit this, bias retune manufactures OOF lift that doesn't transfer).
  - Emit submission for the best candidate ONLY if fixed-bias Δ ≥ +1e-4.

Notes on scale mismatch: recipe_full_te trains with sample_weight="balanced"
(probs are already class-balanced at training time), so its bias is large
on Low/Medium [1.43, 1.47, 3.40] to correct back. Other candidates don't
train that way, so their probs are sharper on the majority class. Log-blend
at small α (0.05-0.20) mostly preserves recipe's scale; at large α the
scale mismatch becomes the dominant effect.
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


# Architecturally-distinct candidates vs recipe_full_te. Skip:
#   - xgb_bin_high / hybrid_binhigh: known-overfit (-0.00084 LB previously)
#   - xgb_spec_678: domain-restricted, not global probabilities
#   - lb_best_fs7 / fs123: fold-seed bag components (regressed LB -0.00055)
#   - tabpfn: weak standalone (0.962), already blend-null on prior baselines
CANDIDATES = [
    # Non-rule-features-only XGB — blends cleanly on every prior base.
    ("xgb_nonrule",       "oof_xgb_nonrule.npy",                "test_xgb_nonrule.npy"),
    # Frank-Hall ordinal decomposition (binary head per cut).
    ("xgb_corn",          "oof_xgb_corn.npy",                   "test_xgb_corn.npy"),
    # LGBM×XGB blend (model-family diversity, older pipeline).
    ("hybrid_lgbmxgb",    "oof_hybrid_lgbmxgb_blend.npy",       "test_hybrid_lgbmxgb_blend.npy"),
    # Digit-XGB standalone — overlaps with recipe's digit features but
    # plain XGB HPs (no balanced weights, no wide OTE). LB 0.97468.
    ("digit_xgb",         "oof_xgb_dist_digits.npy",            "test_xgb_dist_digits.npy"),
    # Prior-generation 3-way greedy: hybrid_v3 + routed_v3 + spec_678.
    ("greedy_blend",      "oof_greedy_blend.npy",               "test_greedy_blend.npy"),
    # Routed-XGB alone (main of hybrid_v3).
    ("xgb_routed_v3",     "oof_xgb_dist_routed_v3.npy",         "test_xgb_dist_routed_v3.npy"),
    # Vanilla XGB on 43-feature dist (baseline in the XGB-dist family).
    ("xgb_vanilla_dist",  "oof_xgb_vanilla_dist.npy",           "test_xgb_vanilla_dist.npy"),
    # LGBM with original-TE (different regularization profile).
    ("lgbm_te_orig",      "oof_lgbm_te_orig.npy",               "test_lgbm_te_orig.npy"),
    # Digit-OTE-on-default on dist+digits feature set.
    ("digit_ote_default", "oof_xgb_dist_digits_ote.npy",        "test_xgb_dist_digits_ote.npy"),
    # LGBM-digits (tree-family diversity on digit FE).
    ("lgbm_digit",        "oof_lgbm_dist_digits.npy",           "test_lgbm_dist_digits.npy"),
]


def load_components():
    comps = {}
    for name, oof_name, test_name in CANDIDATES:
        op = ART / oof_name
        tp = ART / test_name
        if not op.exists() or not tp.exists():
            log(f"  skip {name}: missing {op.name if not op.exists() else tp.name}")
            continue
        comps[name] = {"oof": np.load(op), "test": np.load(tp)}
    return comps


def log_blend_list(probs_list, weights) -> np.ndarray:
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
    log("loading recipe_full_te anchor")
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    anchor_bias = np.array(recipe_res["log_bias"])
    anchor_oof = np.load(ART / "oof_recipe_full_te.npy")
    anchor_test = np.load(ART / "test_recipe_full_te.npy")
    log(f"recipe anchor bias = {anchor_bias.round(4).tolist()}")
    log(f"recipe anchor OOF  = {recipe_res['tuned_log_bias_bal_acc']:.5f}")

    log("loading components")
    comps = load_components()
    log(f"loaded {len(comps)} components: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Sanity: recipe standalone at its own tuned bias.
    anchor_ba = fixed_bias_bal_acc(anchor_oof, y, anchor_bias)
    log(f"recipe reproduced OOF at its bias = {anchor_ba:.5f}")

    # For each candidate, sweep α.
    alpha_grid = np.array(
        [0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    )

    per_candidate = {}
    log(f"\n--- fixed-bias log-blend sweep vs recipe_full_te (anchor OOF {anchor_ba:.5f}) ---")
    for name, v in comps.items():
        # Standalone candidate at recipe's bias (sanity — usually off-calibration).
        cand_solo_ba = fixed_bias_bal_acc(v["oof"], y, anchor_bias)

        sweep = []
        for a in alpha_grid:
            blend = log_blend_list([v["oof"], anchor_oof], [a, 1 - a])
            ba = fixed_bias_bal_acc(blend, y, anchor_bias)
            sweep.append((float(a), ba))

        best_idx = int(np.argmax([s[1] for s in sweep]))
        peak_alpha, peak_ba = sweep[best_idx]
        delta = peak_ba - anchor_ba

        # Diagnostic: tuned-bias on the peak blend (upper bound, NOT for submission).
        peak_blend = log_blend_list(
            [v["oof"], anchor_oof], [peak_alpha, 1 - peak_alpha]
        )
        _, tuned_ba_diag = tune_log_bias(peak_blend, y, prior)

        log(
            f"  {name:22s}  solo@anchor_bias={cand_solo_ba:.5f}  "
            f"peak α={peak_alpha:.3f}  OOF={peak_ba:.5f}  "
            f"Δ={delta:+.5f}  (diag tuned={tuned_ba_diag:.5f})"
        )

        per_candidate[name] = {
            "candidate_oof_at_anchor_bias": cand_solo_ba,
            "sweep": sweep,
            "peak_alpha": peak_alpha,
            "peak_oof_fixed_bias": peak_ba,
            "delta_vs_anchor": delta,
            "peak_tuned_bias_diagnostic": tuned_ba_diag,
        }

    # Identify best candidate by fixed-bias Δ.
    best_name = max(per_candidate, key=lambda n: per_candidate[n]["delta_vs_anchor"])
    best = per_candidate[best_name]
    log(
        f"\nbest single-component add: {best_name}  "
        f"α={best['peak_alpha']:.3f}  "
        f"OOF={best['peak_oof_fixed_bias']:.5f}  "
        f"Δ={best['delta_vs_anchor']:+.5f}"
    )

    # Emit submission iff fixed-bias Δ >= +1e-4 (looser than +5e-4 since
    # this is a saved-OOF selection, not a retrained model).
    action = "no_submission"
    sub_path = None
    if best["delta_vs_anchor"] >= 1e-4:
        a = best["peak_alpha"]
        test_blend = log_blend_list(
            [comps[best_name]["test"], anchor_test], [a, 1 - a]
        )
        preds = (np.log(np.clip(test_blend, 1e-9, 1.0)) + anchor_bias).argmax(axis=1)
        sub_path = SUB / f"submission_recipe_full_te_plus_{best_name}.csv"
        pd.DataFrame(
            {ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}
        ).to_csv(sub_path, index=False)
        log(f"\nwrote {sub_path}  OOF Δ = {best['delta_vs_anchor']:+.5f}")

        # Diagnostic confusion matrix on the emit candidate.
        cm = confusion_matrix(
            y,
            (np.log(np.clip(
                log_blend_list([comps[best_name]["oof"], anchor_oof], [a, 1 - a]),
                1e-9, 1.0,
            )) + anchor_bias).argmax(axis=1),
        )
        log(
            f"OOF confusion matrix at emit blend:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}"
        )
        action = (
            "submission_ready" if best["delta_vs_anchor"] >= 5e-4
            else "submission_borderline"
        )
    else:
        log(
            f"\nno submission: best Δ vs recipe = {best['delta_vs_anchor']:+.5f} "
            f"below +1e-4 gate"
        )

    out = {
        "anchor": "recipe_full_te",
        "anchor_bias": anchor_bias.tolist(),
        "anchor_oof": anchor_ba,
        "per_candidate": per_candidate,
        "best_candidate": best_name,
        "best_alpha": best["peak_alpha"],
        "best_delta": best["delta_vs_anchor"],
        "action": action,
        "submission": str(sub_path) if sub_path else None,
    }
    with open(ART / "blend_recipe_full_te_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {ART}/blend_recipe_full_te_results.json")


if __name__ == "__main__":
    main()
