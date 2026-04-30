"""Bank-cleaning + 3 balanced-vote variants on cleaned 3-anchor LB-validated set.

Pipeline:
1. Reconstruct 6 LB-validated submissions as (oof, test) pairs.
2. Pairwise OOF argmax disagreement + error Jaccard matrix to confirm nesting.
3. Drop nested chain subs; keep {PRIMARY, recipe, catboost_iso} (cleaned bank).
4. Run V1/V2/V3 vote variants on cleaned bank.
5. Score each via 4-gate filter against PRIMARY at recipe bias.
6. NO LB probe; report scorecards.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, fast_bal_acc, tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, normed, iso_cal, load_y, log,
)
from balanced_vote_helpers import (  # noqa: E402
    hard_vote_asymmetric, hard_vote_confidence, soft_vote_class_weighted,
    score_predictions, gate_check,
)

OUT = ART / "balanced_vote_explore_results.json"


def L(name: str):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def reconstruct_bank(y: np.ndarray) -> dict:
    """Build 6 LB-validated subs as (oof, test) pairs."""
    log("Loading components ...")
    recipe = L("recipe_full_te")
    pseudo_s1 = L("recipe_pseudolabel")
    pseudo_s7 = L("recipe_pseudolabel_seed7labeler")
    realmlp = L("realmlp")
    nonrule = L("xgb_nonrule")
    metastack = L("xgb_metastack")
    catboost = L("recipe_full_te_catboost")

    log("Reconstructing 6 LB-validated subs ...")
    # 3-way (LB 0.98005)
    w3 = np.array([0.25, 0.35, 0.40])
    m3way_o = log_blend([recipe[0], pseudo_s1[0], pseudo_s7[0]], w3)
    m3way_t = log_blend([recipe[1], pseudo_s1[1], pseudo_s7[1]], w3)
    # 2-way pseudo (LB 0.97998)
    m2way_o = log_blend([recipe[0], pseudo_s1[0]], np.array([0.5, 0.5]))
    m2way_t = log_blend([recipe[1], pseudo_s1[1]], np.array([0.5, 0.5]))
    # LB-best 3-stack (LB 0.98008)
    nr_o, nr_t = iso_cal(nonrule[0], nonrule[1], y)
    s1_o = log_blend([m3way_o, realmlp[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([m3way_t, realmlp[1]], np.array([0.8, 0.2]))
    stack3_o = log_blend([s1_o, nr_o], np.array([0.925, 0.075]))
    stack3_t = log_blend([s1_t, nr_t], np.array([0.925, 0.075]))
    # PRIMARY (LB 0.98094)
    meta_o, meta_t = iso_cal(metastack[0], metastack[1], y)
    primary_o = log_blend([stack3_o, meta_o], np.array([0.7, 0.3]))
    primary_t = log_blend([stack3_t, meta_t], np.array([0.7, 0.3]))
    # CatBoost iso-cal (LB 0.97935)
    cat_o, cat_t = iso_cal(catboost[0], catboost[1], y)

    return {
        "PRIMARY":      (primary_o, primary_t),     # LB 0.98094
        "lb3_realmlp":  (stack3_o, stack3_t),       # LB 0.98008 (subset of PRIMARY)
        "m3way":        (m3way_o, m3way_t),         # LB 0.98005
        "m2way":        (m2way_o, m2way_t),         # LB 0.97998
        "recipe":       recipe,                      # LB 0.97939
        "catboost":     (cat_o, cat_t),             # LB 0.97935 (with iso)
    }


def diagnostic_correlation(bank: dict, y: np.ndarray) -> dict:
    """Pairwise OOF argmax disagreement + error Jaccard. Confirms nesting."""
    names = list(bank.keys())
    preds = {n: (np.log(np.clip(bank[n][0], 1e-12, 1)) + BIAS).argmax(1) for n in names}
    errs = {n: (preds[n] != y) for n in names}

    log("\n  Pairwise OOF argmax disagreement (% rows):")
    print("    " + " ".join(f"{n:>11s}" for n in names))
    disagree = {}
    for a in names:
        row_str = f"  {a:11s}"
        for b in names:
            d = float((preds[a] != preds[b]).mean())
            disagree[f"{a}__{b}"] = d
            row_str += f" {d*100:10.3f}"
        print(row_str)

    log("\n  Pairwise OOF error Jaccard:")
    print("    " + " ".join(f"{n:>11s}" for n in names))
    jaccard = {}
    for a in names:
        row_str = f"  {a:11s}"
        for b in names:
            inter = (errs[a] & errs[b]).sum()
            union = (errs[a] | errs[b]).sum()
            j = float(inter / max(union, 1))
            jaccard[f"{a}__{b}"] = j
            row_str += f" {j:10.4f}"
        print(row_str)

    return {"disagree": disagree, "jaccard": jaccard,
            "bal_acc": {n: float(fast_bal_acc(y, preds[n])) for n in names}}


def run_vote_variants(cleaned: dict, y: np.ndarray, anchor_pred: np.ndarray) -> dict:
    """Run V1/V2/V3 on cleaned bank. Score each vs anchor."""
    probs_oof = [cleaned[k][0] for k in cleaned]
    probs_test = [cleaned[k][1] for k in cleaned]

    log("\n  V1 — rare-class-favoring hard-vote (asymmetric tie-break H>M>L)")
    v1_oof = hard_vote_asymmetric(probs_oof)
    v1_test = hard_vote_asymmetric(probs_test)
    s1 = score_predictions(v1_oof, y, anchor_pred, "V1_hardvote_asym")
    g1 = gate_check(s1)

    log("  V2 — confidence-weighted hard-vote (vote weight = max_prob)")
    v2_oof = hard_vote_confidence(probs_oof)
    v2_test = hard_vote_confidence(probs_test)
    s2 = score_predictions(v2_oof, y, anchor_pred, "V2_hardvote_conf")
    g2 = gate_check(s2)

    log("  V3 — 1/pi_c-weighted soft-vote (raw — no log-bias retune)")
    v3_oof_p = soft_vote_class_weighted(probs_oof)
    v3_test_p = soft_vote_class_weighted(probs_test)
    v3_oof = v3_oof_p.argmax(1)
    v3_test = v3_test_p.argmax(1)
    s3 = score_predictions(v3_oof, y, anchor_pred, "V3_softvote_classw_raw")
    g3 = gate_check(s3)

    log("  V3+ — same as V3 but with coord-ascent log-bias on OOF")
    bias3, _ = tune_log_bias(v3_oof_p, y, prior=np.array([0.5872, 0.3795, 0.0333]))
    v3p_oof = (np.log(np.clip(v3_oof_p, 1e-12, 1)) + bias3).argmax(1)
    v3p_test = (np.log(np.clip(v3_test_p, 1e-12, 1)) + bias3).argmax(1)
    s4 = score_predictions(v3p_oof, y, anchor_pred, "V3plus_softvote_classw_tuned")
    g4 = gate_check(s4)

    return {
        "V1": {"score": s1, "gate": g1, "test_pred": v1_test},
        "V2": {"score": s2, "gate": g2, "test_pred": v2_test},
        "V3": {"score": s3, "gate": g3, "test_pred": v3_test},
        "V3+": {"score": s4, "gate": g4, "test_pred": v3p_test, "bias": bias3.tolist()},
    }


def main():
    log("=== Balanced-vote exploration on cleaned LB-validated bank ===")
    y = load_y()
    bank = reconstruct_bank(y)

    log("\nSanity bal_acc at recipe bias:")
    for n, (oof, _) in bank.items():
        score = fast_bal_acc(y, (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1))
        log(f"  {n:11s} = {score:.5f}")

    log("\n=== STEP 1: nesting diagnostic ===")
    diag = diagnostic_correlation(bank, y)

    log("\n=== STEP 2: cleaned bank = {PRIMARY, recipe, catboost_iso} ===")
    cleaned = {n: bank[n] for n in ["PRIMARY", "recipe", "catboost"]}
    anchor_pred = (np.log(np.clip(bank["PRIMARY"][0], 1e-12, 1)) + BIAS).argmax(1)

    log("\n=== STEP 3: run 3 vote variants ===")
    variants = run_vote_variants(cleaned, y, anchor_pred)

    log("\n=== STEP 4: scorecard ===")
    print(f"  {'variant':32s} {'bal_acc':>9s} {'delta':>9s} {'errs':>7s} {'rec_L':>9s} {'rec_M':>9s} {'rec_H':>9s} {'net_H':>6s} {'asym':>6s}")
    print(f"  {'PRIMARY (anchor)':32s} {fast_bal_acc(y, anchor_pred):9.5f} {0.0:9.5f} {(anchor_pred != y).sum():7d}")
    for v in ["V1", "V2", "V3", "V3+"]:
        s = variants[v]["score"]
        g = variants[v]["gate"]
        gates_str = " ".join("✓" if g[k] else "✗" for k in ("G1", "G2", "G3", "G4"))
        all_str = " ALL-PASS" if g["all_pass"] else ""
        print(f"  {s['label']:32s} {s['bal_acc']:9.5f} {s['delta_bal']:+9.5f} "
              f"{s['errs']:7d} {s['rec_L']:9.5f} {s['rec_M']:9.5f} "
              f"{s['rec_H']:9.5f} {s['net_H']:+6d} {s['asym']:+6.2f}  [{gates_str}]{all_str}")

    log("\n=== STEP 5: persist results ===")
    output = {
        "anchor": "PRIMARY (LB 0.98094)",
        "anchor_bal_acc": float(fast_bal_acc(y, anchor_pred)),
        "anchor_errs": int((anchor_pred != y).sum()),
        "diagnostic": diag,
        "variants": {v: {"score": variants[v]["score"], "gate": variants[v]["gate"]}
                     for v in variants},
    }
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    log(f"  saved {OUT}")

    log("\n=== STEP 6: emit decision ===")
    any_pass = any(variants[v]["gate"]["all_pass"] for v in variants)
    if any_pass:
        log("  At least one variant clears all 4 gates. Submission CSVs NOT auto-emitted")
        log("  per CLAUDE.md rule (always ASK USER first). Inspect scorecard above.")
    else:
        log("  No variant clears all 4 gates. No LB probe warranted.")


if __name__ == "__main__":
    main()
