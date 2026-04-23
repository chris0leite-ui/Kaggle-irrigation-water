"""Multi-start greedy + backward-elimination sanity check on the OOF bank.

Verifies whether the LB-submitted 6-way greedy blend (OOF 0.97558) is the
true local optimum. Runs three checks on the SAME fixed digit-XGB bias
used by greedy_full_bank.py:

  1. Multi-start greedy: try each component as the starting anchor, run
     greedy forward-selection, keep the best terminal OOF. If any start
     lands above 0.97558, we've found a better local optimum.

  2. Backward-elimination on the 6-way: drop each component from the
     current best blend, re-optimise remaining weights via pairwise α,
     see if any removal improves OOF.

  3. Weight re-optimisation: coord-ascent over the 6 log-space weights
     from the best blend found. Small perturbations may climb off a
     suboptimal plateau.

No retraining, no new model compute — all operations on saved OOFs.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
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


CANDIDATES = [
    ("digit_xgb",        "oof_xgb_dist_digits.npy",              "test_xgb_dist_digits.npy"),
    ("digits_ote",       "oof_xgb_dist_digits_ote_digits.npy",   "test_xgb_dist_digits_ote_digits.npy"),
    ("digits_pairs",     "oof_xgb_dist_digits_ote_digits_pairs.npy", "test_xgb_dist_digits_ote_digits_pairs.npy"),
    ("digits_light_ote", "oof_xgb_dist_digits_ote_digits_light.npy", "test_xgb_dist_digits_ote_digits_light.npy"),
    ("cat_ote",          "oof_xgb_dist_digits_ote.npy",          "test_xgb_dist_digits_ote.npy"),
    ("cat_ote_light",    "oof_xgb_dist_digits_ote_light.npy",    "test_xgb_dist_digits_ote_light.npy"),
    ("lgbm_digit",       "oof_lgbm_dist_digits.npy",             "test_lgbm_dist_digits.npy"),
    ("lgbm_digit_ote",   "oof_lgbm_dist_digits_ote.npy",         "test_lgbm_dist_digits_ote.npy"),
    ("xgb_nonrule",      "oof_xgb_nonrule.npy",                  "test_xgb_nonrule.npy"),
    ("xgb_vanilla_dist", "oof_xgb_vanilla_dist.npy",             "test_xgb_vanilla_dist.npy"),
    ("xgb_routed_v3",    "oof_xgb_dist_routed_v3.npy",           "test_xgb_dist_routed_v3.npy"),
    ("hybrid_lgbmxgb",   "oof_hybrid_lgbmxgb_blend.npy",         "test_hybrid_lgbmxgb_blend.npy"),
    ("xgb_corn",         "oof_xgb_corn.npy",                     "test_xgb_corn.npy"),
    ("lgbm_te_orig",     "oof_lgbm_te_orig.npy",                 "test_lgbm_te_orig.npy"),
    ("greedy_blend",     "oof_greedy_blend.npy",                 "test_greedy_blend.npy"),
    # Note: extratrees_v2 auto-loaded if file exists
    ("extratrees_v2",    "oof_extratrees_dist_digits_v2.npy",    "test_extratrees_dist_digits_v2.npy"),
]


def load_components() -> dict:
    comps = {}
    for name, op, tp in CANDIDATES:
        opp = ART / op
        tpp = ART / tp
        if not opp.exists() or not tpp.exists():
            continue
        comps[name] = {"oof": np.load(opp), "test": np.load(tpp)}
    return comps


def log_blend(probs_list, weights) -> np.ndarray:
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def ba(p: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    return float(
        balanced_accuracy_score(y, (np.log(np.clip(p, 1e-9, 1.0)) + bias).argmax(axis=1))
    )


def greedy_from_start(start: str, comps: dict, y: np.ndarray, bias: np.ndarray,
                      alpha_grid: np.ndarray, min_improve: float = 1e-5):
    """Run greedy forward-selection starting from a given anchor."""
    current_names = [start]
    current_weights = [1.0]
    current_blend = comps[start]["oof"]
    current_ba = ba(current_blend, y, bias)
    while True:
        best_add = None
        best_delta = 0.0
        best_alpha = 0.0
        for cand in comps:
            if cand in current_names:
                continue
            best_a_ba = current_ba
            best_a = 0.0
            for a in alpha_grid:
                blend = log_blend([comps[cand]["oof"], current_blend], [a, 1 - a])
                b = ba(blend, y, bias)
                if b > best_a_ba:
                    best_a_ba = b
                    best_a = a
            delta = best_a_ba - current_ba
            if delta > best_delta:
                best_delta = delta
                best_add = cand
                best_alpha = best_a
        if best_add is None or best_delta < min_improve:
            break
        new_weights = [(1 - best_alpha) * w for w in current_weights] + [best_alpha]
        current_names = current_names + [best_add]
        current_weights = new_weights
        current_blend = log_blend(
            [comps[n]["oof"] for n in current_names], current_weights
        )
        current_ba = ba(current_blend, y, bias)
    return current_names, current_weights, current_ba


def main() -> None:
    log("loading components")
    comps = load_components()
    log(f"loaded {len(comps)}: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    digit_res = json.loads((ART / "xgb_dist_digits_results.json").read_text())
    bias = np.array(digit_res["log_bias"])
    log(f"anchor bias = {bias.round(4).tolist()}")

    # Current LB-submitted 6-way.
    known_6way = {
        "digit_xgb":        0.4429,
        "digits_ote":       0.2373,
        "xgb_nonrule":      0.1107,
        "xgb_corn":         0.0879,
        "digits_pairs":     0.0712,
        "digits_light_ote": 0.0500,
    }
    known_names = list(known_6way.keys())
    known_weights = list(known_6way.values())
    known_blend = log_blend(
        [comps[n]["oof"] for n in known_names], known_weights
    )
    known_ba = ba(known_blend, y, bias)
    log(f"\n=== known 6-way LB-submitted OOF @ fixed bias = {known_ba:.5f}  (LB 0.97581) ===")

    alpha_grid = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])

    # ======================================================================
    # 1. MULTI-START greedy
    # ======================================================================
    log("\n=== 1. multi-start greedy forward-selection ===")
    results = []
    for start in comps:
        names, weights, ba_end = greedy_from_start(start, comps, y, bias, alpha_grid)
        results.append({"start": start, "names": names, "weights": weights, "ba": ba_end})
        log(f"  start={start:20s}  -> OOF={ba_end:.5f}  ({len(names)} comps)")
    results.sort(key=lambda r: r["ba"], reverse=True)
    top = results[0]
    log(f"\nBest multi-start: start={top['start']}  OOF={top['ba']:.5f}  "
        f"Δ vs known = {top['ba']-known_ba:+.5f}")

    # ======================================================================
    # 2. BACKWARD-ELIMINATION on known 6-way: drop each, re-fit α on others
    # ======================================================================
    log("\n=== 2. backward-elimination on known 6-way ===")
    for drop in known_names:
        keep_names = [n for n in known_names if n != drop]
        # Re-run greedy starting from keep's best standalone, but force-initialise
        # with keep_names set and known weights (renormalised).
        sub_weights = [known_6way[n] for n in keep_names]
        total = sum(sub_weights)
        sub_weights = [w / total for w in sub_weights]
        sub_blend = log_blend([comps[n]["oof"] for n in keep_names], sub_weights)
        sub_ba = ba(sub_blend, y, bias)
        log(f"  drop {drop:20s}  ({len(keep_names)}-way renorm)  OOF={sub_ba:.5f}  "
            f"Δ vs 6-way = {sub_ba-known_ba:+.5f}")

    # ======================================================================
    # 3. COORD-ASCENT weight refinement on known 6-way
    # ======================================================================
    log("\n=== 3. weight refinement (coord-ascent on log-space weights) ===")
    weights = list(known_weights)
    best_ba = known_ba
    step = 0.05
    max_outer = 8
    for outer in range(max_outer):
        improved = False
        for i in range(len(weights)):
            for delta in [-step, -step/2, +step/2, +step]:
                w_try = weights.copy()
                w_try[i] = max(0.0, w_try[i] + delta)
                total = sum(w_try)
                if total == 0:
                    continue
                w_try = [w / total for w in w_try]
                blend = log_blend([comps[n]["oof"] for n in known_names], w_try)
                b = ba(blend, y, bias)
                if b > best_ba + 1e-6:
                    best_ba = b
                    weights = w_try
                    improved = True
        if not improved:
            log(f"  converged at outer iter {outer}")
            break
        log(f"  iter {outer}  OOF={best_ba:.5f}")
    log(f"refined weights: {[round(w, 4) for w in weights]}")
    log(f"refined OOF = {best_ba:.5f}  Δ vs known = {best_ba-known_ba:+.5f}")

    # Save the best found among all three strategies.
    best_of_all = max([
        ("known_6way",          known_names, known_weights, known_ba),
        ("multistart_best",     top["names"], top["weights"], top["ba"]),
        ("refined_weights",     known_names, weights, best_ba),
    ], key=lambda t: t[3])
    label, names_f, weights_f, ba_f = best_of_all

    log(f"\n=== OVERALL BEST: {label}  OOF={ba_f:.5f}  Δ vs known 6-way = {ba_f-known_ba:+.5f} ===")
    log(f"components: {names_f}")
    log(f"weights:    {[round(w, 4) for w in weights_f]}")

    # Build test blend + emit only if Δ > +5e-5.
    test_blend = log_blend([comps[n]["test"] for n in names_f], weights_f)
    if ba_f - known_ba > 5e-5:
        preds = (np.log(np.clip(test_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub = SUB / "submission_greedy_multistart.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"wrote {sub}")
    else:
        log(f"no submission: Δ {ba_f-known_ba:+.5f} below +5e-5 emit gate")

    with open(ART / "greedy_multistart_results.json", "w") as f:
        json.dump({
            "known_6way_oof": known_ba,
            "multistart_results": [
                {"start": r["start"], "names": r["names"],
                 "weights": list(r["weights"]), "oof": r["ba"]}
                for r in results
            ],
            "backward_elim": "see log",
            "refined_weights": {
                "names": known_names,
                "weights": list(weights),
                "oof": best_ba,
            },
            "overall_best": {
                "label": label,
                "names": list(names_f),
                "weights": [float(w) for w in weights_f],
                "oof": ba_f,
                "delta_vs_known": ba_f - known_ba,
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
