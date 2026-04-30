"""Compute all 10 wild candidates in parallel + diagnostics + LB projections.

Each candidate emits a submission CSV under submissions/. Diagnostic JSON
summarises flip count, direction breakdown, net_H, and projected LB at
multiple precision points. Stability + score-band features used where
relevant.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def csv(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def normed(a: np.ndarray) -> np.ndarray:
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def log_blend(probs_list, weights, eps=1e-9):
    w = weights / weights.sum()
    logits = np.zeros_like(probs_list[0])
    for wi, p in zip(w, probs_list):
        logits += wi * np.log(np.clip(p, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def emit(new_pred: np.ndarray, name: str, test_ids: np.ndarray):
    out_csv = SUB / f"{name}.csv"
    pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    }).to_csv(out_csv, index=False)
    return str(out_csv)


def diag(label, anchor, new_pred, test_ids, baseline_lb=0.98150):
    LMH = ["L", "M", "H"]
    n_flips = int((new_pred != anchor).sum())
    dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((anchor == fr) & (new_pred == to)).sum())
            if n > 0:
                dirs[f"{LMH[fr]}->{LMH[to]}"] = n
    h_a = int(((anchor != 2) & (new_pred == 2)).sum())
    h_r = int(((anchor == 2) & (new_pred != 2)).sum())

    # LB projections at 95% / 88% / 50% precision (uniform)
    n_test = len(anchor)
    # Use B's class counts as N_*_test estimates
    N_L, N_M, N_H = 159460, 100261, 10279
    proj = {}
    for prec in [0.95, 0.88, 0.50]:
        md = 0.0
        for d, n in dirs.items():
            fr, to = d.split("->")
            n_corr = n * prec
            n_wrong = n * (1 - prec)
            # macro contribution: gain on `to`, loss on `fr`
            N_to = {"L": N_L, "M": N_M, "H": N_H}[to]
            N_fr = {"L": N_L, "M": N_M, "H": N_H}[fr]
            md += (n_corr / N_to - n_wrong / N_fr) / 3
        proj[f"prec_{int(prec*100)}"] = round(baseline_lb + md, 6)

    csv_path = emit(new_pred, label, test_ids)

    return {
        "label": label,
        "n_flips": n_flips,
        "directions": dirs,
        "net_h": h_a - h_r,
        "h_added": h_a,
        "h_removed": h_r,
        "lb_proj": proj,
        "csv": csv_path,
    }


def main():
    # Common loads
    test_ids = pd.read_csv("data/test.csv")["id"].to_numpy()

    fb  = csv("submission_idea4b_selective_override")            # LB 0.98150 (current LB-best)
    i5  = csv("submission_idea5_anchor_switch")                  # LB 0.98148
    b   = csv("submission_2other_raw_tier1b_k2")                 # LB 0.98140
    k4  = csv("submission_lbbest_overridden_by_unanimous_others") # LB 0.98134
    v1  = csv("submission_sklearn_rf_meta_natural_standalone_v1_lb98129")  # LB 0.98129
    raw = csv("submission_rawashishsin_2600_standalone")          # LB 0.98109
    t1b = csv("submission_tier1b_greedy_meta")                    # LB 0.98094

    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")

    # dgp_score for test rows
    test_df = pd.read_csv("data/test.csv")
    sm = test_df["Soil_Moisture"].astype(float).to_numpy()
    rf = test_df["Rainfall_mm"].astype(float).to_numpy()
    tc = test_df["Temperature_C"].astype(float).to_numpy()
    ws = test_df["Wind_Speed_kmh"].astype(float).to_numpy()
    dry = (sm < 25).astype(int)
    norain = (rf < 300).astype(int)
    hot = (tc > 30).astype(int)
    windy = (ws > 10).astype(int)
    nomulch = (test_df["Mulching_Used"].astype(str) == "No").astype(int).to_numpy()
    stage = test_df["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, ["Flowering", "Vegetative"]), 2, 0)
    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc

    results = {}

    # W1 — Score-band restriction on 4b: keep 4b's flips ONLY on score=6
    # For other scores, revert to B
    new_pred = b.copy()
    fb_flip = (fb != b)
    keep_mask = fb_flip & (score == 6)
    new_pred[keep_mask] = fb[keep_mask]
    results["W1_score6_only"] = diag("submission_W1_score6_only", b, new_pred, test_ids)

    # W2 — Anti-stability filter on 4b: keep flips only where agreement < 0.85
    new_pred = b.copy()
    keep_mask = fb_flip & (agr < 0.85)
    new_pred[keep_mask] = fb[keep_mask]
    results["W2_anti_stability"] = diag("submission_W2_anti_stability", b, new_pred, test_ids)

    # W3 — Score-stratified plurality
    # For each row, find majority class across {B, 4b, v1, raw, tier1b, k4} per dgp_score bucket
    # Override 4b where per-score majority disagrees
    pool = np.stack([b, fb, v1, raw, t1b, k4], axis=1)
    # Per-row majority
    counts = np.zeros((len(b), 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (pool == c).sum(axis=1)
    plurality = counts.argmax(axis=1)
    # Override 4b with plurality where 4b != plurality
    new_pred = fb.copy()
    flip_mask = plurality != fb
    new_pred[flip_mask] = plurality[flip_mask]
    results["W3_plurality"] = diag("submission_W3_plurality", fb, new_pred, test_ids)

    # W4 — Triple-mechanism consensus override on B
    # Where {4b, Idea5, k4} all agree on class != B
    triple_agree = (fb == i5) & (i5 == k4)
    diff_b = fb != b
    flip_mask = triple_agree & diff_b
    new_pred = b.copy()
    new_pred[flip_mask] = fb[flip_mask]
    results["W4_triple_mech"] = diag("submission_W4_triple_mech", b, new_pred, test_ids)

    # W5 — Idea5's 9 M->H flips applied on top of 4b
    # Find rows where Idea5 says H and 4b says M
    flip_mask = (i5 == 2) & (fb == 1)
    new_pred = fb.copy()
    new_pred[flip_mask] = 2
    results["W5_i5_MtoH_only"] = diag("submission_W5_i5_MtoH_only", fb, new_pred, test_ids)

    # W6 — Reverse ADD-H: where 4b says M and {raw, tier1b} unan H AND 14-bank H
    raw_eq_t1b = (raw == t1b) & (raw == 2)
    bank_h = maj == 2
    flip_mask = (fb == 1) & raw_eq_t1b & bank_h
    new_pred = fb.copy()
    new_pred[flip_mask] = 2
    results["W6_reverse_ADD_H"] = diag("submission_W6_reverse_ADD_H", fb, new_pred, test_ids)

    # W7 — Drop bottom-stability flips from 4b
    fb_flips_idx = np.where(fb != b)[0]
    if len(fb_flips_idx) > 8:
        # Sort by stability ascending, drop bottom 8
        flip_agr = agr[fb_flips_idx]
        bottom_idx = fb_flips_idx[np.argsort(flip_agr)[:8]]
        new_pred = fb.copy()
        new_pred[bottom_idx] = b[bottom_idx]
        results["W7_drop_bottom"] = diag("submission_W7_drop_bottom", fb, new_pred, test_ids)
    else:
        results["W7_drop_bottom"] = {"label": "W7_drop_bottom", "n_flips": 0, "note": "<=8 flips, skipped"}

    # W8 — Pure plurality across 5 LB-validated
    pool5 = np.stack([fb, i5, k4, b, v1], axis=1)
    counts = np.zeros((len(fb), 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (pool5 == c).sum(axis=1)
    new_pred = counts.argmax(axis=1).astype(np.int8)
    # Tie-break: where max count is tied, fall back to 4b
    max_cnt = counts.max(axis=1)
    n_at_max = (counts == max_cnt[:, None]).sum(axis=1)
    new_pred[n_at_max > 1] = fb[n_at_max > 1]
    results["W8_plurality5"] = diag("submission_W8_plurality5", fb, new_pred, test_ids)

    # W9 — LB-weighted soft-vote geomean
    lb_weights = {
        "fb":  (fb,  0.98150),
        "i5":  (i5,  0.98148),
        "k4":  (k4,  0.98134),
        "b":   (b,   0.98140),
        "v1":  (v1,  0.98129),
    }
    n_tr = len(fb)
    # Build soft-probs from argmax + LB weight
    soft_probs = []
    for n, (arr, lb) in lb_weights.items():
        argmax_prob = 0.5 + (lb - 0.97) * 5  # ~0.555-0.55 for LB 0.971-0.982
        other_prob = (1 - argmax_prob) / 2
        sp = np.full((n_tr, 3), other_prob, dtype=np.float32)
        for c in range(3):
            mask = arr == c
            sp[mask, c] = argmax_prob
        soft_probs.append(sp)
    weights = np.array([0.98150, 0.98148, 0.98134, 0.98140, 0.98129])
    weights = weights - 0.97
    bagged = log_blend(soft_probs, weights)
    new_pred = bagged.argmax(axis=1).astype(np.int8)
    results["W9_lb_weighted"] = diag("submission_W9_lb_weighted", fb, new_pred, test_ids)

    # W10 — Score=6 14-bank majority specialist
    new_pred = fb.copy()
    s6_mask = score == 6
    new_pred[s6_mask] = maj[s6_mask]
    results["W10_score6_bank"] = diag("submission_W10_score6_bank", fb, new_pred, test_ids)

    # Sort by best-case projection (95% precision)
    ranked = sorted(results.items(),
                    key=lambda x: x[1].get("lb_proj", {}).get("prec_95", 0),
                    reverse=True)

    print(f"\n{'='*60}")
    print(f"  RANKING by best-case LB projection (95% precision)")
    print(f"{'='*60}")
    for label, d in ranked:
        if "lb_proj" not in d: continue
        flips = d["n_flips"]
        nh = d["net_h"]
        p95 = d["lb_proj"]["prec_95"]
        p88 = d["lb_proj"]["prec_88"]
        p50 = d["lb_proj"]["prec_50"]
        dirs = d["directions"]
        print(f"\n{label}:  flips={flips}  net_H={nh:+d}")
        print(f"  directions: {dirs}")
        print(f"  LB proj: 95%={p95:.5f}  88%={p88:.5f}  50%={p50:.5f}")

    out_json = ART / "wild10_sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== summary written to {out_json} ===")


if __name__ == "__main__":
    main()
