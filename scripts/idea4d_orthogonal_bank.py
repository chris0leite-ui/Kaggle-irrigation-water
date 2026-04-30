"""Idea 4d — orthogonal-bank consensus override on top of 4b (LB 0.98150).

Lesson from 4b_plus_w5_strict90 (LB 0.98143, -7bp):
  The 14-bank is lineage-correlated with 4b's inputs:
    - 3 RF natural variants ≈ bagged_v1' axis (a)
    - raw + tier1b directly = axis (b)
    - recipe + pseudos ≈ raw lineage
    - realmlp + xgb_nonrule + xgb_metastack ≈ tier1b lineage
  So "14-bank majority confirms" duplicates raw+tier1b vote with correlated
  members. Works in H→M direction (where everyone-except-4b sees M correctly)
  but FAILS in M→L direction (where recipe-family bank is wrong when RF-natural
  is right). Back-out: M→L precision ~25-30% at strict90 filter (vs 61.4%
  break-even).

Fix: replace 14-bank with structurally-orthogonal bank — 4 models with
DIFFERENT objectives/FE, NOT in raw or tier1b lineage:
  - xgb_corn         (Frank-Hall ordinal decomposition, binary-cut obj)
  - recipe_macrorec  (custom focal-on-macro-recall obj)
  - recipe_basemargin_K2 (rule prior anchor at training)
  - recipe_residte   (residual TE, different signal pathway)

Filter: rows where
  (a) 4b's argmax differs from the orthogonal-bank k=3-of-4 majority
  (b) raw + tier1b k=2 unanimous agree with the orthogonal-bank majority
  (c) Direction is favorable break-even
       (M→H, M→L, L→M, L→H — skip H→M and H→L, 4b already handled)

Output: candidate submission, per-direction breakdown, projection.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")

LMH = ["L", "M", "H"]
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}

# Test-side class counts proxy from 4b prediction
N_L, N_M, N_H = 159459, 100367, 10174

BREAK_EVEN = {
    (2, 1): N_M / (N_M + N_H),    # H->M: 0.908
    (2, 0): N_L / (N_L + N_H),    # H->L: 0.940
    (1, 2): N_H / (N_H + N_M),    # M->H: 0.092
    (1, 0): N_L / (N_L + N_M),    # M->L: 0.614
    (0, 2): N_H / (N_H + N_L),    # L->H: 0.060
    (0, 1): N_M / (N_M + N_L),    # L->M: 0.386
}

# Allowed directions: skip H-origin (4b's calibrated 4b H predictions are protected)
ALLOWED = {(1, 2), (1, 0), (0, 1), (0, 2)}


def load_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(LMH_REV).to_numpy(dtype=np.int8)


def load_test_argmax(name: str) -> np.ndarray:
    arr = np.load(ART / f"test_{name}.npy").astype(np.float32)
    arr = arr / np.clip(arr.sum(1, keepdims=True), 1e-9, None)
    return arr.argmax(1).astype(np.int8)


def main():
    print("=== Idea 4d: orthogonal-bank consensus override on 4b (LB 0.98150) ===\n")

    fb = load_argmax("submission_idea4b_selective_override")
    raw = load_argmax("submission_rawashishsin_2600_standalone")
    tier1b = load_argmax("submission_tier1b_greedy_meta")

    n_test = len(fb)
    print(f"Anchor 4b class counts: L={int((fb==0).sum())} M={int((fb==1).sum())} H={int((fb==2).sum())}")

    # Orthogonal bank: 4 models with structurally-independent training paths
    bank_names = [
        "xgb_corn",
        "recipe_full_te_macrorec_T1_lam03",
        "recipe_full_te_basemargin_K2",
        "recipe_full_te_residte",
    ]
    bank_argmaxes = []
    for n in bank_names:
        am = load_test_argmax(n)
        bank_argmaxes.append(am)
        print(f"  loaded orthogonal-bank: {n}  argmax dist L/M/H = "
              f"{int((am==0).sum())}/{int((am==1).sum())}/{int((am==2).sum())}")
    bank_arr = np.stack(bank_argmaxes, axis=1)  # (n_test, 4)

    # Per-row k-of-4 majority
    counts = np.zeros((n_test, 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (bank_arr == c).sum(axis=1)
    bank_maj = counts.argmax(axis=1)
    bank_max = counts.max(axis=1)

    # Statistics
    n_unanimous = int((bank_max == 4).sum())
    n_3of4 = int((bank_max == 3).sum())
    n_2of4 = int((bank_max == 2).sum())
    print(f"\nOrthogonal-bank consensus distribution:")
    print(f"  k=4 unanimous: {n_unanimous} ({n_unanimous/n_test*100:.1f}%)")
    print(f"  k=3 majority:  {n_3of4} ({n_3of4/n_test*100:.1f}%)")
    print(f"  k=2 plurality: {n_2of4} ({n_2of4/n_test*100:.1f}%)")
    print()

    # Filter (a): 4b argmax differs from orthogonal-bank majority
    diff_mask = bank_maj != fb
    # Filter (b): raw + tier1b unanimous, agreeing with orthogonal-bank majority
    rt_unan = (raw == tier1b) & (raw == bank_maj)
    # Filter strength: strict (k>=4) vs loose (k>=3)
    strict_mask = bank_max >= 4
    loose_mask = bank_max >= 3

    # Direction-restrict
    direction_ok = np.zeros(n_test, dtype=bool)
    for fr, to in ALLOWED:
        direction_ok |= ((fb == fr) & (bank_maj == to))

    filters = {
        "k4_unan_RTunan_dirOK": diff_mask & rt_unan & strict_mask & direction_ok,
        "k3_maj_RTunan_dirOK":  diff_mask & rt_unan & loose_mask & direction_ok,
        "k4_unan_RTunan_anydir": diff_mask & rt_unan & strict_mask,
        "k3_maj_RTunan_anydir":  diff_mask & rt_unan & loose_mask,
        # Also without RT-unan constraint (relax to k-of-4 alone)
        "k4_unan_dirOK":         diff_mask & strict_mask & direction_ok,
    }

    print("Filter candidate matrix:")
    print()
    results = {}
    for fname, mask in filters.items():
        n_flips = int(mask.sum())
        if n_flips == 0:
            print(f"  {fname:35s} 0 flips — saturated")
            results[fname] = {"n_flips": 0}
            continue
        # Direction breakdown
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                n = int(((fb == fr) & (bank_maj == to) & mask).sum())
                if n > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = n
        # Project at multiple precision levels — break-even-aware
        proj = {}
        for p in [0.30, 0.50, 0.65, 0.80]:
            md = 0.0
            for d, n in dirs.items():
                fr, to = d.split("->")
                fr_i, to_i = LMH.index(fr), LMH.index(to)
                Ns = [N_L, N_M, N_H]
                md += n * (p / Ns[to_i] - (1 - p) / Ns[fr_i]) / 3
            proj[p] = round(0.98150 + md, 5)
        # Asymmetry summary
        net_h = int((dirs.get("L->H", 0) + dirs.get("M->H", 0))
                    - (dirs.get("H->L", 0) + dirs.get("H->M", 0)))

        results[fname] = {
            "n_flips": n_flips,
            "dirs": dirs,
            "net_h": net_h,
            "proj@0.30": proj[0.30],
            "proj@0.50": proj[0.50],
            "proj@0.65": proj[0.65],
            "proj@0.80": proj[0.80],
        }
        print(f"  {fname:35s} n={n_flips:3d}  dirs={dirs}  net_H={net_h:+d}")
        print(f"    proj LB:  @30%={proj[0.30]}  @50%={proj[0.50]}  "
              f"@65%={proj[0.65]}  @80%={proj[0.80]}")
        print()

    # Emit the strongest direction-restricted variant
    test_ids = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")["id"].tolist()
    candidates_to_emit = []

    for fname in ["k4_unan_RTunan_dirOK", "k3_maj_RTunan_dirOK"]:
        info = results[fname]
        if info.get("n_flips", 0) == 0:
            continue
        # Build the override
        mask = filters[fname]
        new_pred = fb.copy()
        new_pred[mask] = bank_maj[mask]
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
        })
        out = SUB / f"submission_idea4d_{fname}.csv"
        sub.to_csv(out, index=False)
        candidates_to_emit.append((str(out.name), info))
        print(f"emitted: {out.name}  flips={info['n_flips']}")

    # Save results JSON
    (ART / "idea4d_results.json").write_text(json.dumps({
        "anchor": "4b LB 0.98150",
        "bank_names": bank_names,
        "filters": {k: {kk: vv for kk, vv in v.items() if kk != "mask"}
                    for k, v in results.items()},
        "emitted": [c[0] for c in candidates_to_emit],
    }, indent=2))

    print()
    print("=== Recommendation ===")
    if not candidates_to_emit:
        print("All filters returned 0 flips — orthogonal-bank consensus saturated on 4b.")
        return
    best = sorted(candidates_to_emit,
                  key=lambda c: (c[1].get("proj@0.50", 0)), reverse=True)[0]
    print(f"Highest-EV candidate: {best[0]}")
    print(f"  flips: {best[1]['n_flips']}, net_H: {best[1]['net_h']:+d}")
    print(f"  proj LB: @50%={best[1]['proj@0.50']}, @65%={best[1]['proj@0.65']}")


if __name__ == "__main__":
    main()
