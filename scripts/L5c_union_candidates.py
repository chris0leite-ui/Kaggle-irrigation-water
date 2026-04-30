"""L5c — build union candidates: 4b's 108 flips ∪ L5b top-K NEW flips.

L5b found that top-K of focal-disagreer (sorted by p_bank(new)) achieves
high OOF precision but only overlaps 41-94 of 4b's 108 flips. 4b's
remaining flips were precision-selected by bagged_v1' under a DIFFERENT
mechanism (not bank-confidence). The hypothesis: combining both
selection mechanisms keeps 4b's strong flips AND adds bank-confident
focal-disagreer flips.

Output: submission_L5c_4bUnionK{K}.csv for K ∈ {50, 100, 200}.
Projected LB per K combines:
  - 4b's contribution = +0.00010 (empirical, B → 4b)
  - L5b NEW flips at OOF precision per K (only the K-flips not in 4b)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
CLS = {"Low": 0, "Medium": 1, "High": 2}

BANK_NAMES = [
    "sklearn_rf_meta_natural", "sklearn_rf_meta_natural_a1lgbm",
    "sklearn_rf_meta_natural_r10_with_tier1b", "rawashishsin_2600",
    "tier1b_greedy_meta", "recipe_full_te", "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler", "realmlp", "xgb_nonrule",
    "xgb_metastack", "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost", "lgbm_meta_natural",
]
FOCAL_NAMES = ["recipe_focal_g2h3", "recipe_focal_g2_aH1",
               "recipe_focal_g2_invfreq", "recipe_focal_effnum"]


def _norm(p, eps=1e-9):
    return p / np.clip(p.sum(axis=1, keepdims=True), eps, None)


def _stack_probs(names, side):
    return np.stack([_norm(np.load(ART / f"{side}_{n}.npy").astype(np.float32))
                     for n in names], axis=0)


def _csv_argmax(name):
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLS).to_numpy(dtype=np.int8)


def main():
    print("=== L5c: union(4b, L5b top-K NEW) candidates ===\n")

    bank_test_p = _norm(_stack_probs(BANK_NAMES, "test").mean(0))
    focal_test_argmax = _norm(_stack_probs(FOCAL_NAMES, "test").mean(0)).argmax(1)

    raw_te = _norm(np.load(ART / "test_recipe_full_te.npy")).argmax(1)
    t1b_te = _norm(np.load(ART / "test_tier1b_greedy_meta.npy")).argmax(1)
    bank_maj_te = _stack_probs(BANK_NAMES, "test").mean(0).argmax(1)

    b_test = _csv_argmax("submission_2other_raw_tier1b_k2")           # B (LB 0.98140)
    b_4b   = _csv_argmax("submission_idea4b_selective_override")       # 4b (LB 0.98150)

    # Test focal-disagreer flip set
    flip_te = (b_test != focal_test_argmax) \
        & (raw_te == focal_test_argmax) \
        & (t1b_te == focal_test_argmax) \
        & (bank_maj_te == focal_test_argmax)
    idx_te = np.where(flip_te)[0]
    new_class_te = focal_test_argmax[idx_te]
    p_new_te = bank_test_p[idx_te, new_class_te]

    # 4b's flip mask
    flip_4b = b_test != b_4b

    # Per-K thresholds from L5b OOF results (already validated above BE)
    # K=50 thr=0.8289, K=100 thr=0.7995, K=200 thr=0.7577
    thresholds = {
        50:  (0.8289, 0.9592),   # (p_thr, OOF_H_to_M_precision)
        100: (0.7995, 0.9381),
        200: (0.7577, 0.9101),
    }

    test_ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    candidates = {}

    print("union (4b ∪ L5b_new_at_threshold) projections:\n")
    for K, (thr, prec_oof) in thresholds.items():
        keep_te = p_new_te >= thr
        idx_keep = idx_te[keep_te]
        # Of the kept, NEW means not already in 4b's flip set
        new_only = np.array([not flip_4b[i] for i in idx_keep])
        new_idx = idx_keep[new_only]
        new_class_for_idx = new_class_te[keep_te][new_only]
        n_new = len(new_idx)

        # Build union submission
        new_pred = b_4b.copy()
        new_pred[new_idx] = new_class_for_idx

        # Direction breakdown of the NEW additions (vs B since b_4b = B for non-4b-flips)
        dirs_new = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                # NEW means: B[i] = old_class (not flipped by 4b), now flipped by L5c
                mm = (b_test[new_idx] == fr) & (new_class_for_idx == to)
                if mm.sum() > 0:
                    dirs_new[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(mm.sum())

        # Project LB delta vs 4b (NOT vs B): only the NEW flips contribute
        # (4b is already at LB 0.98150)
        delta_vs_4b = 0.0
        N_M, N_L, N_H = 100261, 158730, 10279
        for fr_to, n_dir_te in dirs_new.items():
            corr = prec_oof * n_dir_te
            wrong = (1 - prec_oof) * n_dir_te
            if fr_to == "H->M":
                d = (corr / N_M - wrong / N_H) / 3
            elif fr_to == "M->L":
                d = (corr / N_L - wrong / N_M) / 3
            elif fr_to == "L->M":
                d = (corr / N_M - wrong / N_L) / 3
            else:
                d = 0
            delta_vs_4b += d
        proj_lb = 0.98150 + delta_vs_4b

        out_csv = SUB / f"submission_L5c_4bUnionK{K}.csv"
        pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(
                {0: "Low", 1: "Medium", 2: "High"}),
        }).to_csv(out_csv, index=False)

        print(f"K={K:3d} (thr={thr:.4f}, OOF H→M prec={prec_oof:.4f})")
        print(f"  total flips = {int(flip_4b.sum()) + n_new} "
              f"({int(flip_4b.sum())} from 4b + {n_new} new)")
        print(f"  new-flip dirs = {dirs_new}")
        print(f"  projected LB = {proj_lb:.5f}  (Δ vs 4b = {delta_vs_4b:+.5f})")
        print(f"  → {out_csv}\n")

        candidates[f"K{K}"] = {
            "path": str(out_csv),
            "p_threshold": thr,
            "oof_h_to_m_precision": prec_oof,
            "n_new_flips": n_new,
            "n_total_flips": int(flip_4b.sum()) + n_new,
            "new_directions": dirs_new,
            "projected_lb": proj_lb,
            "delta_vs_4b": delta_vs_4b,
        }

    # Save
    (ART / "L5c_union_candidates.json").write_text(
        json.dumps({"thresholds": {str(k): list(v) for k, v in thresholds.items()},
                    "candidates": candidates}, indent=2, default=str))
    print(f"→ wrote scripts/artifacts/L5c_union_candidates.json")


if __name__ == "__main__":
    main()
