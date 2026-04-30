"""L5b — rank focal-disagreer flips by p_bank(new_class) and check
precision at top-K cuts. Decisive test of "rank by bank confidence
in the proposed flip class."

If top-50 / top-100 of focal-disagreer flips have ≥92% H->M precision
on OOF, an analogous test-side top-K extends 4b productively.
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
    return np.stack([
        _norm(np.load(ART / f"{side}_{n}.npy").astype(np.float32))
        for n in names
    ], axis=0)


def _csv_argmax(name):
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLS).to_numpy(dtype=np.int8)


def main():
    print("=== L5b: rank focal-disagreer flips by p_bank(new) ===\n")

    y = pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])[
        "Irrigation_Need"].map(CLS).to_numpy().astype(np.int8)

    bank_oof_p  = _norm(_stack_probs(BANK_NAMES, "oof").mean(0))
    bank_test_p = _norm(_stack_probs(BANK_NAMES, "test").mean(0))
    focal_oof_argmax  = _norm(_stack_probs(FOCAL_NAMES, "oof").mean(0)).argmax(1)
    focal_test_argmax = _norm(_stack_probs(FOCAL_NAMES, "test").mean(0)).argmax(1)

    raw_oof = _norm(np.load(ART / "oof_recipe_full_te.npy")).argmax(1)
    t1b_oof = _norm(np.load(ART / "oof_tier1b_greedy_meta.npy")).argmax(1)
    raw_te  = _norm(np.load(ART / "test_recipe_full_te.npy")).argmax(1)
    t1b_te  = _norm(np.load(ART / "test_tier1b_greedy_meta.npy")).argmax(1)

    bank_maj_oof = _stack_probs(BANK_NAMES, "oof").mean(0).argmax(1)
    bank_maj_te  = _stack_probs(BANK_NAMES, "test").mean(0).argmax(1)

    anchor_oof = _norm(np.load(ART / "oof_rawashishsin_2600.npy")).argmax(1)

    # ---- OOF: focal-maj disagreer vs rawashishsin anchor, 4-way consensus ----
    flip_oof = (anchor_oof != focal_oof_argmax) \
        & (raw_oof == focal_oof_argmax) \
        & (t1b_oof == focal_oof_argmax) \
        & (bank_maj_oof == focal_oof_argmax)
    idx_oof = np.where(flip_oof)[0]
    n_oof = len(idx_oof)
    new_class_oof = focal_oof_argmax[idx_oof]
    old_class_oof = anchor_oof[idx_oof]
    p_new_oof = bank_oof_p[idx_oof, new_class_oof]
    correct_oof = (new_class_oof == y[idx_oof])
    print(f"OOF flip set (focal_maj disagreer): n={n_oof}  "
          f"baseline_prec={correct_oof.mean():.5f}")

    # Sort by p_new descending, walk through quantile cuts
    order = np.argsort(-p_new_oof)
    Ks = [50, 100, 200, 300, 500, 800, 1283][-7:]
    print(f"\nOOF precision at top-K (sorted by p_bank(new_class) desc):")
    rows = []
    for K in Ks:
        sub = order[:K]
        prec = correct_oof[sub].mean()
        # Per-direction
        dir_prec = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                mm = (old_class_oof[sub] == fr) & (new_class_oof[sub] == to)
                if mm.sum() < 5:
                    continue
                p = correct_oof[sub][mm].mean()
                dir_prec[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = (
                    int(mm.sum()), float(p))
        print(f"  K={K:5d}  prec={prec:.5f}  "
              f"min p_new in cut={p_new_oof[sub].min():.4f}")
        for fr_to, (n_, p_) in dir_prec.items():
            be = {"H->M": 0.909, "L->M": 0.612, "M->L": 0.387,
                  "M->H": 0.909}.get(fr_to, 0.5)
            mark = "✓" if p_ >= be else "✗"
            print(f"    {fr_to}: n={n_:4d}  prec={p_:.4f}  BE={be} {mark}")
        rows.append({
            "K": K, "precision": float(prec),
            "min_p_new": float(p_new_oof[sub].min()),
            "per_direction_precision": dir_prec,
        })

    # ---- TEST: same mechanism, 4-way consensus, focal_maj as disagreer ----
    b_test = _csv_argmax("submission_2other_raw_tier1b_k2")
    b_4b   = _csv_argmax("submission_idea4b_selective_override")
    flip_te = (b_test != focal_test_argmax) \
        & (raw_te == focal_test_argmax) \
        & (t1b_te == focal_test_argmax) \
        & (bank_maj_te == focal_test_argmax)
    idx_te = np.where(flip_te)[0]
    n_te = len(idx_te)
    new_class_te = focal_test_argmax[idx_te]
    p_new_te = bank_test_p[idx_te, new_class_te]
    print(f"\nTEST flip set (focal_maj disagreer): n={n_te}")
    order_te = np.argsort(-p_new_te)

    # Find the test K that matches the OOF top-K's MIN p_new threshold —
    # i.e., apply same probability cutoff on test
    print(f"\nTEST top-K cuts by same p_new threshold as OOF top-K:")
    test_kept_K = {}
    flip_4b_set = set(np.where(b_test != b_4b)[0].tolist())
    for r in rows:
        K_oof = r["K"]
        thr = r["min_p_new"]
        keep_te = p_new_te >= thr
        idx_keep = idx_te[keep_te]
        # Overlap with 4b's 108 flips
        overlap_4b = sum(1 for i in idx_keep if i in flip_4b_set)
        new_vs_4b = len(idx_keep) - overlap_4b
        # Direction breakdown of kept flips
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                mm = (b_test[idx_keep] == fr) & (new_class_te[keep_te] == to)
                if mm.sum() > 0:
                    dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(mm.sum())
        print(f"  OOF K={K_oof}, thr={thr:.4f}  "
              f"test n={len(idx_keep)}  "
              f"overlap_w_4b={overlap_4b}/108  "
              f"new={new_vs_4b}  dirs={dirs}")
        test_kept_K[K_oof] = {
            "p_threshold": thr, "n_test_kept": len(idx_keep),
            "overlap_with_4b_flips": overlap_4b,
            "new_vs_4b": new_vs_4b, "directions": dirs,
        }

    # ---- Build candidate: 4b + top-K conformal additions (K with passing precision) ----
    print(f"\nLB projection per K (anchor=B, applying flips at OOF-prec):")
    test_ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    candidates = {}
    for r in rows:
        K_oof = r["K"]
        thr = r["min_p_new"]
        keep_te = p_new_te >= thr
        idx_keep = idx_te[keep_te]
        # Project LB delta vs B (0.98140) using OOF directional precision
        delta = 0.0
        for fr_to, (n_, p_) in r["per_direction_precision"].items():
            n_dir_te = sum(1 for i_idx, i in enumerate(idx_keep)
                           if (b_test[i] == CLS[{"L":"Low","M":"Medium","H":"High"}[fr_to[0]]])
                           and (new_class_te[keep_te][i_idx] == CLS[{"L":"Low","M":"Medium","H":"High"}[fr_to[3]]]))
            if n_dir_te == 0:
                continue
            corr = p_ * n_dir_te
            wrong = (1 - p_) * n_dir_te
            N_M, N_L, N_H = 100261, 158730, 10279  # rough test class sizes
            if fr_to == "H->M":
                d = (corr / N_M - wrong / N_H) / 3
            elif fr_to == "M->L":
                d = (corr / N_L - wrong / N_M) / 3
            elif fr_to == "L->M":
                d = (corr / N_M - wrong / N_L) / 3
            else:
                d = 0  # other directions rare
            delta += d
        proj_lb = 0.98140 + delta
        print(f"  K={K_oof:5d}  test_n={len(idx_keep):5d}  "
              f"projected LB={proj_lb:.5f}  Δ={delta:+.5f}")

        # Emit submission for K=50, 100, 200 only
        if K_oof in (50, 100, 200) and len(idx_keep) > 0:
            new_pred = b_test.copy()
            new_pred[idx_keep] = new_class_te[keep_te]
            out_csv = SUB / f"submission_L5b_focal_topK{K_oof}.csv"
            pd.DataFrame({
                "id": test_ids,
                "Irrigation_Need": pd.Series(new_pred).map(
                    {0: "Low", 1: "Medium", 2: "High"}),
            }).to_csv(out_csv, index=False)
            candidates[f"K{K_oof}"] = {"path": str(out_csv),
                                       "n_flips": len(idx_keep),
                                       "projected_lb": proj_lb}
            print(f"    → emitted {out_csv}")

    out = {
        "oof_n_total_flips": n_oof,
        "test_n_total_flips": n_te,
        "by_K": rows,
        "test_kept_per_K": test_kept_K,
        "candidates": candidates,
    }
    (ART / "L5b_topk_pnew_ranking.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n→ wrote scripts/artifacts/L5b_topk_pnew_ranking.json")


if __name__ == "__main__":
    main()
