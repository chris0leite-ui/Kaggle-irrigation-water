"""L5 — split-conformal shrinkage of 4b's 108 flips.

Hypothesis (orthogonal to L1-L4): some of 4b's 108 flips have weak
prediction intervals (PI) under bank-mean probabilities and should be
dropped, narrowing the override to a high-PI-confidence subset.

If PI-confidence ranking correlates with empirical precision on OOF,
the filter is real signal → apply to TEST-side 4b flips.

Mechanism:
  1. OOF: compute 14-bank-mean probs + per-row nonconformity
     s_i = 1 - p_bank(y_i | x_i).
  2. q_α = (1 - α) quantile of {s_i} across all OOF rows.
  3. PI(x) = {c : 1 - p_bank(c | x) ≤ q_α}.
  4. For each 4b flip on TEST: B is "old class", 4b is "new class".
     - DROP the flip if B is still in PI at level α (PI says B might be right).
     - KEEP the flip if B is excluded from PI.

OOF VALIDATION: simulate 4b-style flips on OOF using focal_majority
as the disagreer (bagged_v1' OOF is unavailable; focal_majority gives
a similar 4-way-consensus flip set). Measure precision of:
   {filtered flips: B excluded from PI at level α}
   vs
   {all flips}
across α grid. If filter improves precision materially, transfer to
TEST. If precision is flat across α, conformal-PI is uninformative
and the filter cannot help.
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

ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.20]


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


def _majority(votes):
    n = votes.shape[1]
    out = np.empty(n, dtype=np.int8)
    for r in range(n):
        out[r] = np.bincount(votes[:, r], minlength=3).argmax()
    return out


def main():
    print("=== L5: conformal-PI shrinkage of 4b's 108 flips ===\n")

    y = pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])[
        "Irrigation_Need"].map(CLS).to_numpy().astype(np.int8)

    # 14-bank mean probabilities
    bank_oof_p  = _stack_probs(BANK_NAMES, "oof").mean(axis=0)
    bank_oof_p  = _norm(bank_oof_p)
    bank_test_p = _stack_probs(BANK_NAMES, "test").mean(axis=0)
    bank_test_p = _norm(bank_test_p)

    # Calibration: nonconformity on every OOF row
    s_oof = 1.0 - bank_oof_p[np.arange(len(y)), y]  # higher = less conformal
    print(f"nonconformity s_oof: mean={s_oof.mean():.4f}  "
          f"p50={np.median(s_oof):.4f}  p95={np.percentile(s_oof, 95):.4f}")

    # Per-α threshold (finite-sample correction)
    n = len(s_oof)
    q = {}
    for a in ALPHAS:
        k = int(np.ceil((1.0 - a) * (n + 1)))
        k = min(k, n) - 1
        q[a] = float(np.sort(s_oof)[k])
        print(f"  α={a:.2f}  q_α={q[a]:.5f}  "
              f"(rows in PI on average: "
              f"{((1.0 - bank_oof_p) <= q[a]).any(axis=1).mean()*100:.1f}%)")

    # ----- TEST-side: 4b's 108 flips -----
    b_test = _csv_argmax("submission_2other_raw_tier1b_k2")
    b_4b   = _csv_argmax("submission_idea4b_selective_override")
    flip_4b_mask = b_test != b_4b
    flip_4b_idx = np.where(flip_4b_mask)[0]
    n_flip = len(flip_4b_idx)
    print(f"\n4b flip count: {n_flip}")

    # For each flip: nonconformity score for B's class vs 4b's class
    s_oldB = 1.0 - bank_test_p[flip_4b_idx, b_test[flip_4b_idx]]
    s_new4b = 1.0 - bank_test_p[flip_4b_idx, b_4b[flip_4b_idx]]
    p_oldB = bank_test_p[flip_4b_idx, b_test[flip_4b_idx]]
    p_new4b = bank_test_p[flip_4b_idx, b_4b[flip_4b_idx]]

    print(f"\n4b flips — bank prob distribution:")
    print(f"  p(B[old]):  mean={p_oldB.mean():.4f}  "
          f"p10={np.percentile(p_oldB,10):.4f}  "
          f"p90={np.percentile(p_oldB,90):.4f}")
    print(f"  p(4b[new]): mean={p_new4b.mean():.4f}  "
          f"p10={np.percentile(p_new4b,10):.4f}  "
          f"p90={np.percentile(p_new4b,90):.4f}")

    # Per-α: how many of 108 flips survive (B excluded from PI)
    print(f"\nshrinkage by α (KEEP flip iff B is OUTSIDE PI):")
    test_kept = {}
    for a in ALPHAS:
        keep = s_oldB > q[a]
        test_kept[a] = keep
        print(f"  α={a:.2f}  q={q[a]:.5f}  "
              f"keep={int(keep.sum())}/{n_flip}  "
              f"drop={int((~keep).sum())}")

    # ----- OOF validation via focal_majority disagreer -----
    focal_oof_p = _stack_probs(FOCAL_NAMES, "oof").mean(axis=0)
    focal_oof_p = _norm(focal_oof_p)
    focal_oof_argmax = focal_oof_p.argmax(1)
    bank_oof_argmax = bank_oof_p.argmax(1)
    raw_oof = _norm(np.load(ART / "oof_recipe_full_te.npy")).argmax(1)
    t1b_oof = _norm(np.load(ART / "oof_tier1b_greedy_meta.npy")).argmax(1)
    bank_maj_oof = _majority(np.stack(
        [_norm(np.load(ART / f"oof_{n}.npy")).argmax(1) for n in BANK_NAMES],
        axis=0
    ))

    # Anchor: rawashishsin_2600 (closest-to-B role from L3)
    anchor_oof = _norm(np.load(ART / "oof_rawashishsin_2600.npy")).argmax(1)

    # OOF flip set: 4-way consensus + anchor disagreement
    flip_oof_mask = (anchor_oof != focal_oof_argmax) \
        & (raw_oof == focal_oof_argmax) \
        & (t1b_oof == focal_oof_argmax) \
        & (bank_maj_oof == focal_oof_argmax)
    flip_oof_idx = np.where(flip_oof_mask)[0]
    n_oof_flip = len(flip_oof_idx)
    print(f"\nOOF analog flip set (focal_maj disagreer vs rawashishsin "
          f"anchor): {n_oof_flip}")

    if n_oof_flip > 0:
        old_class_oof = anchor_oof[flip_oof_idx]
        new_class_oof = focal_oof_argmax[flip_oof_idx]
        s_old_oof = 1.0 - bank_oof_p[flip_oof_idx, old_class_oof]
        baseline_prec = float((new_class_oof == y[flip_oof_idx]).mean())
        print(f"baseline precision (no PI filter): {baseline_prec:.5f} "
              f"(n={n_oof_flip})")

        print(f"\nOOF precision under PI-filter (KEEP iff old class out of PI):")
        oof_results = {"baseline_prec": baseline_prec, "baseline_n": n_oof_flip,
                       "by_alpha": {}}
        for a in ALPHAS:
            keep = s_old_oof > q[a]
            n_kept = int(keep.sum())
            if n_kept == 0:
                print(f"  α={a:.2f}  kept=0  (no rows survive)")
                oof_results["by_alpha"][str(a)] = {
                    "kept": 0, "precision": None}
                continue
            prec = float((new_class_oof[keep] == y[flip_oof_idx[keep]]).mean())
            # Per-direction breakdown
            dir_prec = {}
            for fr in range(3):
                for to in range(3):
                    if fr == to:
                        continue
                    mm = keep & (old_class_oof == fr) & (new_class_oof == to)
                    if mm.sum() < 5:
                        continue
                    p = (new_class_oof[mm] == y[flip_oof_idx[mm]]).mean()
                    dir_prec[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = (
                        int(mm.sum()), float(p))
            print(f"  α={a:.2f}  kept={n_kept:5d}  "
                  f"prec={prec:.5f}  Δ={prec-baseline_prec:+.5f}")
            for fr_to, (nn, pp) in dir_prec.items():
                be = {"H->M": 0.909, "L->M": 0.612, "M->L": 0.387,
                      "M->H": 0.909}.get(fr_to, 0.5)
                mark = "✓" if pp >= be else "✗"
                print(f"      {fr_to}: n={nn}  prec={pp:.4f}  BE={be} {mark}")
            oof_results["by_alpha"][str(a)] = {
                "kept": n_kept, "precision": prec,
                "delta_vs_baseline": prec - baseline_prec,
                "per_direction_precision": dir_prec,
            }
    else:
        oof_results = {"baseline_prec": None, "baseline_n": 0,
                       "by_alpha": {}}

    # ----- Emit candidate submissions for high-shrinkage levels -----
    print(f"\nemitting candidate test submissions for inspection:")
    test_ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
    candidates = {}
    for a in [0.05, 0.10]:
        keep = test_kept[a]
        new_pred = b_test.copy()
        idx = flip_4b_idx[keep]
        new_pred[idx] = b_4b[idx]
        n_kept = int(keep.sum())
        if n_kept == n_flip:
            print(f"  α={a:.2f}: kept all {n_flip} → identical to 4b, skipping")
            continue
        out_csv = SUB / f"submission_4b_conformal_a{int(a*100):02d}.csv"
        pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(
                {0: "Low", 1: "Medium", 2: "High"}),
        }).to_csv(out_csv, index=False)
        candidates[str(a)] = {"path": str(out_csv), "n_flips": n_kept}
        print(f"  α={a:.2f}: emitted {out_csv} (kept {n_kept}/{n_flip})")

    out = {
        "alphas": ALPHAS,
        "q_alpha": {str(a): q[a] for a in ALPHAS},
        "test_n_flips_4b": n_flip,
        "test_kept_by_alpha": {str(a): int(test_kept[a].sum()) for a in ALPHAS},
        "test_p_old_mean": float(p_oldB.mean()),
        "test_p_new_mean": float(p_new4b.mean()),
        "oof_validation": oof_results,
        "candidates": candidates,
    }
    (ART / "L5_conformal_4b_shrinkage.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n→ wrote scripts/artifacts/L5_conformal_4b_shrinkage.json")


if __name__ == "__main__":
    main()
