"""Hard-vote blending over saved submission CSVs.

We only have class labels (no saved OOF/test probs), so this script
operates purely on Low/Medium/High votes per row. Produces several
candidate blends:

    A) diverse-3 majority:  hybrid_v3, blend_lgbm_xgb_dist, lgbm_dgp
       tiebreaker -> hybrid_v3 (best LB)

    B) diverse-5 majority:  hybrid_v3, blend_lgbm_xgb_dist, lgbm_dgp,
                            lgbm_dist_bag_tuned, baseline_lgbm_tuned
       tiebreaker -> hybrid_v3

    C) weighted plurality:  weight each sub by max(LB - 0.9, 0.01)
       (LB from CLAUDE.md; OOF fallback for the two without LB)
       tiebreaker -> hybrid_v3

    D) rule-deferred hybrid_v3: for rows where rule score in {0,1,2,9}
       (rule >= 99.5% accurate per CLAUDE.md), use dgp_formula;
       otherwise use hybrid_v3. Defensive variant for the hidden split.

Also prints pairwise agreement between committee members so we can
see which blends are actually mixing different opinions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SUB = Path("submissions")
ID = "id"
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]

# (name, filename, LB-or-OOF score). LB is used where known,
# OOF is the fallback (marked lb=False).
COMMITTEE = {
    "hybrid_v3":   ("submission_xgb_hybrid_v3_routed012_spec678.csv", 0.97271, True),
    "hybrid":      ("submission_xgb_hybrid_routed_spec.csv",          0.97224, True),
    "blend_lx":    ("submission_blend_lgbm_xgb_dist.csv",             0.97170, True),
    "lgbm_dgp":    ("submission_lgbm_dgp_tuned.csv",                  0.97137, True),
    "baseline":    ("submission_baseline_lgbm_tuned.csv",             0.96972, True),
    "dgp_rule":    ("submission_dgp_formula.csv",                     0.95835, True),
    "lgbm_bag":    ("submission_lgbm_dist_bag_tuned.csv",             0.97289, False),
    "xgb_dist":    ("submission_xgb_dist_tuned.csv",                  0.97304, False),
}


def load_sub(fname: str) -> pd.DataFrame:
    df = pd.read_csv(SUB / fname)
    df = df.sort_values(ID).reset_index(drop=True)
    return df


def write_sub(ids: np.ndarray, labels: np.ndarray, fname: str) -> Path:
    out = pd.DataFrame({ID: ids, TARGET: labels})
    path = SUB / fname
    out.to_csv(path, index=False)
    return path


def majority(votes_matrix: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """votes_matrix: (n_rows, n_voters) string labels.
    fallback: (n_rows,) labels, used when no strict majority.
    """
    n, k = votes_matrix.shape
    out = np.empty(n, dtype=object)
    for i in range(n):
        c = Counter(votes_matrix[i])
        top = c.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1]:
            out[i] = top[0][0]
        else:
            out[i] = fallback[i]
    return out


def weighted_plurality(votes_matrix: np.ndarray, weights: np.ndarray,
                       fallback: np.ndarray) -> np.ndarray:
    """votes_matrix: (n_rows, n_voters) labels; weights: (n_voters,)."""
    n, k = votes_matrix.shape
    out = np.empty(n, dtype=object)
    for i in range(n):
        scores: dict[str, float] = {}
        for j in range(k):
            scores[votes_matrix[i, j]] = scores.get(votes_matrix[i, j], 0.0) + weights[j]
        best = max(scores.values())
        winners = [c for c, v in scores.items() if v == best]
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            out[i] = fallback[i]
    return out


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def main():
    loaded = {}
    for key, (fname, _, _) in COMMITTEE.items():
        loaded[key] = load_sub(fname)
    ids = loaded["hybrid_v3"][ID].values

    # Sanity: all share the same id order
    for key, df in loaded.items():
        assert np.array_equal(df[ID].values, ids), f"id mismatch in {key}"

    preds = {k: df[TARGET].values for k, df in loaded.items()}

    print("=== Pairwise agreement rates (on 270k test rows) ===")
    keys = list(preds.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            print(f"  {a:>10s} vs {b:<10s}  {agreement(preds[a], preds[b]):.4f}")

    # fallback: the best single sub (hybrid_v3, LB 0.97271)
    fallback = preds["hybrid_v3"]

    # ---- Blend A: diverse-3 majority ----
    keys_A = ["hybrid_v3", "blend_lx", "lgbm_dgp"]
    vm = np.stack([preds[k] for k in keys_A], axis=1)
    blend_A = majority(vm, fallback)
    path_A = write_sub(ids, blend_A, "submission_blend_vote3_diverse.csv")
    agree_best_A = agreement(blend_A, fallback)
    disagree_count_A = int((blend_A != fallback).sum())

    # ---- Blend B: diverse-5 majority ----
    keys_B = ["hybrid_v3", "blend_lx", "lgbm_dgp", "lgbm_bag", "baseline"]
    vm = np.stack([preds[k] for k in keys_B], axis=1)
    blend_B = majority(vm, fallback)
    path_B = write_sub(ids, blend_B, "submission_blend_vote5_diverse.csv")
    agree_best_B = agreement(blend_B, fallback)
    disagree_count_B = int((blend_B != fallback).sum())

    # ---- Blend C: weighted plurality ----
    # weight = max(score - 0.9, 0.01); effectively LB/OOF-based
    keys_C = ["hybrid_v3", "hybrid", "blend_lx", "lgbm_dgp",
              "lgbm_bag", "xgb_dist", "baseline"]
    weights_C = np.array([max(COMMITTEE[k][1] - 0.9, 0.01) for k in keys_C])
    vm = np.stack([preds[k] for k in keys_C], axis=1)
    blend_C = weighted_plurality(vm, weights_C, fallback)
    path_C = write_sub(ids, blend_C, "submission_blend_weighted.csv")
    agree_best_C = agreement(blend_C, fallback)
    disagree_count_C = int((blend_C != fallback).sum())

    # ---- Blend E: pairwise-veto ----
    # Override hybrid_v3 only on rows where the top-2 non-hybrid voters
    # (blend_lx and lgbm_dgp) AGREE on the SAME alternative class.
    # Strictest "two strong independent signals against best" rule.
    blend_lx_v = preds["blend_lx"]
    lgbm_dgp_v = preds["lgbm_dgp"]
    override_E = (blend_lx_v == lgbm_dgp_v) & (blend_lx_v != fallback)
    blend_E = fallback.copy()
    blend_E[override_E] = blend_lx_v[override_E]
    path_E = write_sub(ids, blend_E, "submission_blend_pairwise_veto.csv")
    disagree_count_E = int((blend_E != fallback).sum())
    agree_best_E = agreement(blend_E, fallback)

    # ---- Blend F: Borda count (LB-weighted) ----
    # Each voter ranks classes by their own prediction:
    #   voted class gets 2 points, other 2 classes get 0
    # Weights: LB/OOF score minus 0.9 (same as C).
    # Ties -> hybrid_v3.
    keys_F = keys_C
    weights_F = weights_C
    vm = np.stack([preds[k] for k in keys_F], axis=1)
    n = vm.shape[0]
    blend_F = np.empty(n, dtype=object)
    for i in range(n):
        scores = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        for j in range(vm.shape[1]):
            scores[vm[i, j]] += 2.0 * weights_F[j]
        best = max(scores.values())
        winners = [c for c, v in scores.items() if v == best]
        blend_F[i] = winners[0] if len(winners) == 1 else fallback[i]
    path_F = write_sub(ids, blend_F, "submission_blend_borda_weighted.csv")
    disagree_count_F = int((blend_F != fallback).sum())
    agree_best_F = agreement(blend_F, fallback)

    # ---- Blend G: supermajority on "High" only ----
    # High is the rare class (~3.3%) and drives balanced accuracy.
    # Rule: keep hybrid_v3's prediction UNLESS (a) hybrid_v3 does NOT
    # predict High but >= 3 of 6 diverse voters predict High, or
    # (b) hybrid_v3 predicts High but <=1 of 6 voters agree -> switch to
    # majority non-High class. Targets boundary {6,7,8} score rows.
    keys_G = ["hybrid_v3", "hybrid", "blend_lx", "lgbm_dgp", "lgbm_bag", "xgb_dist"]
    vm = np.stack([preds[k] for k in keys_G], axis=1)
    high_votes = (vm == "High").sum(axis=1)
    not_high_maj = np.array([
        Counter([x for x in vm[i] if x != "High"]).most_common(1)[0][0]
        if (vm[i] != "High").sum() > 0 else "Medium"
        for i in range(vm.shape[0])
    ], dtype=object)
    high_maj = np.array([
        Counter([x for x in vm[i] if x == "High"]).most_common(1)[0][0]
        if (vm[i] == "High").sum() > 0 else "High"
        for i in range(vm.shape[0])
    ], dtype=object)
    blend_G = fallback.copy()
    # (a) promote to High if 3+ voters say High and hybrid_v3 didn't
    promote = (high_votes >= 3) & (fallback != "High")
    blend_G[promote] = "High"
    # (b) demote from High if hybrid_v3 says High but <=1 others agree
    demote = (high_votes <= 1) & (fallback == "High")
    blend_G[demote] = not_high_maj[demote]
    path_G = write_sub(ids, blend_G, "submission_blend_high_supermajority.csv")
    disagree_count_G = int((blend_G != fallback).sum())
    agree_best_G = agreement(blend_G, fallback)

    # ---- Blend D: rule-deferred hybrid_v3 ----
    # On rule-trivial scores (0,1,2,9) the DGP rule is >= 99.5% accurate
    # per CLAUDE.md. Those rows are where dgp_rule and hybrid_v3 already
    # agree nearly 100% so this is mostly a no-op, but it's a clean
    # defensive mix: rule where rule is excellent, model elsewhere.
    rule = preds["dgp_rule"]
    # We don't have raw features here, so approximate "rule-trivial" as
    # "rows where rule predicts Low with high confidence (score 0-2) or
    # High (score 9)". Since rule labels correlate with score, we use
    # the simple rule: keep hybrid_v3 except replace Medium-predicted
    # rows where rule == hybrid_v3 with dgp; no-op in that case.
    # Correct approach: only override hybrid_v3 -> dgp_rule for rows
    # where hybrid_v3 disagrees with dgp_rule AND dgp_rule's label is
    # Low (rule-trivial Low scores 0-2) or High (rule-trivial High
    # score 9). This prefers the rule on its confident extreme rows.
    mask_hi = (rule == "High") & (preds["hybrid_v3"] != "High")
    mask_lo = (rule == "Low") & (preds["hybrid_v3"] != "Low")
    # Be surgical: only defer for rows where the blend_lx also agrees
    # with the rule (so we have 2 votes against hybrid_v3).
    blend_lx = preds["blend_lx"]
    override = ((mask_hi & (blend_lx == "High")) |
                (mask_lo & (blend_lx == "Low")))
    blend_D = fallback.copy()
    blend_D[override] = rule[override]
    path_D = write_sub(ids, blend_D, "submission_blend_rule_deferred.csv")
    agree_best_D = agreement(blend_D, fallback)
    disagree_count_D = int((blend_D != fallback).sum())

    print("\n=== Blend summaries ===")
    print(f"  A vote3 diverse   file={path_A.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_A:5d}  agree_best={agree_best_A:.4f}")
    print(f"  B vote5 diverse   file={path_B.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_B:5d}  agree_best={agree_best_B:.4f}")
    print(f"  C weighted        file={path_C.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_C:5d}  agree_best={agree_best_C:.4f}")
    print(f"  D rule-deferred   file={path_D.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_D:5d}  agree_best={agree_best_D:.4f}")
    print(f"  E pairwise-veto   file={path_E.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_E:5d}  agree_best={agree_best_E:.4f}")
    print(f"  F borda-weighted  file={path_F.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_F:5d}  agree_best={agree_best_F:.4f}")
    print(f"  G high-supermaj   file={path_G.name:40s}  rows_changed_vs_hybrid_v3={disagree_count_G:5d}  agree_best={agree_best_G:.4f}")

    # Class distribution comparison
    print("\n=== Class distribution (row counts) ===")
    for name, arr in [("hybrid_v3", fallback),
                      ("A vote3", blend_A),
                      ("B vote5", blend_B),
                      ("C weighted", blend_C),
                      ("D rule-def", blend_D),
                      ("E pairwise", blend_E),
                      ("F borda", blend_F),
                      ("G highsupmj", blend_G)]:
        c = Counter(arr)
        print(f"  {name:12s}  Low={c.get('Low',0):6d}  Medium={c.get('Medium',0):6d}  High={c.get('High',0):6d}")


if __name__ == "__main__":
    main()
