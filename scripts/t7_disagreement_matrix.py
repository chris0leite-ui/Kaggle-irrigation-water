"""T7 — Test prediction agreement matrix across LB-verified submissions.

Identifies the test rows where our top LB-verified submissions disagree.
The +0.00020 LB lift to the pack lives in those rows.

Inputs (all LB-verified):
  primary  submission_tier1b_greedy_meta.csv          LB 0.98094
  3way     submission_3way_recipe025_s1035_s7040.csv  LB 0.98005
  realmlp  submission_lb3_realmlp_nonruleiso.csv      LB 0.98008
  pseudo   submission_recipe_greedy_recipe_pseudolabel.csv  LB 0.97998
  recipe   submission_recipe_full_te.csv              LB 0.97939
  catboost submission_recipe_full_te_catboost.csv     LB 0.97935

Outputs:
  scripts/artifacts/t7_agreement_matrix.json
  scripts/artifacts/t7_disagreement_rows.csv
  scripts/artifacts/t7_summary.txt
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

CANDIDATES = {
    "primary":  ("submission_tier1b_greedy_meta.csv",         "LB 0.98094"),
    "realmlp":  ("submission_lb3_realmlp_nonruleiso.csv",     "LB 0.98008"),
    "3way":     ("submission_3way_recipe025_s1035_s7040.csv", "LB 0.98005"),
    "pseudo":   ("submission_recipe_greedy_recipe_pseudolabel.csv",  "LB 0.97998"),
    "recipe":   ("submission_recipe_full_te.csv",             "LB 0.97939"),
    "catboost": ("submission_recipe_full_te_catboost.csv",    "LB 0.97935"),
}


def load_pred(name):
    df = pd.read_csv(SUB / name)
    return df["id"].values, df[TARGET].map(CLS_MAP).to_numpy()


def main():
    summary_lines = []

    def out(s=""): summary_lines.append(s); print(s, flush=True)

    out("=== T7: Test prediction agreement matrix ===\n")

    preds = {}
    ref_ids = None
    for k, (fname, lb) in CANDIDATES.items():
        ids, p = load_pred(fname)
        if ref_ids is None:
            ref_ids = ids
        else:
            assert (ids == ref_ids).all(), f"id mismatch for {k}"
        preds[k] = p
        out(f"  {k:<10} {fname:<55} {lb}  class dist={np.bincount(p, minlength=3).tolist()}")
    out()

    n = len(ref_ids)
    out(f"n_test = {n:,}\n")

    # Pairwise agreement matrix.
    keys = list(preds.keys())
    out(f"--- pairwise disagreement counts (rows differ) ---")
    out(f"{'':10}" + "".join(f"{k:>10}" for k in keys))
    for k1 in keys:
        row = [k1.ljust(10)]
        for k2 in keys:
            d = int((preds[k1] != preds[k2]).sum())
            row.append(f"{d:>10}")
        out("".join(row))
    out()

    # Rows where ≥1 candidate disagrees with primary.
    primary = preds["primary"]
    others = [preds[k] for k in keys if k != "primary"]
    any_diff = np.zeros(n, dtype=bool)
    for o in others:
        any_diff |= (o != primary)
    out(f"rows where any non-primary differs from primary: {any_diff.sum():,} ({any_diff.mean():.2%})")

    # Rows where ≥3 of the 5 non-primary predict differently from primary.
    diff_count = np.zeros(n, dtype=np.int32)
    for o in others:
        diff_count += (o != primary).astype(np.int32)
    high_minority = (diff_count >= 3)
    out(f"rows where ≥3/5 non-primary disagree with primary: {high_minority.sum():,} "
        f"({high_minority.mean():.3%})")

    # For each primary class, what does the majority of non-primary say?
    out(f"\n--- on rows where ≥3/5 disagree with primary ---")
    out(f"{'primary_class':<14} {'count':>8} {'majority_non_primary_class':>30}")
    for c in range(3):
        m = high_minority & (primary == c)
        if not m.any():
            continue
        votes = []
        for o in others:
            votes.append(o[m])
        votes_arr = np.stack(votes)  # shape (5, m_count)
        # majority class per row in the disagreement subset
        maj_per_row = []
        for j in range(m.sum()):
            cnt = Counter(votes_arr[:, j])
            maj_per_row.append(cnt.most_common(1)[0][0])
        maj_dist = np.bincount(maj_per_row, minlength=3)
        out(f"{IDX2CLS[c]:<14} {int(m.sum()):>8} {str(maj_dist.tolist()):>30}")
    out()

    # Confidence-weighted disagreements: use LB rank as confidence.
    # Lower-LB candidates may be wrong on these. The interesting question:
    # ON HIGH-MINORITY ROWS, does the majority side agree on a class
    # different from primary?
    out("--- per-cell distribution of disagreement rows ---")
    # dgp_score for test
    test_csv = pd.read_csv(DATA / "test.csv")
    dry = (test_csv["Soil_Moisture"].astype(float) < 25).astype(int)
    norain = (test_csv["Rainfall_mm"].astype(float) < 300).astype(int)
    hot = (test_csv["Temperature_C"].astype(float) > 30).astype(int)
    windy = (test_csv["Wind_Speed_kmh"].astype(float) > 10).astype(int)
    nomulch = (test_csv["Mulching_Used"].astype(str) == "No").astype(int)
    kc = np.where(test_csv["Crop_Growth_Stage"].astype(str).isin(("Flowering", "Vegetative")), 2, 0)
    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    score = score.values

    out(f"{'score':>5} {'n_test':>8} {'n_disagree(any)':>18} {'n_high_minority':>18}")
    for s in range(10):
        m = (score == s)
        out(f"{s:>5} {int(m.sum()):>8} {int((any_diff & m).sum()):>18} "
            f"{int((high_minority & m).sum()):>18}")
    out()

    # Save full disagreement-row dump for manual override consideration.
    rows = []
    for j in np.where(high_minority)[0]:
        rows.append({
            "test_id": int(ref_ids[j]),
            "score": int(score[j]),
            "primary": IDX2CLS[int(primary[j])],
            **{k: IDX2CLS[int(preds[k][j])] for k in keys},
        })
    df = pd.DataFrame(rows)
    df.to_csv(ART / "t7_disagreement_rows.csv", index=False)
    out(f"saved {len(df):,} high-minority disagreement rows -> t7_disagreement_rows.csv")

    # Pairwise matrix as JSON.
    pm = {k1: {k2: int((preds[k1] != preds[k2]).sum()) for k2 in keys} for k1 in keys}
    (ART / "t7_agreement_matrix.json").write_text(json.dumps({
        "candidates": {k: list(v) for k, v in CANDIDATES.items()},
        "pairwise_disagreements": pm,
        "rows_any_diff_vs_primary": int(any_diff.sum()),
        "rows_high_minority_vs_primary": int(high_minority.sum()),
        "n_test": int(n),
    }, indent=2))

    (ART / "t7_summary.txt").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()
