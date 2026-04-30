"""T1 — select borderline test rows for LLM-judge mechanism.

A row is "borderline" when ≥2 of these hold:
  (a) ≥3 of {4b, idea5, B, v1_rf, rawashishsin} disagree on class.
  (b) DGP rule score in {3, 4, 5, 6, 7} (boundary band).
  (c) 14-bank max-prob < 0.85.

We additionally enrich with rule_pred and 14-bank majority/max-prob
columns so the override decision rule has them at hand.

Output:
  scripts/artifacts/T1_borderline_rows.parquet
  scripts/artifacts/T1_borderline_rows_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import BANK_NAMES, load_bank, bank_mean_probs  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

CLASS_INT = {"Low": 0, "Medium": 1, "High": 2}
CLASS_STR = {0: "Low", 1: "Medium", 2: "High"}


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLASS_INT).to_numpy(dtype=np.int8)


def dgp_score(test: pd.DataFrame) -> np.ndarray:
    dry = (test["Soil_Moisture"] < 25).astype(np.int8)
    norain = (test["Rainfall_mm"] < 300).astype(np.int8)
    hot = (test["Temperature_C"] > 30).astype(np.int8)
    windy = (test["Wind_Speed_kmh"] > 10).astype(np.int8)
    nomulch = (test["Mulching_Used"] == "No").astype(np.int8)
    Kc = test["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(np.int8) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + Kc).to_numpy(dtype=np.int8)


def rule_pred(score: np.ndarray) -> np.ndarray:
    out = np.full(score.shape, 0, dtype=np.int8)  # default Low
    out[(score >= 4) & (score <= 6)] = 1
    out[score >= 7] = 2
    return out


def main():
    print("=== T1 select borderline rows ===\n")

    # 1) Argmax of 5 LB-validated subs
    sub_names = {
        "4b": "submission_idea4b_selective_override",
        "idea5": "submission_idea5_anchor_switch",
        "B": "submission_2other_raw_tier1b_k2",
        "v1_rf": "submission_sklearn_rf_meta_natural_standalone_v1_lb98129",
        "raw": "submission_rawashishsin_2600_standalone",
    }
    args = {k: csv_argmax(v) for k, v in sub_names.items()}
    print("loaded 5 LB sub argmaxes:")
    for k, a in args.items():
        print(f"  {k}: L={int((a == 0).sum())} M={int((a == 1).sum())} H={int((a == 2).sum())}")

    # Disagreement = number of distinct argmax classes across 5 subs
    stack = np.stack(list(args.values()), axis=1)  # (270k, 5)
    n_distinct = np.zeros(stack.shape[0], dtype=np.int8)
    for i in range(stack.shape[0]):
        n_distinct[i] = len(set(stack[i].tolist()))
    cond_a = n_distinct >= 2  # at least 2 distinct argmax classes among 5 subs
    print(f"\ncond (a) ≥2 distinct argmaxes among 5 subs: {int(cond_a.sum())} rows")

    # 2) DGP rule boundary band
    test = pd.read_csv(DATA / "test.csv")
    score = dgp_score(test)
    rp = rule_pred(score)
    cond_b = (score >= 3) & (score <= 7)
    print(f"cond (b) score ∈ {{3..7}}: {int(cond_b.sum())} rows")

    # 3) 14-bank max-prob < 0.85
    bank = load_bank("test")  # (14, 270k, 3)
    bank_mean = bank_mean_probs(bank)
    bank_max = bank_mean.max(axis=1)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)
    cond_c = bank_max < 0.85
    print(f"cond (c) bank-mean max-prob < 0.85: {int(cond_c.sum())} rows")

    # Borderline = at least 2 of (a, b, c)
    cond_count = cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int)
    borderline = cond_count >= 2
    print(f"\nborderline (≥2 conditions): {int(borderline.sum())} rows")

    # Within the borderline set, prioritize by:
    #   - ALL 3 conditions trigger first
    #   - then 4b argmax disagrees with bank-majority
    #   - then by lowest bank max-prob (most uncertain)
    fb = args["4b"]
    fb_disagrees_bank = fb != bank_argmax
    triple = (cond_count == 3) & borderline
    print(f"  triple-triggered (all 3): {int(triple.sum())}")
    print(f"  borderline AND 4b!=bank_argmax: {int((borderline & fb_disagrees_bank).sum())}")

    # Build an enriched table for downstream prompt formatting.
    df = pd.DataFrame({
        "test_id": test["id"].values,
        "row_idx": np.arange(len(test)),
        "score": score,
        "rule_pred": [CLASS_STR[i] for i in rp],
        "fb": [CLASS_STR[i] for i in fb],
        "idea5": [CLASS_STR[i] for i in args["idea5"]],
        "B": [CLASS_STR[i] for i in args["B"]],
        "v1_rf": [CLASS_STR[i] for i in args["v1_rf"]],
        "raw": [CLASS_STR[i] for i in args["raw"]],
        "n_distinct_argmax": n_distinct,
        "bank_max_prob": bank_max,
        "bank_argmax": [CLASS_STR[i] for i in bank_argmax],
        "cond_a": cond_a,
        "cond_b": cond_b,
        "cond_c": cond_c,
        "borderline": borderline,
    })

    # Selection priority: triple-triggered + 4b!=bank_argmax first.
    df["priority"] = (
        triple.astype(int) * 100
        + (borderline & fb_disagrees_bank).astype(int) * 10
        + (1.0 - bank_max)  # uncertainty bonus
    )

    bord = df[df["borderline"]].sort_values("priority", ascending=False).reset_index(drop=True)
    print(f"\nfull borderline table: {len(bord)} rows")

    # Save full table + a top-500 cut for T1 budget.
    bord_path = ART / "T1_borderline_rows.csv"
    bord.to_csv(bord_path, index=False)
    print(f"saved {bord_path}")

    top500 = bord.head(500).copy()
    top500_path = ART / "T1_borderline_top500.csv"
    top500.to_csv(top500_path, index=False)
    print(f"saved {top500_path}")

    # Stats summary
    out = ART / "T1_borderline_rows_results.json"
    summary = {
        "n_total": int(borderline.sum()),
        "n_triple": int(triple.sum()),
        "n_borderline_and_4b_disagrees_bank": int((borderline & fb_disagrees_bank).sum()),
        "top500_score_distribution": (
            top500["score"].value_counts().sort_index().to_dict()
        ),
        "top500_4b_class_distribution": top500["fb"].value_counts().to_dict(),
        "top500_bank_argmax_distribution": top500["bank_argmax"].value_counts().to_dict(),
        "top500_bank_max_prob_quantiles": {
            f"q{p}": float(top500["bank_max_prob"].quantile(p / 100))
            for p in [1, 25, 50, 75, 99]
        },
    }
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsummary saved: {out}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
