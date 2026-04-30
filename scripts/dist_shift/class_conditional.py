"""Task 3 — Class-conditional shift P(features | y) orig vs synth-train.

Three things matter here:

  (a) Class prior shift: P(y) orig vs synth.
  (b) Per-class numeric shift: KS + Cohen's d between orig[y=k]
      and synth[y=k] for each (k, num_col).
  (c) Rule-vs-label shift: how often does the closed-form DGP rule
      (dgp_score) match y on orig vs synth, and what does the
      score distribution look like in each.

This is the load-bearing diagnostic: marginal shift could come from
label flips (the NN re-labels rows away from the rule) OR feature
shift (the NN samples features from a different distribution AND
labels them rule-correctly). (a/b) distinguish those.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from scripts.dist_shift.loader import ARTI, NUMS, load


def _dgp_score(df: pd.DataFrame) -> np.ndarray:
    """Reverse-engineered closed-form rule from REPORT.md / dgp_formula.py."""
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + hot + windy + nomulch + kc).to_numpy()


def _rule_pred(score: np.ndarray) -> np.ndarray:
    out = np.full_like(score, "Medium", dtype=object)
    out[score <= 3] = "Low"
    out[score >= 7] = "High"
    return out


def main() -> None:
    train, _test, orig = load()

    # (a) class priors
    pri_orig = orig["Irrigation_Need"].value_counts(normalize=True).reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    pri_synth = train["Irrigation_Need"].value_counts(normalize=True).reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    print("=== (a) Class priors ===")
    print(pd.DataFrame({"orig": pri_orig, "synth": pri_synth,
                        "Δpp (synth-orig)": (pri_synth - pri_orig) * 100}).round(4))

    # (b) per-class numeric shifts
    print("\n=== (b) Per-class Cohen's d (synth - orig) for numerics ===")
    rows = []
    for cls in ["Low", "Medium", "High"]:
        for col in NUMS:
            a = orig.loc[orig["Irrigation_Need"] == cls, col].to_numpy(dtype=float)
            b = train.loc[train["Irrigation_Need"] == cls, col].to_numpy(dtype=float)
            if len(a) < 30 or len(b) < 30:
                continue
            ks = stats.ks_2samp(a, b, method="asymp")
            pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
            d = (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0
            rows.append({
                "y": cls, "col": col, "n_orig": len(a), "n_synth": len(b),
                "ks": round(float(ks.statistic), 4),
                "ks_p": float(ks.pvalue),
                "cohen_d": round(float(d), 3),
                "mean_orig": round(float(a.mean()), 3),
                "mean_synth": round(float(b.mean()), 3),
            })
    df = pd.DataFrame(rows)
    df["abs_d"] = df["cohen_d"].abs()
    print("Top 15 per-class shifts by |d|:")
    print(df.sort_values("abs_d", ascending=False).drop(columns="abs_d").head(15).to_string(index=False))

    # (c) rule-vs-label shift
    print("\n=== (c) Rule-vs-label match rate ===")
    s_orig = _dgp_score(orig)
    s_synth = _dgp_score(train)
    pred_orig = _rule_pred(s_orig)
    pred_synth = _rule_pred(s_synth)
    rule_match_orig = float((pred_orig == orig["Irrigation_Need"].values).mean())
    rule_match_synth = float((pred_synth == train["Irrigation_Need"].values).mean())
    print(f"orig:  rule matches y in {rule_match_orig*100:.4f}% of rows ({(pred_orig == orig['Irrigation_Need'].values).sum()}/{len(orig)})")
    print(f"synth: rule matches y in {rule_match_synth*100:.4f}% of rows ({(pred_synth == train['Irrigation_Need'].values).sum()}/{len(train)})")
    print(f"flip rate diff (orig - synth) = {(rule_match_orig - rule_match_synth)*100:.4f} pp")

    # score histogram per source
    print("\n--- dgp_score distribution (% of rows by score) ---")
    score_dist = pd.DataFrame({
        "orig":  pd.Series(s_orig).value_counts(normalize=True).sort_index() * 100,
        "synth": pd.Series(s_synth).value_counts(normalize=True).sort_index() * 100,
    }).round(3)
    score_dist["Δpp (synth-orig)"] = score_dist["synth"] - score_dist["orig"]
    print(score_dist)

    # save
    out = {
        "class_priors": {
            "orig": pri_orig.to_dict(),
            "synth": pri_synth.to_dict(),
        },
        "per_class_top_shifts": df.sort_values("abs_d", ascending=False).head(20).drop(columns="abs_d").to_dict(orient="records"),
        "rule_match_orig": rule_match_orig,
        "rule_match_synth": rule_match_synth,
        "score_distribution": score_dist.to_dict(),
    }
    (ARTI / "class_conditional_results.json").write_text(json.dumps(out, indent=2, default=str))
    df.to_csv(ARTI / "per_class_shifts.csv", index=False)
    print(f"\nWrote {ARTI/'class_conditional_results.json'}")


if __name__ == "__main__":
    main()
