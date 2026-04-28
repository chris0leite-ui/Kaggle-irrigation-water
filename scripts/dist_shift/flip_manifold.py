"""Task 4 — Where do rule-flips concentrate?

The 10,304 synth flip rows (1.64%) are the ONLY rows the rule gets
wrong. They're the budget for any LB lift past 0.96-equivalent on the
rule. Three diagnostics:

(a) flip rate per dgp_score.
(b) flip rate per (score, direction) — score 3 flips usually go to
    Medium; score 6 flips go to High; etc.
(c) per-cell flip rate over the 128-cell rule cube and the 64-cell
    sub-cube. Are some cells purer than others?

This is a deeper version of the 2026-04-21 DGP residuals EDA but with
a focus on the cell-level structure, not the row-level featuredelta.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.dist_shift.loader import ARTI, load


def _bits(df):
    return {
        "dry": (df["Soil_Moisture"] < 25).astype(int),
        "norain": (df["Rainfall_mm"] < 300).astype(int),
        "hot": (df["Temperature_C"] > 30).astype(int),
        "windy": (df["Wind_Speed_kmh"] > 10).astype(int),
        "nomulch": (df["Mulching_Used"] == "No").astype(int),
        "kc": df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2,
    }


def _score(b):
    return 2 * (b["dry"] + b["norain"]) + b["hot"] + b["windy"] + b["nomulch"] + b["kc"]


def _rule(score):
    out = np.full(len(score), "Medium", dtype=object)
    out[score <= 3] = "Low"
    out[score >= 7] = "High"
    return out


def main() -> None:
    train, _test, orig = load()

    b = _bits(train)
    s = _score(b).to_numpy()
    rule = _rule(s)
    y = train["Irrigation_Need"].to_numpy()
    flip = (rule != y)
    n = len(train)

    # --- (a) flip rate per score
    print("=== (a) Per-score flip rates (synth-train) ===")
    rows = []
    for sc in sorted(set(s)):
        mask = (s == sc)
        n_sc = int(mask.sum())
        n_flip = int((mask & flip).sum())
        rows.append({
            "score": int(sc),
            "n_rows": n_sc,
            "n_flips": n_flip,
            "flip_rate_pct": round(n_flip / n_sc * 100, 4),
            "row_share_pct": round(n_sc / n * 100, 3),
            "rule_says": _rule(np.array([sc]))[0],
        })
    sd = pd.DataFrame(rows)
    print(sd.to_string(index=False))

    # --- (b) flip direction per score
    print("\n=== (b) Per-score flip DIRECTION (within flipped rows) ===")
    direction = []
    for sc in sorted(set(s)):
        mask = (s == sc) & flip
        if mask.sum() == 0:
            continue
        rule_class = _rule(np.array([sc]))[0]
        actual = pd.Series(y[mask]).value_counts().to_dict()
        actual_str = " ".join([f"{k}:{actual.get(k, 0)}" for k in ["Low", "Medium", "High"]])
        direction.append({"score": int(sc), "rule": rule_class, "actual_dist": actual_str, "n": int(mask.sum())})
    print(pd.DataFrame(direction).to_string(index=False))

    # --- (c) per-cell flip rates (128 cells)
    cell_id = (b["dry"] * 32 + b["norain"] * 16 + b["hot"] * 8 + b["windy"] * 4
               + b["nomulch"] * 2 + (b["kc"] // 2)).to_numpy()
    print(f"\n=== (c) Per-cell flip-rate distribution (max 128 cells, observed {len(set(cell_id))}) ===")

    cells = []
    for cid in sorted(set(cell_id)):
        m = (cell_id == cid)
        if m.sum() < 50:
            continue
        n_m = int(m.sum())
        n_f = int((m & flip).sum())
        sc = int(s[m][0])  # score is determined by cell
        cells.append({
            "cell": int(cid), "score": sc, "n_rows": n_m,
            "n_flips": n_f, "flip_rate_pct": round(n_f / n_m * 100, 3),
        })
    cdf = pd.DataFrame(cells)
    cdf = cdf.sort_values("flip_rate_pct", ascending=False).reset_index(drop=True)
    print("Top-10 highest-flip-rate cells:")
    print(cdf.head(10).to_string(index=False))
    print("\nBottom-5 lowest-flip-rate cells (≥1% volume):")
    big = cdf[cdf["n_rows"] >= 6300]  # at least 1% of train
    print(big.tail(5).to_string(index=False))

    # --- (d) feature shift WITHIN flip-rich vs flip-clean rows
    print("\n=== (d) Mean feature shift FLIP vs CLEAN rows (synth-train, all rows) ===")
    nums = ["Soil_pH", "Humidity", "Previous_Irrigation_mm",
            "Electrical_Conductivity", "Organic_Carbon", "Sunlight_Hours",
            "Field_Area_hectare"]
    drows = []
    for col in nums:
        v = train[col].to_numpy(dtype=float)
        a = v[~flip]
        bb = v[flip]
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + bb.var(ddof=1)))
        d = (bb.mean() - a.mean()) / pooled if pooled > 0 else 0.0
        drows.append({
            "col": col,
            "mean_clean": round(float(a.mean()), 3),
            "mean_flip": round(float(bb.mean()), 3),
            "delta": round(float(bb.mean() - a.mean()), 3),
            "cohen_d": round(float(d), 3),
        })
    drows_df = pd.DataFrame(drows).sort_values("cohen_d", key=lambda x: x.abs(), ascending=False)
    print(drows_df.to_string(index=False))

    # save
    out = {
        "per_score_flip": sd.to_dict(orient="records"),
        "per_score_direction": direction,
        "per_cell_top": cdf.head(20).to_dict(orient="records"),
        "non_rule_feature_shift_flip_vs_clean": drows_df.to_dict(orient="records"),
        "n_flips_total": int(flip.sum()),
        "n_train": int(n),
    }
    (ARTI / "flip_manifold_results.json").write_text(json.dumps(out, indent=2, default=str))
    cdf.to_csv(ARTI / "per_cell_flip_rates.csv", index=False)
    print(f"\nWrote {ARTI/'flip_manifold_results.json'}")


if __name__ == "__main__":
    main()
