"""Class-conditional and score-conditional shift between orig and train.

For each (class y) and each numeric, compute Cohen's d between orig and train.
For each (rule-score s) and each numeric, do the same — separates rule-driven
shift (which classes/scores the NN moved rows into/out of) from within-cell
shift (does the NN distort *features* given a fixed rule-cell?).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "scripts" / "artifacts"

NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Sunlight_Hours", "Organic_Carbon",
    "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season", "Crop_Growth_Stage",
    "Mulching_Used", "Irrigation_Type", "Water_Source",
]


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    train = pd.read_pickle(ART / "_dist_shift_train.pkl")
    orig = pd.read_pickle(ART / "_dist_shift_orig.pkl")

    out = {"per_class": {}, "per_score": {}, "rule_within_score": {}}

    print("\n=== CLASS-CONDITIONAL SHIFT (Cohen's d, train vs orig) ===")
    print("Positive d -> orig has higher mean than train.")
    cls_order = ["Low", "Medium", "High"]
    header = f"{'col':28s} | " + " | ".join(f"{c:>22s}" for c in cls_order)
    print(header)
    print("-" * len(header))
    for col in NUMS:
        row = []
        for cls in cls_order:
            a = orig.loc[orig.Irrigation_Need == cls, col].dropna().values
            b = train.loc[train.Irrigation_Need == cls, col].dropna().values
            d = cohens_d(a, b)
            ks, _ = stats.ks_2samp(a, b, mode="asymp") if (len(a) > 1 and len(b) > 1) else (np.nan, np.nan)
            row.append((d, ks, len(a), len(b)))
            out["per_class"].setdefault(cls, {})[col] = {
                "cohen_d": float(d) if not np.isnan(d) else None,
                "ks_stat": float(ks) if not np.isnan(ks) else None,
                "n_orig": int(len(a)),
                "n_train": int(len(b)),
                "orig_mean": float(a.mean()) if len(a) else None,
                "train_mean": float(b.mean()) if len(b) else None,
            }
        cells = " | ".join(f"d={d:+.3f} KS={ks:.3f}" for d, ks, _, _ in row)
        print(f"{col:28s} | {cells}")

    print("\n=== SCORE-CONDITIONAL SHIFT (Cohen's d) ===")
    print("Within each rule-score bin, did the NN distort feature distributions?")
    print(f"{'col':28s} | " + " | ".join(f"s={s}".rjust(10) for s in range(10)))
    print("-" * 150)
    for col in NUMS:
        ds = []
        for s in range(10):
            a = orig.loc[orig.dgp_score == s, col].dropna().values
            b = train.loc[train.dgp_score == s, col].dropna().values
            d = cohens_d(a, b) if (len(a) > 1 and len(b) > 1) else np.nan
            ds.append(d)
            out["per_score"].setdefault(int(s), {})[col] = {
                "cohen_d": float(d) if not np.isnan(d) else None,
                "n_orig": int(len(a)),
                "n_train": int(len(b)),
            }
        cells = " | ".join(f"{d:>+9.3f}" if not np.isnan(d) else f"{'NaN':>9s}" for d in ds)
        print(f"{col:28s} | {cells}")

    print("\n=== RULE-FEATURE shift restricted to within-rule-cell ===")
    print("For Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh:")
    print("compute mean within each (cell, score) and report mean abs diff orig vs train.")
    rule_cols = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"]
    for col in rule_cols:
        diffs = []
        for s in range(10):
            a = orig.loc[orig.dgp_score == s, col].dropna()
            b = train.loc[train.dgp_score == s, col].dropna()
            if len(a) > 5 and len(b) > 5:
                diffs.append((s, b.mean() - a.mean(), len(a), len(b)))
        out["rule_within_score"][col] = [{"score": int(s), "mean_diff": float(d),
                                           "n_orig": int(na), "n_train": int(nb)}
                                          for s, d, na, nb in diffs]
        s_str = ", ".join(f"s{s}:{d:+.2f}(n={na})" for s, d, na, nb in diffs)
        print(f"  {col:25s} | {s_str}")

    print("\n=== CATEGORICAL SHIFT WITHIN CLASS ===")
    print("For each (class, cat), JS divergence between orig and train marginals.")
    for cls in cls_order:
        for col in CATS:
            a = orig.loc[orig.Irrigation_Need == cls, col].astype(str)
            b = train.loc[train.Irrigation_Need == cls, col].astype(str)
            levels = sorted(set(a.unique()) | set(b.unique()))
            ca = a.value_counts().reindex(levels, fill_value=0).values.astype(float)
            cb = b.value_counts().reindex(levels, fill_value=0).values.astype(float)
            pa = ca / ca.sum() if ca.sum() > 0 else ca
            pb = cb / cb.sum() if cb.sum() > 0 else cb
            eps = 1e-12
            pa, pb = pa + eps, pb + eps
            pa, pb = pa / pa.sum(), pb / pb.sum()
            m = 0.5 * (pa + pb)
            js = 0.5 * (np.sum(pa * np.log(pa / m)) + np.sum(pb * np.log(pb / m)))
            out.setdefault("cat_per_class", {}).setdefault(cls, {})[col] = float(js)

    print(f"{'col':28s} | " + " | ".join(f"{c:>10s}" for c in cls_order))
    print("-" * 90)
    for col in CATS:
        cells = []
        for cls in cls_order:
            cells.append(f"{out['cat_per_class'][cls][col]:>10.5f}")
        print(f"{col:28s} | {' | '.join(cells)}")

    (ART / "_dist_shift_conditional.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
