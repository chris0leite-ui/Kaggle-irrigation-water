"""Univariate distribution shift between original (10k) and train (630k).

For each numeric: KS statistic + Cohen's d + percentile diff.
For each categorical: chi-square + JS divergence + per-level marginal shift.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "scripts" / "artifacts"

CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season",
    "Crop_Growth_Stage", "Mulching_Used", "Irrigation_Type", "Water_Source",
]
NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Sunlight_Hours", "Organic_Carbon",
    "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def main():
    train = pd.read_pickle(ART / "_dist_shift_train.pkl")
    orig = pd.read_pickle(ART / "_dist_shift_orig.pkl")

    print(f"train n={len(train):,}  orig n={len(orig):,}")
    print(f"\n=== NUMERIC SHIFTS (orig vs train) ===")
    print(f"{'col':28s} {'orig_mean':>10s} {'train_mean':>10s} "
          f"{'orig_std':>9s} {'train_std':>9s} "
          f"{'cohen_d':>8s} {'KS':>6s} {'KS_p':>9s} {'p1_diff':>8s} {'p99_diff':>8s}")
    print("-" * 130)

    num_results = {}
    for c in NUMS:
        a = orig[c].dropna().values
        b = train[c].dropna().values
        ks_stat, ks_p = stats.ks_2samp(a, b, mode="asymp")
        d = cohens_d(a, b)
        p1_diff = np.percentile(b, 1) - np.percentile(a, 1)
        p99_diff = np.percentile(b, 99) - np.percentile(a, 99)
        num_results[c] = {
            "orig_mean": float(a.mean()),
            "train_mean": float(b.mean()),
            "orig_std": float(a.std(ddof=1)),
            "train_std": float(b.std(ddof=1)),
            "cohen_d": float(d),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "p1_diff": float(p1_diff),
            "p99_diff": float(p99_diff),
            "orig_min": float(a.min()),
            "orig_max": float(a.max()),
            "train_min": float(b.min()),
            "train_max": float(b.max()),
        }
        print(f"{c:28s} {a.mean():>10.4f} {b.mean():>10.4f} "
              f"{a.std(ddof=1):>9.4f} {b.std(ddof=1):>9.4f} "
              f"{d:>+8.4f} {ks_stat:>6.4f} {ks_p:>9.2e} {p1_diff:>+8.3f} {p99_diff:>+8.3f}")

    print(f"\n=== CATEGORICAL SHIFTS (orig vs train) ===")
    print(f"{'col':28s} {'JS':>9s} {'chi2':>9s} {'chi2_p':>9s} {'max_marg_diff':>14s}  worst_levels")
    print("-" * 130)
    cat_results = {}
    for c in CATS:
        a = orig[c].astype(str)
        b = train[c].astype(str)
        levels = sorted(set(a.unique()) | set(b.unique()))
        ca = a.value_counts().reindex(levels, fill_value=0).values.astype(float)
        cb = b.value_counts().reindex(levels, fill_value=0).values.astype(float)
        pa = ca / ca.sum() if ca.sum() > 0 else ca
        pb = cb / cb.sum() if cb.sum() > 0 else cb
        js = js_divergence(pa, pb)
        chi2, chi2_p, _, _ = stats.chi2_contingency(np.array([ca, cb]) + 1e-9)
        diffs = pb - pa  # train minus orig
        ord_idx = np.argsort(-np.abs(diffs))[:3]
        worst = [(levels[i], float(diffs[i])) for i in ord_idx]
        cat_results[c] = {
            "n_levels": len(levels),
            "js": float(js),
            "chi2": float(chi2),
            "chi2_p": float(chi2_p),
            "max_marg_diff": float(np.max(np.abs(diffs))),
            "level_marg": {levels[i]: {"orig": float(pa[i]), "train": float(pb[i]),
                                        "diff": float(diffs[i])} for i in range(len(levels))},
        }
        worst_str = ", ".join(f"{lvl}:{d:+.3f}" for lvl, d in worst)
        print(f"{c:28s} {js:>9.5f} {chi2:>9.1f} {chi2_p:>9.2e} "
              f"{np.max(np.abs(diffs)):>14.4f}  {worst_str}")

    out = {"numeric": num_results, "categorical": cat_results}
    (ART / "_dist_shift_univariate.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {ART / '_dist_shift_univariate.json'}")
    return out


if __name__ == "__main__":
    main()
