"""Task 1 — Marginal shift per column, orig vs synth-train.

For numerics: 2-sample Kolmogorov-Smirnov + Wasserstein-1 + Cohen's d.
For categoricals: chi-square over level counts + per-level frequency
delta (orig − train, in pp).

Saves: dist_shift/marginal_results.json + marginal_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.dist_shift.loader import ARTI, CATS, NUMS, load


def _ks_and_wasserstein(a: np.ndarray, b: np.ndarray) -> dict:
    ks = stats.ks_2samp(a, b, method="asymp")
    w = stats.wasserstein_distance(a, b)
    pooled_std = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    cohen_d = (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0
    return {
        "n_orig": int(len(a)),
        "n_train": int(len(b)),
        "mean_orig": float(a.mean()),
        "mean_train": float(b.mean()),
        "std_orig": float(a.std(ddof=1)),
        "std_train": float(b.std(ddof=1)),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(w),
        "cohen_d": float(cohen_d),
    }


def _chi2(a: pd.Series, b: pd.Series) -> dict:
    levels = sorted(set(a.unique()) | set(b.unique()))
    co = a.value_counts().reindex(levels, fill_value=0).values.astype(float)
    ct = b.value_counts().reindex(levels, fill_value=0).values.astype(float)
    table = np.vstack([co, ct])
    chi2, p, dof, _ = stats.chi2_contingency(table + 0.5)  # +0.5 smoothing
    fo = co / co.sum()
    ft = ct / ct.sum()
    delta_pp = (fo - ft) * 100.0
    levels_dict = {
        lvl: {"orig_pct": float(fo[i] * 100), "train_pct": float(ft[i] * 100),
              "delta_pp": float(delta_pp[i])}
        for i, lvl in enumerate(levels)
    }
    return {
        "n_levels": len(levels),
        "chi2": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
        "max_abs_delta_pp": float(np.abs(delta_pp).max()),
        "levels": levels_dict,
    }


def main() -> None:
    train, _test, orig = load()
    print(f"orig {orig.shape} | train {train.shape}")

    out: dict[str, dict] = {"numerics": {}, "categoricals": {}}

    for col in NUMS:
        out["numerics"][col] = _ks_and_wasserstein(
            orig[col].to_numpy(dtype=float),
            train[col].to_numpy(dtype=float),
        )
    for col in CATS:
        out["categoricals"][col] = _chi2(orig[col], train[col])

    (ARTI / "marginal_results.json").write_text(json.dumps(out, indent=2))

    # Summary table for the report
    rows = []
    for col, r in out["numerics"].items():
        rows.append({
            "kind": "num", "col": col,
            "ks": round(r["ks_stat"], 4),
            "ks_p": f"{r['ks_pvalue']:.2e}",
            "wass": round(r["wasserstein"], 4),
            "cohen_d": round(r["cohen_d"], 3),
            "mean_orig": round(r["mean_orig"], 3),
            "mean_train": round(r["mean_train"], 3),
        })
    for col, r in out["categoricals"].items():
        rows.append({
            "kind": "cat", "col": col,
            "ks": "-",
            "ks_p": f"{r['pvalue']:.2e}",
            "wass": "-",
            "cohen_d": "-",
            "mean_orig": f"{r['n_levels']} levels",
            "mean_train": f"max|Δpp|={r['max_abs_delta_pp']:.2f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(ARTI / "marginal_summary.csv", index=False)

    print("\n=== NUMERIC SHIFTS (sorted by |Cohen's d|) ===")
    nums = df[df["kind"] == "num"].copy()
    nums["cd_abs"] = nums["cohen_d"].abs()
    print(nums.sort_values("cd_abs", ascending=False).drop(columns="cd_abs").to_string(index=False))

    print("\n=== CATEGORICAL SHIFTS (sorted by p-value) ===")
    cats = df[df["kind"] == "cat"].copy()
    cats["pf"] = cats["ks_p"].astype(float)
    print(cats.sort_values("pf").drop(columns="pf").to_string(index=False))


if __name__ == "__main__":
    main()
