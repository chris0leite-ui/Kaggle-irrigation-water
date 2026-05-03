"""EDA on a 50% stratified subsample. Generic across tabular comps.

Emits:
  - eda-summary.md (≤30 lines, agent-loadable)
  - plots/eda/report.html (self-contained, base64 images)
  - plots/eda/feature_signals.csv (F-stat / chi² ranked)

Reads target_col / id_col / task from comp-context.md.
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif, f_regression


def parse_comp_context() -> dict:
    text = Path("comp-context.md").read_text()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s*(.+?)\s*(#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=72)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ctx = parse_comp_context()
    target_col = ctx.get("target_col", "target")
    id_col = ctx.get("id_col", "id")
    task = ctx.get("task", "classification")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # 50% stratified subsample
    if task == "classification":
        sub = (train.groupby(target_col, group_keys=False)
                    .apply(lambda g: g.sample(frac=0.5, random_state=42)))
    else:
        sub = train.sample(frac=0.5, random_state=42)

    y = sub[target_col].values
    feat = sub.drop(columns=[target_col, id_col], errors="ignore")
    test_feat = test.drop(columns=[id_col], errors="ignore")
    num_cols = feat.select_dtypes(include="number").columns.tolist()
    cat_cols = feat.select_dtypes(include="object").columns.tolist()

    # Numeric feature signals
    if task == "classification":
        f_vals, _ = f_classif(feat[num_cols].fillna(0), y)
    else:
        f_vals, _ = f_regression(feat[num_cols].fillna(0), y.astype(float))
    num_signals = pd.Series(f_vals, index=num_cols).sort_values(ascending=False)

    # Categorical feature signals (chi² for classification)
    cat_signals = pd.Series(dtype=float)
    if cat_cols and task == "classification":
        chi_vals = []
        for c in cat_cols:
            ct = pd.crosstab(feat[c], y)
            chi2, _, _, _ = chi2_contingency(ct)
            chi_vals.append(chi2)
        cat_signals = pd.Series(chi_vals, index=cat_cols).sort_values(ascending=False)

    Path("plots/eda").mkdir(parents=True, exist_ok=True)
    pd.concat([num_signals.rename("F"), cat_signals.rename("chi2")], axis=1).to_csv(
        "plots/eda/feature_signals.csv")

    # Distribution drift train vs test (numeric only)
    drift = []
    for c in num_cols:
        if c in test_feat.columns:
            drift.append((c, abs(feat[c].mean() - test_feat[c].mean())
                          / max(feat[c].std(), 1e-9)))
    drift_df = pd.DataFrame(drift, columns=["col", "z_diff"]).sort_values(
        "z_diff", ascending=False)

    # Class priors
    if task == "classification":
        priors = pd.Series(y).value_counts(normalize=True).sort_index()
    else:
        priors = pd.Series(y).describe()

    # Plot top-3 numeric signals
    top_imgs = []
    for c in num_signals.head(3).index:
        fig, ax = plt.subplots(figsize=(6, 3))
        if task == "classification":
            for cl in sorted(np.unique(y)):
                ax.hist(feat[feat.index.isin(sub[sub[target_col] == cl].index)][c],
                        bins=50, alpha=0.5, label=str(cl))
            ax.legend()
        else:
            ax.scatter(feat[c], y, alpha=0.1, s=5)
        ax.set_title(f"{c} (F={num_signals[c]:.0f})")
        top_imgs.append((c, fig_to_b64(fig)))

    # Markdown summary
    summary_lines = [
        f"# EDA summary — {ctx.get('slug', '?')}",
        f"- train rows: {len(train):,}  test rows: {len(test):,}",
        f"- numeric features: {len(num_cols)}  categorical: {len(cat_cols)}",
        f"- missingness in train: {train.isna().mean().mean():.4f}",
        f"- class priors: {dict(priors)}" if task == "classification"
        else f"- target stats: {priors.to_dict()}",
        f"- top-5 numeric signals (F): {num_signals.head(5).to_dict()}",
    ]
    if not cat_signals.empty:
        summary_lines.append(
            f"- top-5 categorical signals (chi²): {cat_signals.head(5).to_dict()}")
    summary_lines.append(
        f"- top-3 train/test drift (z): {drift_df.head(3).to_dict('records')}")
    Path("eda-summary.md").write_text("\n".join(summary_lines) + "\n")

    # HTML report
    html = ["<html><body><h1>EDA</h1>"]
    for ln in summary_lines:
        html.append(f"<p>{ln}</p>")
    for c, b64 in top_imgs:
        html.append(f'<h3>{c}</h3><img src="data:image/png;base64,{b64}"/>')
    html.append("</body></html>")
    Path("plots/eda/report.html").write_text("\n".join(html))
    print("EDA done. See eda-summary.md and plots/eda/report.html")


if __name__ == "__main__":
    main()
