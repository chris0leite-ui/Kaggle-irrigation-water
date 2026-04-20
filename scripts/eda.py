"""
EDA pass for playground-series-s6e4 (Irrigation_Need).

Uses a stratified 50% subsample of train so that the analysis (feature
rankings, distribution shapes, decision-rule intuitions) is not
over-fit to the full training set. The remaining half is left untouched
as a holdout for later sanity checks.

Outputs:
  plots/eda/*.png       — per-feature stacked bars / KDEs
  plots/eda/report.html — self-contained report with embedded images
"""
from __future__ import annotations

import base64
import html
import os
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

plt.rcParams["figure.dpi"] = 110
OUT = Path("plots/eda")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLASS_COLORS = {"Low": "#4daf4a", "Medium": "#ff7f00", "High": "#e41a1c"}
SEED = 42
EDA_FRAC = 0.5  # holdout the other 50%

# ---------------------------------------------------------------- load / split
tr_full = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

eda, holdout = train_test_split(
    tr_full,
    test_size=1 - EDA_FRAC,
    stratify=tr_full[TARGET],
    random_state=SEED,
)
eda = eda.reset_index(drop=True)

num_cols = eda.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in eda.columns if c not in num_cols + [TARGET, ID]]

print(f"train_full={tr_full.shape}  eda_subsample={eda.shape}  holdout={holdout.shape}")
print(f"cat_cols ({len(cat_cols)}): {cat_cols}")
print(f"num_cols ({len(num_cols)}): {num_cols}")

# ---------------------------------------------------------------- summaries
miss_tr = tr_full.isna().sum()
miss_te = te.isna().sum()
prior_full = tr_full[TARGET].value_counts(normalize=True).reindex(CLASSES).round(4)
prior_eda = eda[TARGET].value_counts(normalize=True).reindex(CLASSES).round(4)

cat_card = pd.DataFrame(
    [{"feature": c, "n_levels": eda[c].nunique(),
      "levels": ", ".join(sorted(eda[c].unique()))}
     for c in cat_cols]
)

num_stats = eda[num_cols].describe().T.round(3)

# test-only categorical levels (drift check uses the full train)
drift_rows = []
for c in cat_cols:
    extra = sorted(set(te[c].unique()) - set(tr_full[c].unique()))
    if extra:
        drift_rows.append({"feature": c, "test_only_levels": extra})
cat_drift = pd.DataFrame(drift_rows) if drift_rows else pd.DataFrame(
    columns=["feature", "test_only_levels"]
)

# ---------------------------------------------------------------- rankings
F, p = f_classif(eda[num_cols].values, eda[TARGET].values)
num_rank = (
    pd.DataFrame({"feature": num_cols, "F_stat": F, "p_value": p})
    .sort_values("F_stat", ascending=False)
    .reset_index(drop=True)
    .round(3)
)

cat_rank_rows = []
for c in cat_cols:
    ct = pd.crosstab(eda[c], eda[TARGET])
    chi, pv, _, _ = chi2_contingency(ct.values)
    cat_rank_rows.append((c, chi, pv))
cat_rank = (
    pd.DataFrame(cat_rank_rows, columns=["feature", "chi2", "p_value"])
    .sort_values("chi2", ascending=False)
    .reset_index(drop=True)
    .round(3)
)

# ---------------------------------------------------------------- plots
def save_fig(fig, filename: str) -> str:
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


cat_images: dict[str, str] = {}
for c in cat_cols:
    ct = pd.crosstab(eda[c], eda[TARGET], normalize="index").reindex(columns=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ct.plot(kind="bar", stacked=True, width=0.8, ax=ax,
            color=[CLASS_COLORS[c_] for c_ in CLASSES])
    ax.set_title(f"P({TARGET}|{c})")
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    cat_images[c] = save_fig(fig, f"target_by_{c}.png")

num_images: dict[str, str] = {}
for c in num_cols:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for cls in CLASSES:
        vals = eda.loc[eda[TARGET] == cls, c]
        if len(vals) > 10000:
            vals = vals.sample(10000, random_state=SEED)
        vals.plot(kind="kde", ax=ax, label=cls, color=CLASS_COLORS[cls])
    ax.set_title(f"{c} density by class")
    ax.legend(fontsize=8)
    num_images[c] = save_fig(fig, f"num_by_class_{c}.png")

# ---------------------------------------------------------------- console
print("\n=== class priors (full vs EDA) ===")
print(pd.concat({"full": prior_full, "eda": prior_eda}, axis=1))

print("\n=== numeric rank (F-stat) ===")
print(num_rank.to_string(index=False))

print("\n=== categorical rank (chi2) ===")
print(cat_rank.to_string(index=False))


# ---------------------------------------------------------------- HTML report
def df_to_html(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_html(index=index, classes="data", border=0, escape=False)


def img_tag(b64: str, alt: str) -> str:
    return (f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
            f'loading="lazy">')


priors_df = pd.DataFrame({"full_train": prior_full, "eda_subsample": prior_eda})
priors_df.index.name = "class"
priors_df = priors_df.reset_index()

miss_df = pd.DataFrame({"train_missing": miss_tr, "test_missing": miss_te})
miss_df = miss_df[(miss_df != 0).any(axis=1)].reset_index().rename(columns={"index": "feature"})
if miss_df.empty:
    miss_df = pd.DataFrame([{"feature": "(none)", "train_missing": 0, "test_missing": 0}])

num_stats_html = num_stats.reset_index().rename(columns={"index": "feature"})

drift_html = (
    df_to_html(cat_drift) if not cat_drift.empty
    else "<p>None — test shares train's categorical vocabulary.</p>"
)

cat_blocks = "\n".join(
    f'<section class="feat"><h3>{html.escape(c)}</h3>{img_tag(cat_images[c], c)}</section>'
    for c in cat_cols
)
num_blocks = "\n".join(
    f'<section class="feat"><h3>{html.escape(c)}</h3>{img_tag(num_images[c], c)}</section>'
    for c in num_cols
)

html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Irrigation_Need — EDA report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, system-ui, sans-serif;
         max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; color: #222; }}
  h1 {{ margin-bottom: 0; }}
  .subtitle {{ color: #666; margin-top: 0.3rem; }}
  h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.25rem; margin-top: 2.2rem; }}
  h3 {{ margin: 0.2rem 0 0.5rem 0; font-size: 1rem; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
           gap: 1rem; }}
  .feat {{ border: 1px solid #eee; border-radius: 8px; padding: 0.8rem;
           background: #fafafa; }}
  .feat img {{ width: 100%; height: auto; display: block; }}
  table.data {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  table.data th, table.data td {{ padding: 4px 10px; border-bottom: 1px solid #eee;
                                   text-align: right; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  table.data thead th {{ background: #f4f4f4; }}
  .callout {{ background: #fff8e1; border-left: 4px solid #f9a825;
              padding: 0.8rem 1rem; border-radius: 4px; margin: 1rem 0; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>Irrigation_Need — EDA report</h1>
<p class="subtitle">playground-series-s6e4 · balanced-accuracy metric · 3-class
(Low / Medium / High)</p>

<div class="callout">
<b>Subsample discipline.</b> To keep EDA-driven decisions from over-fitting
the training set, this report is computed on a stratified
<b>{int(EDA_FRAC*100)}%</b> subsample of <code>train.csv</code>
(<code>seed={SEED}</code>, stratified on <code>{TARGET}</code>). The
remaining rows are untouched and available as a holdout for later
verification. Full train: {len(tr_full):,} rows · EDA: {len(eda):,} rows ·
Holdout: {len(holdout):,} rows · Test: {len(te):,} rows.
</div>

<h2>1 · Target distribution</h2>
{df_to_html(priors_df)}
<p>Low dominates at ~59%, High is ~3%. Under balanced accuracy each class
contributes 1/3 of the score, so the model's ability to recall
<b>High</b> drives the leaderboard even though it contributes little to
log-loss or accuracy.</p>

<h2>2 · Data quality</h2>
<div class="two-col">
  <div>
    <h3>Missingness</h3>
    {df_to_html(miss_df)}
  </div>
  <div>
    <h3>Categorical drift (test-only levels)</h3>
    {drift_html}
  </div>
</div>

<h2>3 · Categorical cardinalities</h2>
{df_to_html(cat_card)}

<h2>4 · Numeric summary statistics</h2>
{df_to_html(num_stats_html)}

<h2>5 · Feature signal ranking</h2>
<div class="two-col">
  <div>
    <h3>Numeric — F-stat vs target</h3>
    {df_to_html(num_rank)}
  </div>
  <div>
    <h3>Categorical — chi² vs target</h3>
    {df_to_html(cat_rank)}
  </div>
</div>

<h2>6 · Target distribution per categorical</h2>
<div class="grid">
{cat_blocks}
</div>

<h2>7 · Numeric density by class</h2>
<div class="grid">
{num_blocks}
</div>

</body>
</html>
"""

(OUT / "report.html").write_text(html_doc, encoding="utf-8")
print(f"\nreport written to {OUT}/report.html "
      f"({os.path.getsize(OUT / 'report.html') / 1024:.0f} KB)")
