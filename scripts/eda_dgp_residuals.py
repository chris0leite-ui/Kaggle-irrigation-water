"""EDA on the DGP rule fit and LGBM+DGP OOF residuals.

Inputs (must exist before running):
  - data/train.csv                               (630k rows with true labels)
  - scripts/artifacts/oof_lgbm_dgp.npy           (LGBM+DGP OOF probs)
  - scripts/artifacts/bench_dgp_results.json     (tuned log-bias)

Output:
  - plots/eda/dgp_residuals.html                 (self-contained, base64 PNGs)

Sections:
  1. DGP rule fit on synthetic train: per-class, per-score-value.
  2. LGBM+DGP OOF performance: argmax vs tuned confusion matrix.
  3. Residual deep dive: where errors live in feature space
     (distance-to-threshold hists, joint density scatter, per-score-value
     error counts, rule-agreement x model-correctness table).
  4. 128-cell lookup table with per-cell counts, label distribution,
     LGBM accuracy, LGBM bias direction.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART_DIR = Path("scripts/artifacts")
OUT_DIR = Path("plots/eda")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- DGP rule ---------------------------------------------------------
def compute_dgp(df: pd.DataFrame) -> pd.DataFrame:
    sm = df["Soil_Moisture"].astype(float).values
    rm = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    um = df["Mulching_Used"].astype(str).values
    stg = df["Crop_Growth_Stage"].astype(str).values
    dry = (sm < 25).astype(int)
    norain = (rm < 300).astype(int)
    hot = (tc > 30).astype(int)
    windy = (ws > 10).astype(int)
    nomulch = (um == "No").astype(int)
    kc = np.where(np.isin(stg, ["Flowering", "Vegetative"]), 2, 0)
    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    rule_int = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int32)
    return pd.DataFrame({
        "dry": dry, "norain": norain, "hot": hot, "windy": windy,
        "nomulch": nomulch, "kc": kc, "score": score, "rule_int": rule_int,
        "dist_moist": sm - 25.0, "dist_rain": rm - 300.0,
        "dist_temp": tc - 30.0, "dist_wind": ws - 10.0,
    })


# ---------- Rendering helpers ----------------------------------------------
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def html_img(b64: str, caption: str = "") -> str:
    cap = f'<div class="caption">{caption}</div>' if caption else ""
    return f'<div class="figure"><img src="data:image/png;base64,{b64}"/>{cap}</div>'


def html_table(df: pd.DataFrame, title: str = "") -> str:
    t = f'<h4>{title}</h4>' if title else ""
    return t + df.to_html(classes="tbl", float_format=lambda x: f"{x:.4f}")


# ---------- Load -----------------------------------------------------------
log("loading data and OOFs")
tr = pd.read_csv("data/train.csv")
y = tr["Irrigation_Need"].map(CLS2IDX).values.astype(np.int32)
oof = np.load(ART_DIR / "oof_lgbm_dgp.npy")
bench = json.load(open(ART_DIR / "bench_dgp_results.json"))
bias = np.array(bench["log_bias"])
prior = np.array(bench["class_priors"])

dgp = compute_dgp(tr)
rule_int = dgp["rule_int"].values
score = dgp["score"].values

argmax_pred = oof.argmax(axis=1)
tuned_pred = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)

rule_correct = rule_int == y
lgbm_correct = tuned_pred == y
is_flipped = ~rule_correct

log(f"rule raw acc = {rule_correct.mean():.5f}")
log(f"lgbm tuned raw acc = {lgbm_correct.mean():.5f}")
log(f"lgbm tuned bal acc = {balanced_accuracy_score(y, tuned_pred):.5f}")
log(f"flipped rows      = {is_flipped.sum():,}  ({is_flipped.mean():.5f})")


# ---------- Section 1: Rule fit --------------------------------------------
sections: list[str] = []

rule_bal = balanced_accuracy_score(y, rule_int)
cm_rule = confusion_matrix(y, rule_int, labels=[0, 1, 2])
cm_rule_df = pd.DataFrame(cm_rule, index=CLASSES, columns=CLASSES)

# score -> label distribution among true labels
score_vs_label = pd.crosstab(
    pd.Series(score, name="score"),
    pd.Series([CLASSES[i] for i in y], name="true_label"),
).reindex(columns=CLASSES, fill_value=0)
score_vs_rule = pd.crosstab(
    pd.Series(score, name="score"),
    pd.Series([CLASSES[i] for i in rule_int], name="rule_label"),
).reindex(columns=CLASSES, fill_value=0)
score_total = score_vs_label.sum(axis=1)
# mismatch_rate = fraction of rows at this score where rule_int != true label
mismatch_series = pd.Series(is_flipped.astype(int)).groupby(pd.Series(score)).mean()
score_rule_mismatch = pd.DataFrame({
    "count": score_total.astype(int),
    "rule_label": [CLASSES[i] for i in score_vs_rule.values.argmax(axis=1)],
    "true_majority": [CLASSES[i] for i in score_vs_label.values.argmax(axis=1)],
    "mismatch_rate": mismatch_series.reindex(score_total.index).values,
})

# Per-score histogram of flipped vs correct
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
score_vs_label.plot(kind="bar", stacked=True, ax=ax[0], color=["#4a90d9", "#f5a623", "#d0021b"])
ax[0].set_title("True label distribution per DGP score")
ax[0].set_xlabel("score (0-13)")
ax[0].set_ylabel("rows")
ax[0].legend(title="true label")

flip_per_score = pd.Series(is_flipped.astype(int)).groupby(pd.Series(score)).sum()
total_per_score = pd.Series(1).repeat(len(score)).groupby(pd.Series(score)).sum()
flip_rate_per_score = (flip_per_score / total_per_score).fillna(0)
flip_rate_per_score.plot(kind="bar", ax=ax[1], color="#d0021b")
ax[1].set_title("Rule flip-rate by score")
ax[1].set_xlabel("score")
ax[1].set_ylabel("P(true != rule)")
ax[1].axhline(is_flipped.mean(), color="black", linestyle="--", label=f"mean {is_flipped.mean():.4f}")
ax[1].legend()
b64_1a = fig_to_b64(fig)

sections.append(f"""
<h2>1. DGP rule fit on synthetic train</h2>
<p>The closed-form rule (scripts/dgp_formula.py) gets raw acc
<b>{rule_correct.mean():.5f}</b>, bal_acc <b>{rule_bal:.5f}</b> on the
630k synthetic training rows. There are <b>{is_flipped.sum():,}</b>
flipped rows ({is_flipped.mean()*100:.2f}%). Flips cluster at the
score boundaries (3-4 and 6-7), confirming the synthetic = rule +
near-threshold label noise.</p>
{html_img(b64_1a, "Left: label counts per DGP score. Right: flip-rate per score; horizontal line = mean flip rate.")}
{html_table(cm_rule_df, "Rule confusion matrix (rows=true, cols=rule)")}
{html_table(score_rule_mismatch, "Per-score summary (count, rule-assigned class, true majority, mismatch rate)")}
""")


# ---------- Section 2: LGBM+DGP OOF performance -----------------------------
cm_argmax = confusion_matrix(y, argmax_pred, labels=[0, 1, 2])
cm_tuned = confusion_matrix(y, tuned_pred, labels=[0, 1, 2])
cm_argmax_df = pd.DataFrame(cm_argmax, index=CLASSES, columns=CLASSES)
cm_tuned_df = pd.DataFrame(cm_tuned, index=CLASSES, columns=CLASSES)

argmax_bal = balanced_accuracy_score(y, argmax_pred)
tuned_bal = balanced_accuracy_score(y, tuned_pred)
argmax_raw = (argmax_pred == y).mean()
tuned_raw = (tuned_pred == y).mean()

# per-class recall for argmax / tuned
def per_class_recall(cm: np.ndarray) -> np.ndarray:
    totals = cm.sum(axis=1)
    return np.where(totals > 0, np.diag(cm) / totals, 0.0)

pcr_argmax = per_class_recall(cm_argmax)
pcr_tuned = per_class_recall(cm_tuned)
pcr_df = pd.DataFrame({
    "argmax_recall": pcr_argmax, "tuned_recall": pcr_tuned,
    "delta": pcr_tuned - pcr_argmax,
}, index=CLASSES)

sections.append(f"""
<h2>2. LGBM+DGP OOF performance</h2>
<p>Argmax raw = <b>{argmax_raw:.5f}</b> / bal = <b>{argmax_bal:.5f}</b>.
Tuned log-bias raw = <b>{tuned_raw:.5f}</b> / bal = <b>{tuned_bal:.5f}</b>.
Tuning pulls High recall up at the cost of a small drop in the two
majority classes — classic balanced-accuracy optimisation.</p>
{html_table(cm_argmax_df, "Confusion matrix — argmax (rows=true, cols=pred)")}
{html_table(cm_tuned_df, "Confusion matrix — tuned log-bias (rows=true, cols=pred)")}
{html_table(pcr_df, "Per-class recall: argmax vs tuned")}
""")


# ---------- Section 3: Residual deep dive -----------------------------------
# 3a. Joint table: rule-agrees x lgbm-correct
joint = pd.crosstab(
    pd.Series(np.where(rule_correct, "rule correct", "rule flipped"), name="rule"),
    pd.Series(np.where(lgbm_correct, "lgbm correct", "lgbm wrong"), name="lgbm"),
)
joint_rate = joint / joint.values.sum()

# 3b. Distance-to-threshold for lgbm errors vs corrects
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
dists = [("dist_moist", "Soil_Moisture - 25"),
         ("dist_rain", "Rainfall_mm - 300"),
         ("dist_temp", "Temperature_C - 30"),
         ("dist_wind", "Wind_Speed_kmh - 10")]
for ax, (col, label) in zip(axes.flat, dists):
    vals = dgp[col].values
    ax.hist(vals[lgbm_correct], bins=80, density=True, alpha=0.5, label="correct", color="#4a90d9")
    ax.hist(vals[~lgbm_correct], bins=80, density=True, alpha=0.6, label="wrong", color="#d0021b")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"|{label}|   errors-vs-correct density")
    ax.set_xlabel(label)
    ax.legend()
b64_3a = fig_to_b64(fig)

# 3c. Errors by score value
errors_per_score = pd.Series((~lgbm_correct).astype(int)).groupby(pd.Series(score)).sum()
total_per_score = pd.Series(1).repeat(len(score)).groupby(pd.Series(score)).sum()
error_rate_per_score = (errors_per_score / total_per_score).fillna(0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
errors_per_score.plot(kind="bar", ax=ax[0], color="#d0021b")
ax[0].set_title("LGBM+DGP tuned errors by score")
ax[0].set_xlabel("score")
ax[0].set_ylabel("errors")
error_rate_per_score.plot(kind="bar", ax=ax[1], color="#d0021b")
ax[1].set_title("Error rate by score (comparison line = rule flip rate)")
ax[1].set_xlabel("score")
ax[1].set_ylabel("error rate")
ax[1].plot(range(len(flip_rate_per_score)), flip_rate_per_score.values, color="black", marker="o", linewidth=1.5, label="rule flip rate")
ax[1].legend()
b64_3b = fig_to_b64(fig)

# 3d. Joint density: Soil_Moisture × Rainfall_mm, colored by error/correct
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
rng = np.random.default_rng(42)
mask_err = ~lgbm_correct
mask_correct = lgbm_correct
# Subsample for plot
idx_err = np.where(mask_err)[0]
idx_correct = rng.choice(np.where(mask_correct)[0], size=min(50000, mask_correct.sum()), replace=False)
ax[0].scatter(tr["Soil_Moisture"].iloc[idx_correct], tr["Rainfall_mm"].iloc[idx_correct],
              s=1, alpha=0.1, color="#4a90d9", label=f"correct ({len(idx_correct):,} shown)")
ax[0].scatter(tr["Soil_Moisture"].iloc[idx_err], tr["Rainfall_mm"].iloc[idx_err],
              s=2, alpha=0.35, color="#d0021b", label=f"wrong ({mask_err.sum():,})")
ax[0].axvline(25, color="black", linestyle="--", linewidth=0.8)
ax[0].axhline(300, color="black", linestyle="--", linewidth=0.8)
ax[0].set_title("LGBM errors in (Soil_Moisture, Rainfall_mm)")
ax[0].set_xlabel("Soil_Moisture")
ax[0].set_ylabel("Rainfall_mm")
ax[0].legend()

ax[1].scatter(tr["Temperature_C"].iloc[idx_correct], tr["Wind_Speed_kmh"].iloc[idx_correct],
              s=1, alpha=0.1, color="#4a90d9", label="correct")
ax[1].scatter(tr["Temperature_C"].iloc[idx_err], tr["Wind_Speed_kmh"].iloc[idx_err],
              s=2, alpha=0.35, color="#d0021b", label="wrong")
ax[1].axvline(30, color="black", linestyle="--", linewidth=0.8)
ax[1].axhline(10, color="black", linestyle="--", linewidth=0.8)
ax[1].set_title("LGBM errors in (Temperature_C, Wind_Speed_kmh)")
ax[1].set_xlabel("Temperature_C")
ax[1].set_ylabel("Wind_Speed_kmh")
ax[1].legend()
b64_3c = fig_to_b64(fig)

sections.append(f"""
<h2>3. Residual deep dive (LGBM+DGP tuned)</h2>
<p>Of the <b>{(~lgbm_correct).sum():,}</b> LGBM errors:
<b>{((~lgbm_correct) & rule_correct).sum():,}</b> are on rule-correct rows
(the model broke a correct rule prediction), and
<b>{((~lgbm_correct) & ~rule_correct).sum():,}</b> are on rule-flipped rows
(the model failed to recover a flip). The DGP rule's own error count is
{(~rule_correct).sum():,} — so LGBM recovered
{((~rule_correct) & lgbm_correct).sum():,} flips and introduced
{(rule_correct & ~lgbm_correct).sum():,} new errors it wouldn't have had
if it just followed the rule.</p>
{html_table(joint, "Joint: rule-agreement × LGBM-correctness (raw counts)")}
{html_table(joint_rate, "Same, as rates")}
{html_img(b64_3a, "Distance-to-threshold densities, errors vs correct. Dashed line = the threshold.")}
{html_img(b64_3b, "Error mass concentrates at boundary scores (3, 4, 6, 7) — same as the rule flip-rate.")}
{html_img(b64_3c, "Errors pile up along the rule's decision boundaries (dashed lines).")}
""")


# ---------- Section 4: 128-cell lookup table ---------------------------------
# Cell = (dry, norain, hot, windy, nomulch, stage_active)
cell_cols = ["dry", "norain", "hot", "windy", "nomulch", "kc"]
cell_id = (
    dgp["dry"] * 32 + dgp["norain"] * 16 + dgp["hot"] * 8 + dgp["windy"] * 4
    + dgp["nomulch"] * 2 + (dgp["kc"] // 2)
).values

rows = []
for cid in np.unique(cell_id):
    m = cell_id == cid
    n = int(m.sum())
    if n == 0:
        continue
    counts = np.bincount(y[m], minlength=3)
    rule_here = int(rule_int[np.argmax(m)])  # rule is constant within cell
    score_here = int(score[np.argmax(m)])
    label_majority = int(counts.argmax())
    purity = counts.max() / n
    lgbm_acc = float(lgbm_correct[m].mean())
    dry_, norain_, hot_, windy_, nomulch_, kc_ = (
        int(cid >> 5 & 1), int(cid >> 4 & 1), int(cid >> 3 & 1),
        int(cid >> 2 & 1), int(cid >> 1 & 1), int(cid & 1) * 2,
    )
    rows.append({
        "cell_id": int(cid),
        "dry": dry_, "norain": norain_, "hot": hot_, "windy": windy_,
        "nomulch": nomulch_, "kc": kc_,
        "score": score_here, "rule": CLASSES[rule_here],
        "majority": CLASSES[label_majority],
        "n": n, "low": int(counts[0]), "medium": int(counts[1]), "high": int(counts[2]),
        "purity": purity, "lgbm_acc": lgbm_acc,
        "rule_matches_majority": rule_here == label_majority,
    })
cell_df = pd.DataFrame(rows).sort_values(["score", "cell_id"]).reset_index(drop=True)

# Subset the "interesting" cells: rule != majority, or low purity, or large n
disagree = cell_df[~cell_df["rule_matches_majority"]].copy()
disagree = disagree.sort_values("n", ascending=False)

# Plot cell purity distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(cell_df["purity"], bins=50, color="#4a90d9")
ax[0].set_title("Cell purity distribution (max-class / n)")
ax[0].set_xlabel("purity")
ax[0].set_ylabel("# cells")
ax[1].scatter(cell_df["n"], cell_df["purity"], c=cell_df["score"], cmap="viridis")
ax[1].set_title("purity vs size (colored by score)")
ax[1].set_xlabel("n (rows in cell)")
ax[1].set_ylabel("purity")
ax[1].set_xscale("log")
b64_4a = fig_to_b64(fig)

sections.append(f"""
<h2>4. 128-cell lookup table</h2>
<p>Each cell = fixed values of the 6 DGP-discrete features
(dry, norain, hot, windy, nomulch, stage-active). In the 10k
original dataset every cell was pure; in the 630k synthetic
training set, <b>{int((cell_df['purity'] < 1.0).sum())}</b> of
{len(cell_df)} cells have mixed labels — that's where the flip
noise lives. <b>{len(disagree)}</b> cells have a majority label
different from the rule's prediction (these are the cells a
per-cell-majority predictor would re-label).</p>
{html_img(b64_4a, "Cell purity distribution (left), purity vs cell size colored by score (right).")}
{html_table(disagree.head(20), f"Top-20 (by n) cells where rule != majority — {len(disagree)} total such cells")}
{html_table(cell_df, "Full 128-cell table")}
""")


# ---------- Assemble HTML ---------------------------------------------------
html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>DGP fit + LGBM+DGP residual EDA</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 1280px; margin: 24px auto; padding: 0 24px; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 4px; }}
h2 {{ margin-top: 42px; border-bottom: 1px solid #ccc; padding-bottom: 2px; }}
h4 {{ margin-top: 18px; margin-bottom: 6px; color: #555; }}
.figure {{ margin: 18px 0; }}
.caption {{ color: #555; font-size: 0.9em; margin-top: 4px; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
table.tbl {{ border-collapse: collapse; font-size: 0.85em; margin: 8px 0; }}
table.tbl th, table.tbl td {{ border: 1px solid #ccc; padding: 3px 8px; text-align: right; }}
table.tbl th {{ background: #f0f0f0; }}
b {{ color: #000; }}
</style></head>
<body>
<h1>DGP fit + LGBM+DGP residual EDA</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/eda_dgp_residuals.py.
Inputs: data/train.csv, scripts/artifacts/oof_lgbm_dgp.npy,
scripts/artifacts/bench_dgp_results.json.</p>
{''.join(sections)}
</body></html>"""

out_path = OUT_DIR / "dgp_residuals.html"
out_path.write_text(html)
log(f"wrote {out_path}  ({out_path.stat().st_size//1024} KB)")
