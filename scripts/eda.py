"""
EDA pass for playground-series-s6e4 (Irrigation_Need).

Outputs:
  - plots/eda/target_by_<cat>.png   : stacked bar of class frequency per category level
  - plots/eda/num_by_class_<num>.png: kde of each numeric feature per class
  - prints a compact console summary.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
OUT = "plots/eda"
os.makedirs(OUT, exist_ok=True)

tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

print(f"train={tr.shape}  test={te.shape}")
print(f"cat_cols ({len(cat_cols)}): {cat_cols}")
print(f"num_cols ({len(num_cols)}): {num_cols}")

# ---------- missingness & cardinality ----------
print("\n=== missingness (train) ===")
miss = tr.isna().mean().sort_values(ascending=False)
print(miss[miss > 0] if (miss > 0).any() else "no missing values")

print("\n=== missingness (test) ===")
miss_te = te.isna().mean().sort_values(ascending=False)
print(miss_te[miss_te > 0] if (miss_te > 0).any() else "no missing values")

print("\n=== categorical cardinalities & levels ===")
for c in cat_cols:
    vals = tr[c].unique().tolist()
    print(f"  {c:25s} n={len(vals):3d}  levels={vals}")

# train/test coverage check: any categorical level in test that's missing in train?
print("\n=== test-only categorical levels ===")
any_drift = False
for c in cat_cols:
    extra = set(te[c].unique()) - set(tr[c].unique())
    if extra:
        any_drift = True
        print(f"  {c}: {extra}")
if not any_drift:
    print("  none — test shares train's categorical vocabulary")

# ---------- numeric summaries ----------
print("\n=== numeric summary (train) ===")
print(tr[num_cols].describe().T.round(3))

# ---------- target rate per categorical level ----------
print("\n=== class proportions overall ===")
print(tr[TARGET].value_counts(normalize=True).reindex(CLASSES).round(4))

print("\n=== balanced-accuracy intuition ===")
print("  under bal_acc, 'High' (3.3%) weighs 1/3 of the score")
print("  predicting argmax-of-softmax biases heavily against 'High'")

# ---------- save per-category target distribution plots ----------
for c in cat_cols:
    ct = pd.crosstab(tr[c], tr[TARGET], normalize="index").reindex(columns=CLASSES)
    ax = ct.plot(kind="bar", stacked=True, figsize=(6, 3.5), width=0.8,
                 color=["#4daf4a", "#ff7f00", "#e41a1c"])
    ax.set_title(f"P({TARGET}|{c})  — baseline=Low 0.587 / Med 0.379 / High 0.033")
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/target_by_{c}.png")
    plt.close()

# ---------- numeric densities split by class ----------
for c in num_cols:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for cls, col in zip(CLASSES, ["#4daf4a", "#ff7f00", "#e41a1c"]):
        vals = tr.loc[tr[TARGET] == cls, c]
        if len(vals) > 10000:
            vals = vals.sample(10000, random_state=0)
        vals.plot(kind="kde", ax=ax, label=cls, color=col)
    ax.set_title(f"{c} density by class")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/num_by_class_{c}.png")
    plt.close()

# ---------- rank numeric features by class-separation ----------
print("\n=== numeric class-separation (F-stat, higher = more informative) ===")
from sklearn.feature_selection import f_classif
X_num = tr[num_cols].values
y = tr[TARGET].values
F, p = f_classif(X_num, y)
rank = (pd.DataFrame({"feat": num_cols, "F": F, "p": p})
        .sort_values("F", ascending=False).reset_index(drop=True))
print(rank.round(3).to_string(index=False))

# ---------- mutual-information rank for categoricals ----------
print("\n=== categorical class-separation (chi2 on frequency table) ===")
from scipy.stats import chi2_contingency
rows = []
for c in cat_cols:
    ct = pd.crosstab(tr[c], tr[TARGET])
    chi, pv, _, _ = chi2_contingency(ct.values)
    rows.append((c, chi, pv))
cat_rank = pd.DataFrame(rows, columns=["feat", "chi2", "p"]).sort_values("chi2", ascending=False)
print(cat_rank.round(3).to_string(index=False))

print(f"\nplots saved to {OUT}/")
