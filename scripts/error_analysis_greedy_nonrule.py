"""Error analysis on greedy+nonrule OOF -> identify dominant error
clusters, then build targeted specialist(s).

Goal: find 1-2 biggest error clusters in the LB-best (greedy + nonrule
@ alpha=0.15, tuned). Feature-distribution comparison of errors vs
correct rows at the same (rule_pred, true_label) pair reveals the
per-feature signature of each error type. Then train an XGB on the
rows in that cluster (or its neighborhood) as a specialist override.

Output (diagnostic pass only — specialist pipelines separate):
  scripts/artifacts/error_analysis_greedy_nonrule.json
  plots/eda/error_analysis_greedy_nonrule.md (markdown summary)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import mannwhitneyu


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
PLOTS = Path("plots/eda")
PLOTS.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def compute_dgp(df):
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    return score, rule_pred


def cohen_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) + 1e-12)
    return (a.mean() - b.mean()) / pooled


def main():
    log("loading data + greedy+nonrule OOF")
    tr = pd.read_csv("data/train.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias = np.array(greedy_res["greedy_bias"])

    oof_base = log_blend2(oof_nonrule, oof_greedy, 0.15)
    log_p = np.log(np.clip(oof_base, 1e-9, 1.0))
    preds = (log_p + bias).argmax(axis=1)
    score, rule_pred = compute_dgp(tr)

    log(f"greedy+nonrule bal_acc ref: 0.97421  preds shape {preds.shape}")

    # Confusion matrix overall
    cm = confusion_matrix(y, preds)
    log(f"Overall CM:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Error rows grouped by (rule_pred, true_label, pred_label)
    errs = preds != y
    log(f"Total errors: {errs.sum()}")

    # 1) Break down by (rule_pred, true, pred) — the off-diagonal cells
    log("\n--- error cells: (rule_pred, true, pred) ---")
    cells = []
    for r in range(3):
        for t in range(3):
            for p in range(3):
                if t == p:
                    continue
                mask = errs & (rule_pred == r) & (y == t) & (preds == p)
                n = int(mask.sum())
                if n >= 100:  # skip trivial cells
                    cells.append({"rule": CLASSES[r], "true": CLASSES[t],
                                  "pred": CLASSES[p], "count": n, "mask": mask})

    cells = sorted(cells, key=lambda d: -d["count"])
    log(f"found {len(cells)} cells with >=100 errors")
    for c in cells[:15]:
        log(f"  rule={c['rule']:6s}  true={c['true']:6s}  pred={c['pred']:6s}  "
            f"n={c['count']}")

    top_cells = cells[:5]  # top 5 error clusters

    # 2) Break down by score
    log("\n--- errors by score ---")
    per_score_err = []
    for s in range(10):
        mask = score == s
        n = int(mask.sum())
        err = int((errs & mask).sum())
        rate = err / n if n > 0 else 0
        per_score_err.append({"score": s, "n_rows": n, "errs": err, "rate": rate})
        log(f"  score={s}  rows={n:6d}  errs={err:5d}  rate={rate:.4f}")

    # 3) For each top cluster, compute Cohen's d for continuous features
    log("\n--- per-cluster feature signatures (errs vs correct @ same rule_pred) ---")
    numeric = [
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "Soil_pH", "Organic_Carbon", "Electrical_Conductivity", "Humidity",
        "Sunlight_Hours", "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    cluster_features = []
    for c in top_cells:
        mask_err = c["mask"]
        # correct rows in the same rule_pred bucket AND same true label
        mask_ref = (~errs) & (rule_pred == CLS2IDX[c["rule"]]) & (y == CLS2IDX[c["true"]])
        n_err = int(mask_err.sum())
        n_ref = int(mask_ref.sum())
        log(f"\ncell rule={c['rule']} true={c['true']} pred={c['pred']}  "
            f"n_err={n_err}  n_ref={n_ref}")
        feat_summary = {}
        for col in numeric:
            a = tr.loc[mask_err, col].values
            b = tr.loc[mask_ref, col].values
            d = cohen_d(a, b)
            try:
                _, p_mw = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p_mw = float("nan")
            feat_summary[col] = {
                "mean_err": float(a.mean()), "mean_ref": float(b.mean()),
                "cohen_d": float(d), "p_mw": float(p_mw),
            }
        # sort by |d|
        ranked = sorted(feat_summary.items(), key=lambda kv: -abs(kv[1]["cohen_d"]))
        for col, stats in ranked[:6]:
            log(f"  {col:28s}  d={stats['cohen_d']:+.3f}  "
                f"mean_err={stats['mean_err']:.3f}  mean_ref={stats['mean_ref']:.3f}  "
                f"p_mw={stats['p_mw']:.2e}")
        cluster_features.append({
            "rule": c["rule"], "true": c["true"], "pred": c["pred"],
            "n_err": n_err, "n_ref": n_ref,
            "features": feat_summary,
        })

    # 4) Categorical breakdown for the top cluster
    log("\n--- cat-feature breakdown for top cluster ---")
    cat_feats = ["Soil_Type", "Crop_Type", "Season", "Irrigation_Type",
                 "Water_Source", "Region"]
    top = top_cells[0]
    mask_err = top["mask"]
    mask_ref = (~errs) & (rule_pred == CLS2IDX[top["rule"]]) & (y == CLS2IDX[top["true"]])
    cat_summary = {}
    for col in cat_feats:
        a = tr.loc[mask_err, col].value_counts(normalize=True)
        b = tr.loc[mask_ref, col].value_counts(normalize=True)
        # compute KL-like divergence (just report top value lift)
        diffs = {}
        for v in set(a.index) | set(b.index):
            pa = a.get(v, 0); pb = b.get(v, 0)
            diffs[v] = pa - pb
        top_lifts = sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:3]
        cat_summary[col] = {v: round(d, 4) for v, d in top_lifts}
        top_str = ", ".join(f"{v}:{d:+.3f}" for v, d in top_lifts)
        log(f"  {col:20s}  top-lifts: {top_str}")

    results = {
        "total_errs": int(errs.sum()),
        "cells_ge_100": [{
            "rule": c["rule"], "true": c["true"], "pred": c["pred"],
            "count": c["count"]
        } for c in cells],
        "top_cells": cluster_features,
        "per_score_errors": per_score_err,
        "top_cluster_cat_lifts": cat_summary,
    }
    with open(ART / "error_analysis_greedy_nonrule.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote {ART}/error_analysis_greedy_nonrule.json")


if __name__ == "__main__":
    main()
