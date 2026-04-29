"""R3 step 1: compute bias drift for each candidate bank component.

Drift = tuned_bias - (-log(prior)).
Natural-cal verdict per class: |drift| ≤ 0.3.
Component verdict: max |drift| across 3 classes ≤ 0.3.

Reports a ranked table so we can build a drift-filtered bank.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

CANDIDATES = [
    # 7 LB-best components
    "rawashishsin_2600",
    "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost",
    "recipe_full_te",
    "realmlp",
    "xgb_corn",
    "xgb_dist_digits",
    # On-disk extras to consider for natural-cal bank curation
    "recipe_full_te_catboost_skte",
    "xgb_dist_routed_v3",
    "xgb_nonrule",
    "lgbm_te_orig",
    "lgbm_dist_digits",
    "xgb_dist_digits_ote",
    "lgbm_dist_digits_ote",
    "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler",
    "recipe_pseudolabel_seed123labeler",
    "recipe_full_te_seed7",
    "recipe_full_te_seed123",
    "recipe_full_te_a01",
    "recipe_full_te_a10",
    "recipe_full_te_dropdet",
    "recipe_full_te_basemargin_K2",
    "recipe_full_te_residte",
    "recipe_full_te_lgbm",
    "recipe_full_te_avp",
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log("loading y to compute prior")
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr = len(y)
    prior = np.bincount(y, minlength=3) / n_tr
    neg_log_prior = -np.log(prior)
    log(f"  prior = [L={prior[0]:.4f} M={prior[1]:.4f} H={prior[2]:.4f}]")
    log(f"  -log(prior) = [L={neg_log_prior[0]:.4f} M={neg_log_prior[1]:.4f} H={neg_log_prior[2]:.4f}]")

    rows = []
    for name in CANDIDATES:
        oof_p = ART / f"oof_{name}.npy"
        if not oof_p.exists():
            log(f"  SKIP {name}: missing")
            continue
        oof = np.load(oof_p).astype(np.float32)
        if oof.shape != (n_tr, 3):
            log(f"  SKIP {name}: shape {oof.shape}")
            continue
        if (oof.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        # Normalize
        oof = oof / oof.sum(1, keepdims=True).clip(1e-9)
        argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
        bias, tuned_bal = tune_log_bias(oof, y, prior)
        drift = bias - neg_log_prior
        max_drift = float(np.abs(drift).max())
        rows.append(dict(
            name=name,
            argmax_bal=float(argmax_bal),
            tuned_bal=float(tuned_bal),
            bias=[round(float(b), 4) for b in bias.tolist()],
            drift=[round(float(d), 4) for d in drift.tolist()],
            max_abs_drift=round(max_drift, 4),
            natural_cal=max_drift <= 0.3,
            partial_natural_cal=max_drift <= 0.7,
        ))

    rows.sort(key=lambda r: r["max_abs_drift"])

    log("=" * 100)
    log(f"{'name':<48s}  {'argmax':<8s}  {'tuned':<8s}  {'drift_L':<8s}  {'drift_M':<8s}  {'drift_H':<8s}  {'verdict':<10s}")
    log("=" * 100)
    for r in rows:
        d = r["drift"]
        if r["natural_cal"]:
            v = "PASS"
        elif r["partial_natural_cal"]:
            v = "PARTIAL"
        else:
            v = "FAIL"
        log(f"{r['name']:<48s}  {r['argmax_bal']:.5f}  {r['tuned_bal']:.5f}  {d[0]:+.4f}  {d[1]:+.4f}  {d[2]:+.4f}  {v}")
    log("=" * 100)

    pass_list = [r["name"] for r in rows if r["natural_cal"]]
    partial_list = [r["name"] for r in rows if r["partial_natural_cal"] and not r["natural_cal"]]
    log(f"\nPASS (|drift| ≤ 0.3): {len(pass_list)} components")
    for n in pass_list:
        log(f"  + {n}")
    log(f"\nPARTIAL (|drift| ≤ 0.7): {len(partial_list)} components")
    for n in partial_list:
        log(f"  ~ {n}")

    out = ART / "drift_diagnostic.json"
    out.write_text(json.dumps(dict(
        prior=prior.tolist(),
        neg_log_prior=neg_log_prior.tolist(),
        components=rows,
        natural_cal_pass=pass_list,
        partial_pass=partial_list,
    ), indent=2, default=float))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
