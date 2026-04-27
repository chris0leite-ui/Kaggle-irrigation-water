"""Pre-flight diagnostic: 5-fold OOF binary AUC for the 3 residual targets.

Hypothesis test for Phase A: is each residual target structurally predictable
at OTE-feature capacity? If any binary AUC < 0.60 on a small XGB over the
residual TE keys + recipe basics, the lever is dead before the full pipeline.

Cheap (~5 min on full data, ~1 min on smoke). Uses 43 dist features +
dgp_score one-hot + key-derived simple stats. NO heavy recipe FE — this
isolates whether the SIGNAL exists, not whether it survives heavy reg.

Output: scripts/artifacts/preflight_residual_auc.json with per-target
{auc_oof, fold_aucs}. Decision rule:
  AUC ≥ 0.60 → Phase A worth running.
  AUC ∈ [0.55, 0.60) → marginal; run anyway, expect null on +0.0003 gate.
  AUC < 0.55 → skip Phase A (signal doesn't exist at OTE capacity).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402
from residual_te_helpers import (  # noqa: E402
    build_residual_targets, compute_rule_pred_score,
)

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log("loading raw train")
    train = pd.read_csv("data/train.csv")
    if SMOKE:
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int64)

    log("computing distance features + rule_pred")
    dist = add_distance_features(train.drop(columns=[TARGET]))
    dgp_score, rule_pred = compute_rule_pred_score(train)
    targets = build_residual_targets(y, dgp_score, rule_pred)

    feat_cols = [
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "dry", "norain", "hot", "windy", "nomulch", "kc_active",
        "dgp_score", "rule_pred", "min_axis_abs", "min_boundary_dist",
        "score_dist_low_mid", "score_dist_mid_high",
        "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
    ]
    # Include 11 raw nums for richer signal.
    raw_nums = ["Soil_Moisture", "Rainfall_mm", "Temperature_C",
                "Wind_Speed_kmh", "Humidity", "Soil_pH", "Sunlight_Hours",
                "Organic_Carbon", "Electrical_Conductivity",
                "Field_Area_hectare", "Previous_Irrigation_mm"]
    for c in raw_nums:
        if c in dist.columns:
            feat_cols.append(c)

    X = dist[feat_cols].astype(np.float32).values
    log(f"  X={X.shape}  features={len(feat_cols)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    for tname, y_bin in targets.items():
        prev = float(y_bin.mean())
        log(f"\n--- {tname}  prevalence={100*prev:.3f}%  positives={int(y_bin.sum()):,}")
        if y_bin.sum() < N_FOLDS * 5:
            log(f"  SKIP — too few positives for stratified CV")
            results[tname] = dict(auc_oof=None, prevalence=prev, skipped=True)
            continue

        oof_p = np.zeros(len(y), dtype=np.float32)
        fold_aucs = []
        # Stratify on the BINARY target so each fold has positives.
        for fold, (tr, va) in enumerate(skf.split(X, y_bin), 1):
            scale = (1 - prev) / max(prev, 1e-6)
            params = dict(
                n_estimators=300 if SMOKE else 600,
                max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1, reg_lambda=1,
                objective="binary:logistic", tree_method="hist",
                eval_metric="auc",
                scale_pos_weight=scale,
                early_stopping_rounds=30, verbosity=0,
                n_jobs=-1, random_state=SEED,
            )
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr], y_bin[tr],
                  eval_set=[(X[va], y_bin[va])], verbose=False)
            p = m.predict_proba(X[va])[:, 1]
            oof_p[va] = p
            auc = roc_auc_score(y_bin[va], p)
            fold_aucs.append(float(auc))
            log(f"  fold {fold} AUC={auc:.4f}  best_iter={m.best_iteration}")
        auc_oof = float(roc_auc_score(y_bin, oof_p))
        log(f"  OOF AUC={auc_oof:.4f}  fold-mean={np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f}")
        results[tname] = dict(
            auc_oof=auc_oof, prevalence=prev,
            fold_aucs=fold_aucs, n_positives=int(y_bin.sum()),
            decision="PROCEED" if auc_oof >= 0.60
                     else "MARGINAL" if auc_oof >= 0.55
                     else "SKIP",
        )

    out = ART / "preflight_residual_auc.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nwrote {out}")
    log("\nVERDICT")
    for k, v in results.items():
        if v.get("auc_oof") is None:
            log(f"  {k}  AUC=N/A  SKIPPED")
        else:
            log(f"  {k}  AUC={v['auc_oof']:.4f}  → {v['decision']}")


if __name__ == "__main__":
    main()
