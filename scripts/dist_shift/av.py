"""Task 2 — Adversarial Validation orig (0) vs synth-train (1).

Trains a 5-fold XGB on a balanced concat of orig (10k) + train-subsample
(10k) using only target-FREE features (cats factorized + raw nums +
4 rule-axis flags + 4 decimal-fraction features). Reports OOF AUC + per-
fold best_iter + gain-importance ranking.

Key contrast: J3 (2026-04-25) found train↔test AV AUC = 0.50247 (no
shift). Orig↔synth-train shift expected to be >> 0.55 given the
Rainfall_mm Cohen's d=0.315 marginal finding.

Saves: dist_shift/av_results.json + per-feature gain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.dist_shift.loader import ARTI, CATS, NUMS, load


def _build_features(df: pd.DataFrame, fact_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Cats factorized using a shared map; rule indicators & decimals added."""
    out = pd.DataFrame(index=df.index)
    fact_maps = fact_maps or {}
    for c in CATS:
        if c not in fact_maps:
            cats = sorted(df[c].astype(str).unique())
            fact_maps[c] = {v: i for i, v in enumerate(cats)}
        m = fact_maps[c]
        out[f"{c}_fact"] = df[c].astype(str).map(m).fillna(-1).astype("int32")
    for c in NUMS:
        out[c] = df[c].astype("float32")
    # rule indicators
    out["dry"] = (df["Soil_Moisture"] < 25).astype("int8")
    out["norain"] = (df["Rainfall_mm"] < 300).astype("int8")
    out["hot"] = (df["Temperature_C"] > 30).astype("int8")
    out["windy"] = (df["Wind_Speed_kmh"] > 10).astype("int8")
    # decimal fractions (kernel-audit pattern)
    for c in ["Temperature_C", "Organic_Carbon", "Soil_Moisture", "Soil_pH", "Sunlight_Hours"]:
        out[f"{c}_dec"] = ((df[c] * 100) % 100).round().astype("int16")
    return out, fact_maps


def main() -> None:
    train, _test, orig = load()
    rng = np.random.RandomState(42)

    # Balanced subsample of 10k synth-train
    sub_idx = rng.choice(len(train), size=10000, replace=False)
    synth_sub = train.iloc[sub_idx].reset_index(drop=True)

    n_orig = len(orig)
    n_synth = len(synth_sub)
    print(f"orig {n_orig} synth_sub {n_synth}")

    combined = pd.concat([orig, synth_sub], ignore_index=True)
    y = np.concatenate([np.zeros(n_orig), np.ones(n_synth)]).astype("int32")
    X, _ = _build_features(combined)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype="float32")
    fold_aucs = []
    fold_best_iters = []
    importance = {col: 0.0 for col in X.columns}

    for f, (tr, va) in enumerate(cv.split(X, y), 1):
        clf = xgb.XGBClassifier(
            n_estimators=2000, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0,
            eval_metric="auc", early_stopping_rounds=100,
            tree_method="hist", n_jobs=8, random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        p = clf.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        auc = roc_auc_score(y[va], p)
        fold_aucs.append(float(auc))
        fold_best_iters.append(int(clf.best_iteration))
        booster = clf.get_booster()
        scores = booster.get_score(importance_type="gain")
        # XGBoost may emit numeric-prefixed keys like "f12" if names lost
        # but we used a DataFrame so feature names should be col names
        for k, v in scores.items():
            if k in importance:
                importance[k] += v
            else:
                # fallback: try to map "f<n>" → col
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(X.columns):
                        importance[X.columns[idx]] += v
                else:
                    importance.setdefault(k, 0.0)
                    importance[k] += v
        print(f"  fold {f}: AUC={auc:.5f}  best_iter={clf.best_iteration}")

    overall_auc = roc_auc_score(y, oof)
    print(f"\nOverall OOF AUC = {overall_auc:.5f} (mean fold = {np.mean(fold_aucs):.5f})")

    # Top-15 features by gain
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:15]
    print("\nTop-15 features by total gain:")
    for k, v in top:
        print(f"  {k:40s} {v:12.1f}")

    out = {
        "overall_auc": float(overall_auc),
        "fold_aucs": fold_aucs,
        "fold_best_iters": fold_best_iters,
        "n_orig": int(n_orig), "n_synth_sub": int(n_synth),
        "top_features_by_gain": [{"feat": k, "gain": float(v)} for k, v in top],
        "all_feature_gain": {k: float(v) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])},
    }
    (ARTI / "av_results.json").write_text(json.dumps(out, indent=2))
    np.save(ARTI / "av_oof.npy", oof)
    print(f"\nWrote {ARTI/'av_results.json'} and av_oof.npy")


if __name__ == "__main__":
    main()
