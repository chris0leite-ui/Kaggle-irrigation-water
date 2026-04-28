"""Joint-distribution shift between orig (10k) and train (630k):

(a) Adversarial validation classifier (LGBM) on `is_train` target.
(b) Per-feature gain importance to localize the shift.
(c) AUC restricted to score=k, class=k, and (score, class) cells.
(d) Test if AUC drops to chance (0.5) under within-rule-cell stratification.

If AV-AUC restricted to (score, class) cell ≈ 0.5 -> NN preserves the joint
within-cell distribution; if > 0.6 -> the NN distorts the manifold within
each rule cell, which directly bounds achievable model generalization.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "scripts" / "artifacts"

NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Sunlight_Hours", "Organic_Carbon",
    "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season", "Crop_Growth_Stage",
    "Mulching_Used", "Irrigation_Type", "Water_Source",
]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def fit_av(X, y, n_splits=3, seed=42, cat_cols=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    importances = []
    for tr, va in skf.split(X, y):
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.9, verbose=-1,
            random_state=seed,
        )
        clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                categorical_feature=cat_cols, callbacks=[])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
        importances.append(pd.Series(clf.feature_importances_, index=X.columns))
    auc = roc_auc_score(y, oof)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return auc, oof, imp


def main():
    train = pd.read_pickle(ART / "_dist_shift_train.pkl")
    orig = pd.read_pickle(ART / "_dist_shift_orig.pkl")

    # Subsample train down to ~50k for speed; AV needs class balance not size.
    rng = np.random.default_rng(42)
    sample_n = 50_000
    if len(train) > sample_n:
        idx = rng.choice(len(train), size=sample_n, replace=False)
        train_s = train.iloc[idx].reset_index(drop=True)
    else:
        train_s = train.reset_index(drop=True)

    feat_cols = NUMS + CATS
    df = pd.concat([orig.assign(is_train=0), train_s.assign(is_train=1)],
                   ignore_index=True)
    df = encode(df)
    X = df[feat_cols]
    y = df["is_train"].values

    print(f"\n=== ADVERSARIAL VALIDATION (orig=0, train_subsample=1, n_pos={(y==1).sum():,}, n_neg={(y==0).sum():,}) ===")
    auc, oof, imp = fit_av(X, y, n_splits=3, cat_cols=CATS)
    print(f"Global AV AUC = {auc:.4f}")
    print("\nTop-15 importances:")
    print(imp.head(15).to_string())

    # Cell-restricted AV: same model, but compute AUC restricted to (score, class) cells
    # By using oof on the GLOBAL classifier (which is fine: AUC is rank-only).
    df["oof"] = oof
    df["dgp_score"] = df.index.map(lambda i: orig.dgp_score.iloc[i] if i < len(orig)
                                   else train_s.dgp_score.iloc[i - len(orig)])
    df["irr_need"] = df.index.map(lambda i: orig.Irrigation_Need.iloc[i] if i < len(orig)
                                  else train_s.Irrigation_Need.iloc[i - len(orig)])

    cell_aucs = {}
    for (s, cls), g in df.groupby(["dgp_score", "irr_need"]):
        if g.is_train.nunique() < 2 or min((g.is_train == 0).sum(), (g.is_train == 1).sum()) < 5:
            continue
        cell_aucs[f"score={s},class={cls}"] = {
            "auc": float(roc_auc_score(g.is_train, g.oof)),
            "n_orig": int((g.is_train == 0).sum()),
            "n_train": int((g.is_train == 1).sum()),
        }

    print(f"\n=== CELL-RESTRICTED AV-AUC (subset of large cells) ===")
    print("AUC ~ 0.5 means within-cell joints match; AUC > 0.6 means NN distorts within-cell joint.")
    print(f"{'cell':35s} {'AUC':>7s} {'n_orig':>8s} {'n_train':>8s}")
    big_cells = sorted(cell_aucs.items(), key=lambda kv: -kv[1]["n_orig"])[:30]
    for k, v in big_cells:
        print(f"{k:35s} {v['auc']:>7.4f} {v['n_orig']:>8d} {v['n_train']:>8d}")

    # Score-restricted AV (collapse over class)
    print(f"\n=== SCORE-RESTRICTED AV-AUC ===")
    score_aucs = {}
    for s, g in df.groupby("dgp_score"):
        if g.is_train.nunique() < 2 or min((g.is_train == 0).sum(), (g.is_train == 1).sum()) < 5:
            continue
        score_aucs[int(s)] = {
            "auc": float(roc_auc_score(g.is_train, g.oof)),
            "n_orig": int((g.is_train == 0).sum()),
            "n_train": int((g.is_train == 1).sum()),
        }
    print(f"{'score':>5s} {'AUC':>7s} {'n_orig':>8s} {'n_train':>8s}")
    for s in sorted(score_aucs.keys()):
        v = score_aucs[s]
        print(f"{s:>5d} {v['auc']:>7.4f} {v['n_orig']:>8d} {v['n_train']:>8d}")

    out = {
        "global_auc": float(auc),
        "top_importances": imp.head(20).to_dict(),
        "score_auc": {str(k): v for k, v in score_aucs.items()},
        "cell_auc": cell_aucs,
    }
    (ART / "_dist_shift_av.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
