"""Where do the 10,304 flipped synthetic rows live?

Train flips = rows where Irrigation_Need != rule_pred (1.64% of train).
Original has 0% flips (rule was reverse-engineered FROM original).

Key questions:
(1) Do flipped rows look like the original-cell distribution they "came from",
    or like the original-cell distribution they "moved to"?
(2) On non-rule features (the NN's flip lever), is there a feature combination
    that PERFECTLY identifies a flip vs a non-flip in synthetic train?
(3) Cohen's d on non-rule numerics: flipped vs non-flipped within each cell.
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
NON_RULE_NUMS = [c for c in NUMS if c not in
                 ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"]]
CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season", "Crop_Growth_Stage",
    "Mulching_Used", "Irrigation_Type", "Water_Source",
]


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2)
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    train = pd.read_pickle(ART / "_dist_shift_train.pkl")
    orig = pd.read_pickle(ART / "_dist_shift_orig.pkl")

    train["flipped"] = (train.Irrigation_Need != train.rule_pred).astype(int)
    n_flip = int(train.flipped.sum())
    print(f"Train: {len(train):,} rows, flipped = {n_flip:,} ({n_flip / len(train) * 100:.2f}%)")

    print("\n=== FLIP COUNT BY (score, true_class) ===")
    grid = train.groupby(["dgp_score", "Irrigation_Need", "rule_pred"]).size().unstack(fill_value=0)
    print(grid.head(40))

    print("\n=== FLIPS BY DIRECTION (rule_pred -> y) per score ===")
    flips = train[train.flipped == 1]
    direction = flips.groupby(["dgp_score"]).apply(
        lambda g: g.groupby(["rule_pred", "Irrigation_Need"]).size().unstack(fill_value=0)
    )
    print(direction)

    # For each score, compare flipped vs non-flipped on non-rule features within train
    print("\n=== WITHIN-SCORE Cohen's d: flipped vs non-flipped synthetic rows ===")
    print("Positive d -> flipped rows have HIGHER feature value than non-flipped peers.")
    print(f"{'col':28s} | " + " | ".join(f"s={s:d}".rjust(8) for s in range(10)))
    print("-" * 130)
    out_per_score = {}
    for col in NON_RULE_NUMS + ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"]:
        ds = []
        for s in range(10):
            cell = train[train.dgp_score == s]
            a = cell.loc[cell.flipped == 1, col].dropna().values
            b = cell.loc[cell.flipped == 0, col].dropna().values
            d = cohens_d(a, b)
            ds.append(d)
            out_per_score.setdefault(int(s), {})[col] = {
                "cohen_d": float(d) if not np.isnan(d) else None,
                "n_flip": int(len(a)),
                "n_clean": int(len(b)),
            }
        cells = " | ".join(f"{d:>+8.3f}" if not np.isnan(d) else f"{'NaN':>8s}" for d in ds)
        print(f"{col:28s} | {cells}")

    # Within-cell comparison: flipped synth rows vs ALL orig rows IN THE SAME (score, rule-class) CELL
    print("\n=== Where DO flipped rows belong on the orig manifold? ===")
    print("For each (score, rule_pred) cell: cohen's d between (flipped synth) and (orig same-cell-rule_pred).")
    print(f"{'cell':28s} {'col':25s} {'d':>8s}  {'n_flip':>7s} {'n_orig':>7s}")
    cell_results = {}
    for s in [3, 4, 5, 6, 7, 8]:
        for rp in ["Low", "Medium", "High"]:
            mask_flip = (train.dgp_score == s) & (train.flipped == 1) & (train.rule_pred == rp)
            mask_orig = (orig.dgp_score == s) & (orig.rule_pred == rp)
            if mask_flip.sum() < 10 or mask_orig.sum() < 5:
                continue
            cell_label = f"s={s},rp={rp}"
            cell_results[cell_label] = {}
            for col in ["Rainfall_mm", "Soil_Moisture", "Humidity", "Previous_Irrigation_mm",
                        "Electrical_Conductivity", "Field_Area_hectare"]:
                a = train.loc[mask_flip, col].dropna().values
                b = orig.loc[mask_orig, col].dropna().values
                d = cohens_d(a, b)
                cell_results[cell_label][col] = float(d) if not np.isnan(d) else None
                print(f"{cell_label:28s} {col:25s} {d:>+8.3f}  {len(a):>7d} {len(b):>7d}")
            print()

    # Train a classifier on synthetic train: predict flipped vs not from non-rule features only
    print("\n=== Flip-detector AUC from NON-RULE features only ===")
    feats = NON_RULE_NUMS + [c for c in CATS if c not in ["Mulching_Used", "Crop_Growth_Stage"]]
    df = train[feats + ["flipped"]].copy()
    for c in [c for c in CATS if c not in ["Mulching_Used", "Crop_Growth_Stage"]]:
        df[c] = df[c].astype("category")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    sample_idx = np.random.default_rng(42).choice(len(df), size=200_000, replace=False)
    df_s = df.iloc[sample_idx].reset_index(drop=True)
    y_s = df_s["flipped"].values
    X_s = df_s.drop(columns=["flipped"])
    oof_s = np.zeros(len(X_s))
    importances = []
    for tr, va in skf.split(X_s, y_s):
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.9, verbose=-1,
            random_state=42,
        )
        clf.fit(X_s.iloc[tr], y_s[tr], eval_set=[(X_s.iloc[va], y_s[va])],
                categorical_feature=[c for c in CATS if c not in ["Mulching_Used", "Crop_Growth_Stage"]],
                callbacks=[])
        oof_s[va] = clf.predict_proba(X_s.iloc[va])[:, 1]
        importances.append(pd.Series(clf.feature_importances_, index=X_s.columns))
    auc_nr = roc_auc_score(y_s, oof_s)
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print(f"Flip-detector AUC (non-rule features only, 200k subsample) = {auc_nr:.4f}")
    print("\nTop-10 importances (non-rule features):")
    print(imp.head(10).to_string())

    # Same with ALL features (sanity)
    feats_all = NUMS + CATS
    df = train[feats_all + ["flipped"]].copy()
    for c in CATS:
        df[c] = df[c].astype("category")
    df_s = df.iloc[sample_idx].reset_index(drop=True)
    y_s = df_s["flipped"].values
    X_s = df_s.drop(columns=["flipped"])
    oof_s = np.zeros(len(X_s))
    for tr, va in skf.split(X_s, y_s):
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.9, verbose=-1,
            random_state=42,
        )
        clf.fit(X_s.iloc[tr], y_s[tr], eval_set=[(X_s.iloc[va], y_s[va])],
                categorical_feature=CATS, callbacks=[])
        oof_s[va] = clf.predict_proba(X_s.iloc[va])[:, 1]
    auc_all = roc_auc_score(y_s, oof_s)
    print(f"\nFlip-detector AUC (ALL features, 200k subsample) = {auc_all:.4f}")
    print(f"Gap = {auc_all - auc_nr:.4f} -> rule features carry MOST of the flip-detection signal.")

    out = {
        "n_flip": n_flip,
        "n_train": int(len(train)),
        "per_score_within_synth": out_per_score,
        "flip_vs_orig_cell": cell_results,
        "flip_detector_auc_nonrule": float(auc_nr),
        "flip_detector_auc_all": float(auc_all),
        "flip_detector_top_imp_nonrule": imp.head(15).to_dict(),
    }
    (ART / "_dist_shift_flip.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
