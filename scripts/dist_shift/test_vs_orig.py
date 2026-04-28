"""Quick AV: test vs original. CLAUDE.md confirmed train↔test AUC=0.502
(no shift); train↔orig AUC=0.69 (this analysis). The transitive
question: is test↔orig also AUC≈0.69? If so, AV scores carry over from
train to test cleanly — and the test set has the SAME distortion vs orig.
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
DATA = ROOT / "data"

NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Sunlight_Hours", "Organic_Carbon",
    "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season", "Crop_Growth_Stage",
    "Mulching_Used", "Irrigation_Type", "Water_Source",
]


def main():
    test = pd.read_csv(DATA / "test.csv")
    orig = pd.read_csv(DATA / "irrigation_prediction.csv")

    # Subsample test to 50k for symmetry with train AV
    test_s = test.sample(50_000, random_state=42).reset_index(drop=True)

    df = pd.concat([orig.assign(is_test=0), test_s.assign(is_test=1)],
                   ignore_index=True)
    for c in CATS:
        df[c] = df[c].astype("category")
    X = df[NUMS + CATS]
    y = df["is_test"].values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        clf = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, feature_fraction=0.9, verbose=-1,
            random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                categorical_feature=CATS, callbacks=[])
        oof[va] = clf.predict_proba(X.iloc[va])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"Test (50k) vs Orig (10k) AV AUC = {auc:.4f}")

    out = {"test_vs_orig_av_auc": float(auc),
           "n_test": 50_000, "n_orig": int(len(orig))}
    (ART / "_dist_shift_test_vs_orig.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
