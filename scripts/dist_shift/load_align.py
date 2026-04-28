"""Load original 10k + synthetic 630k, align columns, compute rule features."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

CATS = [
    "Soil_Type", "Crop_Type", "Region", "Season",
    "Crop_Growth_Stage", "Mulching_Used", "Irrigation_Type", "Water_Source",
]
NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Wind_Speed_kmh", "Soil_pH", "Sunlight_Hours", "Organic_Carbon",
    "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
]
TARGET = "Irrigation_Need"


def add_rule_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dry"] = (df["Soil_Moisture"] < 25).astype(int)
    df["norain"] = (df["Rainfall_mm"] < 300).astype(int)
    df["hot"] = (df["Temperature_C"] > 30).astype(int)
    df["windy"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["nomulch"] = (df["Mulching_Used"] == "No").astype(int)
    df["kc"] = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    df["dgp_score"] = (
        2 * (df["dry"] + df["norain"])
        + df["hot"] + df["windy"] + df["nomulch"] + df["kc"]
    )
    df["rule_pred"] = pd.cut(
        df["dgp_score"], bins=[-1, 3, 6, 99], labels=["Low", "Medium", "High"]
    ).astype(str)
    return df


def main() -> dict:
    train = pd.read_csv(DATA / "train.csv")
    orig = pd.read_csv(DATA / "irrigation_prediction.csv")

    # Align column names. Original may have slightly different naming.
    print(f"train shape={train.shape}  cols={list(train.columns)}")
    print(f"orig  shape={orig.shape}  cols={list(orig.columns)}")

    shared = sorted(set(train.columns) & set(orig.columns))
    train_only = sorted(set(train.columns) - set(orig.columns))
    orig_only = sorted(set(orig.columns) - set(train.columns))
    print(f"\nshared cols ({len(shared)}): {shared}")
    print(f"train-only ({len(train_only)}): {train_only}")
    print(f"orig-only  ({len(orig_only)}): {orig_only}")

    train = add_rule_cols(train)
    orig = add_rule_cols(orig)

    # Save aligned, lightweight copies
    out = ROOT / "scripts" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in train.columns if c in NUMS + CATS + [TARGET, "dgp_score", "rule_pred"]]
    # Persist as feather (no extra deps) — pandas core pyarrow optional;
    # fallback to pickle for portability.
    train[keep].to_pickle(out / "_dist_shift_train.pkl")
    orig_keep = [c for c in orig.columns if c in keep]
    orig[orig_keep].to_pickle(out / "_dist_shift_orig.pkl")

    summary = {
        "train_n": int(len(train)),
        "orig_n": int(len(orig)),
        "shared_cols": shared,
        "train_only": train_only,
        "orig_only": orig_only,
        "train_class_dist": train[TARGET].value_counts().to_dict(),
        "orig_class_dist": orig[TARGET].value_counts().to_dict(),
        "train_score_dist": train["dgp_score"].value_counts().sort_index().to_dict(),
        "orig_score_dist": orig["dgp_score"].value_counts().sort_index().to_dict(),
        # Rule accuracy: how often does rule_pred match the actual label?
        "train_rule_acc": float((train["rule_pred"] == train[TARGET]).mean()),
        "orig_rule_acc": float((orig["rule_pred"] == orig[TARGET]).mean()),
    }
    print("\n--- summary ---")
    print(json.dumps(summary, default=str, indent=2))
    (out / "_dist_shift_summary.json").write_text(json.dumps(summary, default=str, indent=2))
    return summary


if __name__ == "__main__":
    main()
