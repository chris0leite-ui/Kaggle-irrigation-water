"""#3 Counterfactual-perturbation robustness features.

For each row (train OOF + test), perturb each of 4 rule-axis numerics
(Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh) by ±IQR×{0.05}
and count how many of v1's argmax predictions flip across the perturbation.
Encodes "primary stability" — rows with high flip count are decision-
boundary rows.

Cheap implementation: instead of retraining v1, compute the analog using
LB-best primary's PROB SHIFTS (no retrain — we estimate flip count via
proximity to threshold using existing per-row predictions).

We compute 5 features per row:
  cfr_sm_flip, cfr_rf_flip, cfr_tc_flip, cfr_ws_flip, cfr_total_flip
  (each = number of perturbations (out of 8 = 2 directions × 4 axes) that
   would cross the rule threshold for that axis)

Then evaluate as standalone CSV (impossible without retrain) AND as
override-gate diagnostic: rows with high CFR have noisy primary predictions
and are good override candidates; rows with low CFR have stable primary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, add_distance_features, DGP_THRESHOLDS  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

AXES = [
    ("Soil_Moisture", "sm", DGP_THRESHOLDS["sm"]),
    ("Rainfall_mm", "rf", DGP_THRESHOLDS["rf"]),
    ("Temperature_C", "tc", DGP_THRESHOLDS["tc"]),
    ("Wind_Speed_kmh", "ws", DGP_THRESHOLDS["ws"]),
]


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    n_train = len(train)
    n_test = len(test)

    # Compute per-axis flip count: how many σ-perturbations cross the threshold
    sigmas = [0.05, 0.10, 0.20]  # fraction of IQR
    print(f"Computing CFR features for {len(AXES)} axes × {len(sigmas)} σs...")

    # Concatenate train + test for IQR
    full_dfs = []
    for df, src in [(train, "train"), (test, "test")]:
        full_dfs.append(df[[c for c, _, _ in AXES]])
    full = pd.concat(full_dfs, axis=0, ignore_index=True)

    train_feats = pd.DataFrame(index=range(n_train))
    test_feats = pd.DataFrame(index=range(n_test))

    total_train = np.zeros(n_train, dtype=np.int32)
    total_test = np.zeros(n_test, dtype=np.int32)

    for col, short, thresh in AXES:
        v_train = train[col].astype(float).to_numpy()
        v_test = test[col].astype(float).to_numpy()
        iqr = float(np.percentile(full[col], 75) - np.percentile(full[col], 25))
        flip_count_train = np.zeros(n_train, dtype=np.int32)
        flip_count_test = np.zeros(n_test, dtype=np.int32)
        for sigma in sigmas:
            delta = sigma * iqr
            for direction in [+1, -1]:
                # Flip if (v + direction*delta) crosses threshold compared to v
                v_pert_train = v_train + direction * delta
                v_pert_test = v_test + direction * delta
                flipped_train = ((v_train < thresh) != (v_pert_train < thresh)).astype(np.int32)
                flipped_test = ((v_test < thresh) != (v_pert_test < thresh)).astype(np.int32)
                flip_count_train += flipped_train
                flip_count_test += flipped_test
        total_train += flip_count_train
        total_test += flip_count_test
        train_feats[f"cfr_{short}_flip"] = flip_count_train
        test_feats[f"cfr_{short}_flip"] = flip_count_test
        print(f"  {col} (thresh={thresh}, IQR={iqr:.3f}): "
              f"flip_count_train mean={flip_count_train.mean():.3f} max={flip_count_train.max()}")

    train_feats["cfr_total"] = total_train
    test_feats["cfr_total"] = total_test
    print(f"\nTotal flip count: train mean={total_train.mean():.2f} max={total_train.max()}")
    print(f"Train rows with cfr_total >= 4 (boundary): {(total_train >= 4).sum()} ({(total_train >= 4).mean()*100:.2f}%)")
    print(f"Test rows with cfr_total >= 4:  {(total_test >= 4).sum()} ({(total_test >= 4).mean()*100:.2f}%)")

    # Save
    np.save(ART / "cfr_features_train.npy", train_feats.to_numpy().astype(np.int32))
    np.save(ART / "cfr_features_test.npy", test_feats.to_numpy().astype(np.int32))
    train_feats.to_csv(ART / "cfr_features_train.csv", index=False)
    test_feats.to_csv(ART / "cfr_features_test.csv", index=False)
    print(f"\nSaved cfr_features_{{train,test}}.npy + .csv")

    # ===== Diagnostic: do override candidates concentrate on high-CFR rows? =====
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_oof = v1_oof / np.clip(v1_oof.sum(1, keepdims=True), 1e-9, None)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_oof = raw_oof / np.clip(raw_oof.sum(1, keepdims=True), 1e-9, None)
    t1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    t1b_oof = t1b_oof / np.clip(t1b_oof.sum(1, keepdims=True), 1e-9, None)
    v1_b, _ = tune_log_bias(v1_oof, y, prior)
    raw_b, _ = tune_log_bias(raw_oof, y, prior)
    t1b_b, _ = tune_log_bias(t1b_oof, y, prior)
    v1_arg = (np.log(np.clip(v1_oof, 1e-9, 1.0)) + v1_b).argmax(1)
    raw_arg = (np.log(np.clip(raw_oof, 1e-9, 1.0)) + raw_b).argmax(1)
    t1b_arg = (np.log(np.clip(t1b_oof, 1e-9, 1.0)) + t1b_b).argmax(1)
    cand = (raw_arg == t1b_arg) & (raw_arg != v1_arg)
    print(f"\nOOF override candidates: {cand.sum()}")
    if cand.sum() > 0:
        ratio = total_train[cand].mean() / total_train.mean()
        print(f"  CFR_total mean on cands: {total_train[cand].mean():.2f}  "
              f"vs all rows: {total_train.mean():.2f}  ratio: {ratio:.2f}x")
        # Per-direction
        consensus = raw_arg
        print(f"  CFR_total per direction:")
        for a in range(3):
            for c in range(3):
                if a == c: continue
                m = cand & (v1_arg == a) & (consensus == c)
                if m.sum() == 0: continue
                # Precision per direction
                target = (y[m] == c).astype(np.int32)
                # Compare CFR for correct vs wrong overrides
                correct_mask = m & (y == c)
                wrong_mask = m & (y != c)
                cfr_correct = total_train[correct_mask].mean() if correct_mask.sum() > 0 else 0
                cfr_wrong = total_train[wrong_mask].mean() if wrong_mask.sum() > 0 else 0
                print(f"    {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: "
                      f"n={m.sum()}  prec={target.mean():.3f}  "
                      f"cfr_correct={cfr_correct:.2f}  cfr_wrong={cfr_wrong:.2f}")

    summary = {
        "n_axes": len(AXES),
        "n_sigmas": len(sigmas),
        "feature_names": list(train_feats.columns),
        "train_cfr_total_mean": float(total_train.mean()),
        "test_cfr_total_mean": float(total_test.mean()),
        "boundary_row_pct_train": float((total_train >= 4).mean()),
        "boundary_row_pct_test": float((total_test >= 4).mean()),
    }
    with open(ART / "n3_cfr_features_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary")


if __name__ == "__main__":
    main()
