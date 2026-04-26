"""Angle C — within-cell rule-disagreement mixup augmentation (dist base).

Mechanism:
  1. Compute rule_cell ∈ {0..63} for every train row (6-bit packing).
  2. For each cell, find pairs (i, j) where y_i != y_j (within-cell flips).
  3. Generate K mixup rows per pair via β(0.4, 0.4) on numerics; cats sampled
     from {x_i, x_j} weighted by α. Hard-target argmax with sample-weight =
     soft_max for confidence attenuation.
  4. Train XGB on 43-feature dist set (train + mixup augmentation).

Why fresh vs SMOTE-NC:
  SMOTE pairs k-NN nearest in feature space → diffused boundary (NULL).
  This pairs by rule-cell ∩ class-disagreement → mixed rows sit exactly
  on the within-cell flip surface where primary still loses macro-recall.

Scope: dist-feature base (not full recipe) for tractable wall time.
Mechanism proof-of-concept; if blend-gate passes, scale to recipe.

5-fold StratifiedKFold(seed=42). SMOKE=1 → 50k subsample, 2 folds.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402
from tier1b_helpers import ART, SEED, log  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
K_MIX = int(os.environ.get("K_MIX", "3"))
BETA_A = float(os.environ.get("BETA_A", "0.4"))
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

NUM_COLS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
            "Humidity", "Sunlight_Hours", "Field_Area_hectare",
            "Previous_Irrigation_mm"]
CAT_COLS = ["Region", "Crop_Type", "Soil_Type", "Crop_Growth_Stage",
            "Mulching_Used", "Season", "Irrigation_Type", "Water_Source"]
DIST_FEATS = ["sm_dist", "rf_dist", "tc_dist", "ws_dist",
              "sm_abs", "rf_abs", "tc_abs", "ws_abs",
              "dry", "norain", "hot", "windy", "nomulch", "kc_active",
              "dgp_score", "rule_pred",
              "score_dist_low_mid", "score_dist_mid_high",
              "min_boundary_dist", "min_axis_abs",
              "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc"] + NUM_COLS


def cell_id(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].values < 25).astype(np.int8)
    nor = (df["Rainfall_mm"].values < 300).astype(np.int8)
    hot = (df["Temperature_C"].values > 30).astype(np.int8)
    win = (df["Wind_Speed_kmh"].values > 10).astype(np.int8)
    nom = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stg = df["Crop_Growth_Stage"].astype(str).values
    kc = np.isin(stg, ("Flowering", "Vegetative")).astype(np.int8)
    return (dry | (nor << 1) | (hot << 2) | (win << 3) | (nom << 4) | (kc << 5)).astype(np.int8)


def build_pairs(train: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    cell = cell_id(train)
    pi_l, pj_l = [], []
    cap = 1000 if SMOKE else 4000
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        if len(idx) < 2:
            continue
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        ys = y[idx]
        i2 = idx[rng.permutation(len(idx))]
        mask = ys != y[i2]
        if mask.sum() == 0:
            continue
        pi_l.append(idx[mask]); pj_l.append(i2[mask])
    return np.concatenate(pi_l), np.concatenate(pj_l)


def make_mixup(train: pd.DataFrame, y: np.ndarray, pi: np.ndarray, pj: np.ndarray,
               rng: np.random.Generator, k: int):
    n = len(pi) * k
    alpha = rng.beta(BETA_A, BETA_A, size=n).astype(np.float32)
    pi_r = np.repeat(pi, k); pj_r = np.repeat(pj, k)
    mixed = pd.DataFrame()
    for c in NUM_COLS:
        a = train[c].values[pi_r].astype(np.float32)
        b = train[c].values[pj_r].astype(np.float32)
        mixed[c] = (1 - alpha) * a + alpha * b
    for c in CAT_COLS:
        sel = rng.random(n) < alpha
        mixed[c] = np.where(sel, train[c].values[pj_r], train[c].values[pi_r])
    onehot = np.eye(3, dtype=np.float32)
    soft = (1 - alpha)[:, None] * onehot[y[pi_r]] + alpha[:, None] * onehot[y[pj_r]]
    return mixed, soft, pi_r, pj_r


def main():
    t0 = time.time()
    log(f"angle C mixup. SMOKE={SMOKE}  K={K_MIX}  β={BETA_A}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        train = train.sample(50_000, random_state=SEED).reset_index(drop=True)
    y = train[TARGET].to_numpy()

    rng = np.random.default_rng(SEED)
    pi, pj = build_pairs(train, y, rng)
    log(f"  {len(pi):,} cross-class within-cell pairs")
    mixed, soft_y, pi_r, pj_r = make_mixup(train, y, pi, pj, rng, K_MIX)
    mix_y = soft_y.argmax(1).astype(np.int64)
    mix_w = soft_y.max(1).astype(np.float32)
    log(f"  {len(mixed):,} mixup rows generated")

    # Engineer dist features on train + test + mixed.
    train_eng = add_distance_features(train).copy()
    test_eng = add_distance_features(test).copy()
    mixed_eng = add_distance_features(mixed).copy()
    Xtr = train_eng[DIST_FEATS].astype(np.float32).to_numpy()
    Xte = test_eng[DIST_FEATS].astype(np.float32).to_numpy()
    Xmx = mixed_eng[DIST_FEATS].astype(np.float32).to_numpy()
    log(f"  dist features: {Xtr.shape[1]} cols")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    xgb_params = dict(
        n_estimators=200 if SMOKE else 1500, max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=30 if SMOKE else 100, verbosity=0,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), 1):
        tr_set = np.zeros(len(y), dtype=bool); tr_set[tr_idx] = True
        keep_mix = tr_set[pi_r] & tr_set[pj_r]
        n_keep = int(keep_mix.sum())
        log(f"=== fold {fold}/{N_FOLDS}  +{n_keep:,} mixup rows "
            f"({100*n_keep/max(1,len(keep_mix)):.1f}% kept) ===")
        Xa = np.concatenate([Xtr[tr_idx], Xmx[keep_mix]])
        ya = np.concatenate([y[tr_idx], mix_y[keep_mix]])
        wa = compute_sample_weight("balanced", ya).astype(np.float32)
        wa[len(tr_idx):] *= mix_w[keep_mix]
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(Xa, ya, sample_weight=wa,
              eval_set=[(Xtr[va_idx], y[va_idx])], verbose=False)
        oof[va_idx] = m.predict_proba(Xtr[va_idx]).astype(np.float32)
        test_pred += m.predict_proba(Xte).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={m.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(3).tolist()}")

    np.save(ART / "oof_angle_c_mixup.npy", oof)
    np.save(ART / "test_angle_c_mixup.npy", test_pred)
    out = dict(
        smoke=SMOKE, n_folds=N_FOLDS, k_mix=K_MIX, beta_a=BETA_A,
        n_pairs=int(len(pi)), n_mix=int(len(mixed)),
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
        wall_min=(time.time() - t0) / 60.0,
    )
    with open(ART / "angle_c_mixup_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote angle_c_mixup_results.json  wall={out['wall_min']:.1f} min")


if __name__ == "__main__":
    main()
