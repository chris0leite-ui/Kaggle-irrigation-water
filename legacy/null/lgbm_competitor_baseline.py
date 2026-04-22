"""Reproduction of yunsuxiaozi's LGBM baseline (claimed CV 0.97943).

Source: https://www.kaggle.com/competitions/playground-series-s6e4/discussion
        (notebook "Lightgbm baseline and advanced", CV 0.97943)

Novel levers vs our pipeline:
  1. Digit-extraction FE: for each numeric col c and k in [-4, 4),
     (c // 10**k) % 10 as int8. Directly attacks host-NN input precision.
  2. Numeric rounding: coarse bucketing that suppresses overfitting
     while preserving the digit information.
  3. Low-frequency category collapse: levels with count<5 → shared default.
  4. Multi-class TargetEncoder applied to CATS + digit cols: 1-in, 3-out
     (P(Low|cat), P(Med|cat), P(High|cat)).
  5. Sample weights = inverse class freq at fit time (vs our log-bias
     tuning at inference — may compound).

Protocol-pinned: 5-fold StratifiedKFold(seed=42, shuffle=True) for OOF
alignment with all our saved OOFs. sklearn TargetEncoder uses its own
internal CV (cv=5) inside each fold's training to prevent leakage.

Baseline refs (OOF tuned bal_acc):
  LGBM-dist              0.97266
  XGB-dist               0.97304
  greedy + nonrule       0.97421  (LB 0.97352)
  yunsuxiaozi claimed    0.97943 (CV, unverified protocol)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fe_digits_and_round(df: pd.DataFrame, maxes: pd.Series) -> pd.DataFrame:
    out = df.copy()
    digit_cols: list[str] = []
    for c in NUMS:
        vals = out[c].astype(np.float64).values
        for k in range(-4, 4):
            col = f"{c}_digit{k}"
            out[col] = ((vals // (10.0 ** k)) % 10).astype(np.int8)
            digit_cols.append(col)
        m = maxes[c]
        if m < 10:
            out[c] = out[c].round(3)
        elif m < 100:
            out[c] = out[c].round(2)
        else:
            out[c] = out[c].round(1)
    return out, digit_cols


def collapse_low_freq(tr: pd.DataFrame, te: pd.DataFrame, cat_cols: list[str],
                      min_count: int = 5) -> None:
    """Map low-freq levels to a shared default index. Modifies in place."""
    for c in cat_cols:
        freq = tr[c].value_counts()
        mapping = {val: idx for idx, (val, _) in
                   enumerate(freq[freq >= min_count].items())}
        default = len(mapping)
        tr[c] = tr[c].map(lambda x: mapping.get(x, default)).astype(np.int32)
        te[c] = te[c].map(lambda x: mapping.get(x, default)).astype(np.int32)


def tune_log_bias(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, best


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log(f"train={len(tr):,}  test={len(te):,}")

    # Drop cols that are constant in test
    drop = [c for c in te.columns if te[c].nunique() == 1]
    if drop:
        tr = tr.drop(columns=drop)
        te = te.drop(columns=drop)
        log(f"dropped constant cols: {drop}")

    log("building digit + rounded features")
    maxes = tr[NUMS].max()
    tr, digit_cols = fe_digits_and_round(tr, maxes)
    te, _ = fe_digits_and_round(te, maxes)
    log(f"  {len(digit_cols)} digit cols ({len(NUMS)} nums x 8 positions)")

    log("collapsing low-freq cat levels (<5)")
    cat_all = CATS + digit_cols
    collapse_low_freq(tr, te, cat_all, min_count=5)
    for c in cat_all:
        log(f"  {c}: {tr[c].nunique()} levels (after collapse)")
        break  # just one sample line

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Sample weights = inverse class freq (competition metric is bal_acc)
    inv_freq = {cls: len(y) / (3 * np.sum(y == cls)) for cls in range(3)}
    sample_w = np.array([inv_freq[v] for v in y], dtype=np.float32)
    log(f"class weights: {inv_freq}")

    feat_nums = [c for c in NUMS]  # rounded floats
    feat_cats = cat_all             # int-mapped cats + digits
    log(f"feature set: {len(feat_cats)} cats + {len(feat_nums)} nums "
        f"(after multiclass TE: {3 * len(feat_cats) + len(feat_nums)} cols)")

    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    fold_bals = []
    best_iters = []

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(tr, y)):
        t0 = time.time()
        X_tr_cats = tr.iloc[tr_idx][feat_cats].values
        X_va_cats = tr.iloc[va_idx][feat_cats].values
        X_te_cats = te[feat_cats].values
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Multi-class TE: fit on train-fold with internal cv=5, transform all
        enc = TargetEncoder(
            target_type="multiclass", smooth="auto",
            cv=5, random_state=SEED,
        )
        X_tr_te = enc.fit_transform(X_tr_cats, y_tr).astype(np.float32)
        X_va_te = enc.transform(X_va_cats).astype(np.float32)
        X_te_te = enc.transform(X_te_cats).astype(np.float32)
        log(f"  fold {fold+1}: TE {X_tr_te.shape[1]} cols  "
            f"({time.time()-t0:.0f}s TE)")

        # Concat TE cols with rounded numerics
        X_tr_num = tr.iloc[tr_idx][feat_nums].values.astype(np.float32)
        X_va_num = tr.iloc[va_idx][feat_nums].values.astype(np.float32)
        X_te_num = te[feat_nums].values.astype(np.float32)
        X_tr_full = np.concatenate([X_tr_te, X_tr_num], axis=1)
        X_va_full = np.concatenate([X_va_te, X_va_num], axis=1)
        X_te_full = np.concatenate([X_te_te, X_te_num], axis=1)

        dtr = lgb.Dataset(X_tr_full, label=y_tr, weight=sample_w[tr_idx])
        dva = lgb.Dataset(X_va_full, label=y_va, weight=sample_w[va_idx])
        params = dict(
            objective="multiclass", num_class=3, metric="multi_logloss",
            learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            verbose=-1, seed=SEED,
        )
        model = lgb.train(
            params, dtr, num_boost_round=3000,
            valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        best_iters.append(int(model.best_iteration))
        oof[va_idx] = model.predict(X_va_full, num_iteration=model.best_iteration)
        test_probs += model.predict(X_te_full, num_iteration=model.best_iteration) / N_FOLDS
        bal = balanced_accuracy_score(y_va, oof[va_idx].argmax(axis=1))
        fold_bals.append(bal)
        log(f"  fold {fold+1}/{N_FOLDS}  iter={model.best_iteration}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.0f}s total)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    print(f"\n=== competitor LGBM reproduction ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  tuned log-bias       : {tuned_bal:.5f}")
    print(f"  yunsuxiaozi claimed  : 0.97943  (CV, protocol unverified)")
    print(f"  LGBM-dist baseline   : 0.97266")
    print(f"  greedy+nonrule (ours): 0.97421  (LB 0.97352)")
    print(f"  Δ vs claim           : {tuned_bal - 0.97943:+.5f}")
    print(f"  Δ vs greedy+nonrule  : {tuned_bal - 0.97421:+.5f}")
    print(f"  fold bals: {[f'{b:.5f}' for b in fold_bals]}")
    print(f"  fold std: {np.std(fold_bals):.5f}")
    print(f"  bias: {bias.round(3).tolist()}")
    print(f"  per-class recall: Low={per_class_recall[0]:.5f} "
          f"Medium={per_class_recall[1]:.5f} High={per_class_recall[2]:.5f}")
    print(f"  OOF confusion matrix:")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    np.save(ART / "oof_lgbm_competitor.npy", oof)
    np.save(ART / "test_lgbm_competitor.npy", test_probs)
    with open(ART / "lgbm_competitor_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "n_cat_cols": len(feat_cats),
            "n_num_cols": len(feat_nums),
            "n_te_cols": 3 * len(feat_cats),
            "total_cols_fed_to_lgbm": 3 * len(feat_cats) + len(feat_nums),
            "best_iters": best_iters,
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned_bal),
            "fold_bals": [float(x) for x in fold_bals],
            "fold_std": float(np.std(fold_bals)),
            "log_bias": bias.tolist(),
            "per_class_recall": per_class_recall.tolist(),
            "delta_vs_claim": float(tuned_bal - 0.97943),
            "delta_vs_greedy_nonrule": float(tuned_bal - 0.97421),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({
        ID: te[ID],
        TARGET: [IDX2CLS[i] for i in tuned_idx],
    }).to_csv(SUB / "submission_lgbm_competitor_tuned.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
