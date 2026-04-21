"""Noise-inversion head (brainstorm idea #3).

The synthetic DGP is `rule + boundary-band flip noise`. If we
assume test labels follow the same noise process as train, the
Bayes-optimal prediction is P(y_obs | x), which factorises as

    P(y_obs | x) = sum_{y_true} P(y_obs | y_true, x) * P(y_true | x)

Since the rule is deterministic, P(y_true | x) is a hard one-hot
on rule(x). That leaves the "noise head" P(y_obs | y_true=r, x)
— a per-rule-label classifier that only sees rows where the rule
predicts class r, and only needs to decide whether the true
label matches the rule or flipped to an adjacent class. This is
a much simpler sub-problem than the full 3-way multiclass.

Three heads:
  - r=0 (rule=Low): binary Low-vs-Medium on rule=Low rows.
  - r=1 (rule=Medium): 3-class on rule=Medium rows (flips go both
    ways — mostly into Low or High).
  - r=2 (rule=High): binary Medium-vs-High on rule=High rows.

Feature set: everything EXCEPT the 6 rule cols (since they're
highly correlated with the rule label that routes the head —
giving the head those cols would just let it re-learn the rule).
We keep the raw distance-to-threshold features so the head can
model "how far from the boundary is this row", which is where
flip probability lives.

5-fold stratified CV on y. For each fold, train the three heads
on tr_idx, predict on va_idx (each row's rule label picks the
head). Tune a single global log-bias on the combined OOF probs.
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


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
RULE_COLS = (
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
)
OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_score_series(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0)
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def rule_label(score: np.ndarray) -> np.ndarray:
    return np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int32)


def add_dist(df: pd.DataFrame) -> pd.DataFrame:
    df["_sm_dist"] = df["Soil_Moisture"].astype(float) - 25.0
    df["_rf_dist"] = df["Rainfall_mm"].astype(float) - 300.0
    df["_tc_dist"] = df["Temperature_C"].astype(float) - 30.0
    df["_ws_dist"] = df["Wind_Speed_kmh"].astype(float) - 10.0
    df["_sm_abs"] = df["_sm_dist"].abs()
    df["_rf_abs"] = df["_rf_dist"].abs()
    df["_tc_abs"] = df["_tc_dist"].abs()
    df["_ws_abs"] = df["_ws_dist"].abs()
    df["_min_abs"] = np.minimum.reduce(
        [df["_sm_abs"].values, df["_rf_abs"].values,
         df["_tc_abs"].values, df["_ws_abs"].values]
    )
    return df


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def train_head(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    allowed_classes: list[int],
    cat_cols: list[str],
) -> np.ndarray:
    remap = {c: i for i, c in enumerate(allowed_classes)}
    y_local = np.array([remap[int(y)] for y in y_tr], dtype=np.int32)
    n_classes_local = len(allowed_classes)
    if n_classes_local == 2:
        params = dict(
            objective="binary", metric="binary_logloss",
            learning_rate=0.05, num_leaves=127,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            min_data_in_leaf=100, verbose=-1, seed=SEED,
        )
        dtr = lgb.Dataset(X_tr, label=y_local, categorical_feature=cat_cols)
        model = lgb.train(params, dtr, num_boost_round=500)
        p1 = model.predict(X_va)
        local = np.stack([1.0 - p1, p1], axis=1)
    else:
        params = dict(
            objective="multiclass", num_class=n_classes_local,
            metric="multi_logloss", learning_rate=0.05, num_leaves=127,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            min_data_in_leaf=100, verbose=-1, seed=SEED,
        )
        dtr = lgb.Dataset(X_tr, label=y_local, categorical_feature=cat_cols)
        model = lgb.train(params, dtr, num_boost_round=500)
        local = model.predict(X_va)
    full = np.full((len(X_va), len(CLASSES)), 1e-6, dtype=np.float64)
    for li, gc in enumerate(allowed_classes):
        full[:, gc] = local[:, li]
    # normalise so rows sum to 1
    full = full / full.sum(axis=1, keepdims=True)
    return full


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    score_tr = dgp_score_series(tr)
    score_te = dgp_score_series(te)
    rule_tr = rule_label(score_tr)
    rule_te = rule_label(score_te)
    log(f"rule-label dist (train): {np.bincount(rule_tr)}")
    log(f"rule-label dist (test):  {np.bincount(rule_te)}")

    tr = add_dist(tr)
    te = add_dist(te)

    # features: everything except the 6 rule cols
    feat_cols = [c for c in tr.columns
                 if c not in RULE_COLS + (ID, TARGET)]
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(tr[c])]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    log(f"features ({len(feat_cols)}): {feat_cols}")
    log(f"cat cols: {cat_cols}")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # observed-class inventory per rule label
    head_spec: dict[int, list[int]] = {}
    for r in range(len(CLASSES)):
        m = rule_tr == r
        if m.sum() == 0:
            continue
        classes_in_head = sorted(set(y[m].tolist()))
        head_spec[r] = classes_in_head
        log(f"  rule={CLASSES[r]}  n={int(m.sum())}  observed classes={[CLASSES[c] for c in classes_in_head]}")

    log("running 5-fold stratified CV with per-rule-label heads")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    X_all = tr[feat_cols]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(tr, y)):
        t0 = time.time()
        rule_fold_tr = rule_tr[tr_idx]
        rule_fold_va = rule_tr[va_idx]
        for r, allowed in head_spec.items():
            m_tr = rule_fold_tr == r
            m_va = rule_fold_va == r
            if m_va.sum() == 0:
                continue
            if len(allowed) == 1:
                oof[va_idx[m_va], :] = 1e-6
                oof[va_idx[m_va], allowed[0]] = 1.0 - 2e-6
                continue
            X_tr_r = X_all.iloc[tr_idx][m_tr]
            X_va_r = X_all.iloc[va_idx][m_va]
            y_tr_r = y[tr_idx][m_tr]
            full = train_head(X_tr_r, y_tr_r, X_va_r, allowed, cat_cols)
            oof[va_idx[m_va]] = full
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  argmax_bal_acc={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    log("tuning log-bias on OOF")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}")
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    log("training full-train heads for test predictions")
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    X_te = te[feat_cols]
    for r, allowed in head_spec.items():
        m_all = rule_tr == r
        m_te = rule_te == r
        if m_te.sum() == 0:
            continue
        if len(allowed) == 1:
            test_pred[m_te, :] = 1e-6
            test_pred[m_te, allowed[0]] = 1.0 - 2e-6
            continue
        full = train_head(X_all[m_all], y[m_all], X_te[m_te], allowed, cat_cols)
        test_pred[m_te] = full

    print("\n=== noise-inversion head (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  prior-reweight       : {reweight_bal:.5f}")
    print(f"  tuned log-bias       : {tuned_bal:.5f}")

    np.save(ART_DIR / "oof_noise_inversion.npy", oof)
    np.save(ART_DIR / "test_noise_inversion.npy", test_pred)
    with open(ART_DIR / "noise_inversion_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "head_spec": {int(k): [int(c) for c in v] for k, v in head_spec.items()},
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_noise_inversion_tuned.csv", index=False
    )
    log(f"artifacts written to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
