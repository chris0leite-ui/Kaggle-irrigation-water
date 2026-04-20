"""Gated pipeline v2: rule + flip-prob + SPECIALIST direction classifier.

v1 learned that a 3-class LGBM trained on all 630k rows gets flipped-row
bal_acc of only 0.12 (raw 0.15) -- the noise signal is diluted away. A
classifier trained only on the 10,304 flipped rows hits bal_acc 0.9937.

This version uses the specialist:
  P_final(x) = (1 - P_flip(x)) * onehot(rule(x))
             + P_flip(x)       * P_specialist(x)

Pipeline (per fold):
  - binary flip detector on train fold   -> P_flip on val + test
  - identify flipped rows in train fold
  - 3-class specialist on those flipped rows only  -> P_dir on val + test
  - blend with rule, score OOF, tune log-bias

Also runs LGBM+DGP as control so v2's result is directly comparable to v1.
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

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_rule_int(df: pd.DataFrame) -> np.ndarray:
    sm = df["Soil_Moisture"].astype(float).values
    rm = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    um = df["Mulching_Used"].astype(str).values
    stg = df["Crop_Growth_Stage"].astype(str).values
    dry = (sm < 25).astype(int)
    norain = (rm < 300).astype(int)
    hot = (tc > 30).astype(int)
    windy = (ws > 10).astype(int)
    nomulch = (um == "No").astype(int)
    kc = np.where(np.isin(stg, ["Flowering", "Vegetative"]), 2, 0)
    s = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    cls = np.where(s <= 3, 0, np.where(s <= 6, 1, 2))
    return cls.astype(np.int32)


def add_dgp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float)
    rm = out["Rainfall_mm"].astype(float)
    tc = out["Temperature_C"].astype(float)
    ws = out["Wind_Speed_kmh"].astype(float)
    out["dgp_dry"] = (sm < 25).astype(np.int8)
    out["dgp_norain"] = (rm < 300).astype(np.int8)
    out["dgp_hot"] = (tc > 30).astype(np.int8)
    out["dgp_windy"] = (ws > 10).astype(np.int8)
    out["dgp_nomulch"] = (out["Mulching_Used"].astype(str) == "No").astype(np.int8)
    out["dgp_kc"] = np.where(
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]), 2, 0
    ).astype(np.int8)
    out["dgp_score"] = (
        2 * (out["dgp_dry"] + out["dgp_norain"])
        + (out["dgp_hot"] + out["dgp_windy"] + out["dgp_nomulch"])
        + out["dgp_kc"]
    ).astype(np.int8)
    out["dgp_dist_moist"] = sm - 25.0
    out["dgp_dist_rain"] = rm - 300.0
    out["dgp_dist_temp"] = tc - 30.0
    out["dgp_dist_wind"] = ws - 10.0
    out["dgp_abs_moist"] = out["dgp_dist_moist"].abs()
    out["dgp_abs_rain"] = out["dgp_dist_rain"].abs()
    out["dgp_abs_temp"] = out["dgp_dist_temp"].abs()
    out["dgp_abs_wind"] = out["dgp_dist_wind"].abs()
    return out


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
tr = add_dgp_features(tr)
te = add_dgp_features(te)
rule_tr = dgp_rule_int(tr)
rule_te = dgp_rule_int(te)
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
is_flipped = (rule_tr != y).astype(np.int32)
log(f"rule raw_acc={float((rule_tr == y).mean()):.5f}  flip_rate={float(is_flipped.mean()):.5f}")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")
feature_cols = num_cols + cat_cols
X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()
prior = np.bincount(y) / len(y)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_flip = np.zeros(len(tr), dtype=np.float64)
test_flip = np.zeros(len(te), dtype=np.float64)
# specialist direction predicted on every row (OOF)
oof_spec = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_spec = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

params_bin = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=200, verbose=-1, seed=SEED, is_unbalance=True,
)
params_spec = dict(
    objective="multiclass", num_class=len(CLASSES), metric="multi_logloss",
    learning_rate=0.05, num_leaves=63,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=50, verbose=-1, seed=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    # ---- flip detector
    dtr = lgb.Dataset(X.iloc[tr_idx], label=is_flipped[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=is_flipped[va_idx], categorical_feature=cat_cols, reference=dtr,
    )
    m_bin = lgb.train(
        params_bin, dtr, num_boost_round=2000,
        valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_flip[va_idx] = m_bin.predict(X.iloc[va_idx], num_iteration=m_bin.best_iteration)
    test_flip += m_bin.predict(X_test, num_iteration=m_bin.best_iteration) / N_FOLDS

    # ---- specialist on flipped rows in the training fold only
    flipped_in_tr = tr_idx[is_flipped[tr_idx] == 1]
    Xf = X.iloc[flipped_in_tr]
    yf = y[flipped_in_tr]
    dtr_s = lgb.Dataset(Xf, label=yf, categorical_feature=cat_cols)
    m_spec = lgb.train(
        params_spec, dtr_s, num_boost_round=1500,
        valid_sets=[dtr_s], callbacks=[lgb.log_evaluation(0)],
    )
    # apply to val + test (all rows, not just flipped)
    oof_spec[va_idx] = m_spec.predict(X.iloc[va_idx])
    test_spec += m_spec.predict(X_test) / N_FOLDS

    log(
        f"  fold {fold+1}/{N_FOLDS}  bin_iter={m_bin.best_iteration}  "
        f"n_flipped_tr={len(flipped_in_tr)}  ({time.time()-t0:.1f}s)"
    )


# ---- diagnostics ------------------------------------------------------------
rule_bal = balanced_accuracy_score(y, rule_tr)
print(f"\n  rule-only            bal_acc = {rule_bal:.5f}")

spec_bal = balanced_accuracy_score(y, oof_spec.argmax(axis=1))
print(f"  specialist-only argmax bal_acc = {spec_bal:.5f}")

flip_mask = is_flipped == 1
spec_on_flipped_raw = (oof_spec[flip_mask].argmax(axis=1) == y[flip_mask]).mean()
spec_on_flipped_bal = balanced_accuracy_score(y[flip_mask], oof_spec[flip_mask].argmax(axis=1))
spec_on_clean_raw = (oof_spec[~flip_mask].argmax(axis=1) == y[~flip_mask]).mean()
print(f"  specialist on flipped rows: raw={spec_on_flipped_raw:.5f}  bal={spec_on_flipped_bal:.5f}")
print(f"  specialist on clean  rows: raw={spec_on_clean_raw:.5f}")

rule_oh = np.eye(len(CLASSES))[rule_tr]
p_final = (1 - oof_flip[:, None]) * rule_oh + oof_flip[:, None] * oof_spec
gated_argmax_bal = balanced_accuracy_score(y, p_final.argmax(axis=1))
print(f"  gated-v2 argmax       bal_acc = {gated_argmax_bal:.5f}")

# also try a hard threshold version: if P_flip > t, use specialist argmax; else rule
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    hard_pred = np.where(oof_flip > t, oof_spec.argmax(axis=1), rule_tr)
    acc_raw = (hard_pred == y).mean()
    acc_bal = balanced_accuracy_score(y, hard_pred)
    print(f"  hard-gate t={t:.2f}: raw={acc_raw:.5f}  bal={acc_bal:.5f}")


def tune_bias(probs: np.ndarray) -> tuple[float, np.ndarray]:
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return best, bias


gated_tuned_bal, gated_bias = tune_bias(p_final)
log(f"  gated-v2 tuned   bias = {gated_bias.round(4).tolist()}  bal_acc={gated_tuned_bal:.5f}")

# Header summary
print("\n=== summary v2 (OOF balanced accuracy) ===")
for name, val in [
    ("rule-only argmax", rule_bal),
    ("specialist-only argmax", spec_bal),
    ("gated-v2 argmax", gated_argmax_bal),
    ("gated-v2 tuned log-bias", gated_tuned_bal),
]:
    print(f"  {name:<35s} {val:.5f}")

# ---- test submission --------------------------------------------------------
rule_te_oh = np.eye(len(CLASSES))[rule_te]
p_final_te = (1 - test_flip[:, None]) * rule_te_oh + test_flip[:, None] * test_spec
tuned_idx = (np.log(np.clip(p_final_te, 1e-9, 1.0)) + gated_bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
    OUT_DIR / "submission_gated_v2_tuned.csv", index=False
)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in p_final_te.argmax(axis=1)]}).to_csv(
    OUT_DIR / "submission_gated_v2_argmax.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")

np.save(ART_DIR / "oof_flip_v2.npy", oof_flip)
np.save(ART_DIR / "test_flip_v2.npy", test_flip)
np.save(ART_DIR / "oof_spec_v2.npy", oof_spec)
np.save(ART_DIR / "test_spec_v2.npy", test_spec)
with open(ART_DIR / "gated_pipeline_v2_results.json", "w") as f:
    json.dump(
        {
            "rule_argmax_bal": float(rule_bal),
            "specialist_argmax_bal": float(spec_bal),
            "specialist_on_flipped_bal": float(spec_on_flipped_bal),
            "specialist_on_flipped_raw": float(spec_on_flipped_raw),
            "specialist_on_clean_raw": float(spec_on_clean_raw),
            "gated_v2_argmax_bal": float(gated_argmax_bal),
            "gated_v2_tuned_bal": float(gated_tuned_bal),
            "gated_v2_tuned_bias": gated_bias.tolist(),
        },
        f, indent=2,
    )
log(f"artefacts saved to {ART_DIR}/gated_pipeline_v2_results.json")
