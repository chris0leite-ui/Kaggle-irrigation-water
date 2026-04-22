"""Gated two-stage pipeline: rule + flip-probability + direction classifier.

Motivation: the closed-form DGP rule is 100% on the original 10k dataset
and 98.364% on synthetic train. Of the 10,304 "flipped" rows, a binary
detector reaches OOF AUC 0.8993 and a 3-class classifier trained only on
those rows reaches OOF bal_acc 0.9937. Big residual signal lives outside
the rule.

This script trains, end-to-end inside 5-fold stratified CV, both:
  - P_flip(x)  = binary is_flipped model
  - P_dir(x)   = 3-class model on ALL rows (equivalent to LGBM+DGP)

and combines them as
  P_final(x) = (1 - P_flip(x)) * onehot(rule(x)) + P_flip(x) * P_dir(x)

Reports:
  1. LGBM+DGP OOF bal_acc (sanity vs stored artefact)
  2. LGBM+DGP bal_acc restricted to the 10,304 flipped rows
     (does it already near-match the specialist direction classifier?)
  3. Gated OOF bal_acc, both argmax and tuned log-bias
  4. Ablations:
        rule-only
        direction-only  (= LGBM+DGP)
        gated
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

# Rule predictions + is_flipped indicator on train
rule_tr = dgp_rule_int(tr)
rule_te = dgp_rule_int(te)
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
is_flipped = (rule_tr != y).astype(np.int32)
log(f"train rule raw_acc = {(rule_tr == y).mean():.5f}  flip_rate={is_flipped.mean():.5f}")

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
log(f"priors: {dict(zip(CLASSES, prior.round(4)))}")


# 5-fold CV: train BOTH models, save OOF + test preds ------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_flip = np.zeros(len(tr), dtype=np.float64)
test_flip = np.zeros(len(te), dtype=np.float64)
oof_dir = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_dir = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

params_bin = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=200, verbose=-1, seed=SEED, is_unbalance=True,
)
params_multi = dict(
    objective="multiclass", num_class=len(CLASSES), metric="multi_logloss",
    learning_rate=0.05, num_leaves=127,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    min_data_in_leaf=200, verbose=-1, seed=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    # flip detector
    dtr = lgb.Dataset(X.iloc[tr_idx], label=is_flipped[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=is_flipped[va_idx], categorical_feature=cat_cols, reference=dtr
    )
    m_bin = lgb.train(
        params_bin, dtr, num_boost_round=2000,
        valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_flip[va_idx] = m_bin.predict(X.iloc[va_idx], num_iteration=m_bin.best_iteration)
    test_flip += m_bin.predict(X_test, num_iteration=m_bin.best_iteration) / N_FOLDS

    # direction multi-class (all rows)
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols, reference=dtr
    )
    m_multi = lgb.train(
        params_multi, dtr, num_boost_round=4000,
        valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_dir[va_idx] = m_multi.predict(X.iloc[va_idx], num_iteration=m_multi.best_iteration)
    test_dir += m_multi.predict(X_test, num_iteration=m_multi.best_iteration) / N_FOLDS

    fold_bal = balanced_accuracy_score(y[va_idx], oof_dir[va_idx].argmax(axis=1))
    log(
        f"  fold {fold+1}/{N_FOLDS}  bin_iter={m_bin.best_iteration}  "
        f"multi_iter={m_multi.best_iteration}  dir_argmax_bal={fold_bal:.5f}  "
        f"({time.time()-t0:.1f}s)"
    )


# ----- diagnostics ----------------------------------------------------------
log("diagnostics:")
# (a) rule-only
rule_bal = balanced_accuracy_score(y, rule_tr)
print(f"  rule-only           bal_acc = {rule_bal:.5f}")

# (b) direction-only argmax (= LGBM+DGP repro)
dir_bal = balanced_accuracy_score(y, oof_dir.argmax(axis=1))
print(f"  direction-only argmax bal_acc = {dir_bal:.5f}")

# (c) direction-only, restricted to flipped rows  —  does LGBM+DGP
#     already near-match the specialist (0.9937)?
flip_mask = is_flipped == 1
dir_flip_bal = balanced_accuracy_score(y[flip_mask], oof_dir[flip_mask].argmax(axis=1))
dir_flip_raw = (oof_dir[flip_mask].argmax(axis=1) == y[flip_mask]).mean()
print(f"  direction-only ON flipped rows: bal={dir_flip_bal:.5f}  raw={dir_flip_raw:.5f}")
print(f"  direction-only ON clean rows:   raw={(oof_dir[~flip_mask].argmax(axis=1) == y[~flip_mask]).mean():.5f}")

# (d) gated blend
rule_oh = np.eye(len(CLASSES))[rule_tr]
p_final = (1 - oof_flip[:, None]) * rule_oh + oof_flip[:, None] * oof_dir
gated_bal_argmax = balanced_accuracy_score(y, p_final.argmax(axis=1))
print(f"  gated argmax         bal_acc = {gated_bal_argmax:.5f}")


# coord-ascent log-bias on each candidate
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


dir_tuned_bal, dir_bias = tune_bias(oof_dir)
gated_tuned_bal, gated_bias = tune_bias(p_final)
log(f"  direction-only tuned bias = {dir_bias.round(4).tolist()}  bal_acc={dir_tuned_bal:.5f}")
log(f"  gated tuned         bias = {gated_bias.round(4).tolist()}  bal_acc={gated_tuned_bal:.5f}")


print("\n=== summary (OOF balanced accuracy) ===")
for name, val in [
    ("rule-only argmax", rule_bal),
    ("direction-only argmax", dir_bal),
    ("direction-only tuned log-bias", dir_tuned_bal),
    ("gated argmax", gated_bal_argmax),
    ("gated tuned log-bias", gated_tuned_bal),
]:
    print(f"  {name:<35s} {val:.5f}")


# ----- build test-set submission for gated+tuned ----------------------------
rule_te_oh = np.eye(len(CLASSES))[rule_te]
p_final_te = (1 - test_flip[:, None]) * rule_te_oh + test_flip[:, None] * test_dir
tuned_idx = (np.log(np.clip(p_final_te, 1e-9, 1.0)) + gated_bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
    OUT_DIR / "submission_gated_tuned.csv", index=False
)
# also argmax version
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in p_final_te.argmax(axis=1)]}).to_csv(
    OUT_DIR / "submission_gated_argmax.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")

np.save(ART_DIR / "oof_flip_gated.npy", oof_flip)
np.save(ART_DIR / "test_flip_gated.npy", test_flip)
np.save(ART_DIR / "oof_dir_gated.npy", oof_dir)
np.save(ART_DIR / "test_dir_gated.npy", test_dir)
with open(ART_DIR / "gated_pipeline_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED, "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "rule_argmax_bal": float(rule_bal),
            "direction_argmax_bal": float(dir_bal),
            "direction_tuned_bal": float(dir_tuned_bal),
            "direction_tuned_bias": dir_bias.tolist(),
            "direction_on_flipped_rows_bal": float(dir_flip_bal),
            "direction_on_flipped_rows_raw": float(dir_flip_raw),
            "gated_argmax_bal": float(gated_bal_argmax),
            "gated_tuned_bal": float(gated_tuned_bal),
            "gated_tuned_bias": gated_bias.tolist(),
        },
        f, indent=2,
    )
log(f"artefacts saved to {ART_DIR}/gated_pipeline_results.json")
