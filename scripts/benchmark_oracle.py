"""Cross-DGP oracle: use 10k original dataset as a flip-detector feature.

Hypothesis: the 10k original Irrigation Prediction dataset is clean
(the reverse-engineered rule gets 100.000 % on it, no noise). An LGBM
trained on it learns the rule as a smooth probability surface — which,
evaluated on synthetic rows, gives us a "clean-world prediction" that
serves as a rule-confidence feature. Where the oracle's argmax
disagrees with the synthetic label, the row is a high-specificity
flip candidate.

Crucially, the oracle is trained on rows NOT in synthetic train/test,
so no leakage. This is a materially different source of signal than
`flip_detector.py`, which was trained on synthetic's own
(rule, label) disagreement — meaning it could only model the flip
process from inside the same noisy distribution.

Features added to the standard LGBM+DGP feature set (benchmark_dgp.py):
  - oracle_p_low, oracle_p_medium, oracle_p_high   (continuous probs)
  - oracle_argmax                                  (int8 predicted class)
  - oracle_confidence                              (max prob)
  - oracle_entropy                                 (softmax entropy)
  - oracle_margin                                  (top1 - top2 prob)

Pipeline:
  1. Train LGBM on full 10k original (no CV here — original is OOD to
     synthetic folds, so this produces leak-free features).
  2. Predict probs on all 630k synthetic train + all 270k test.
  3. Concatenate oracle features with DGP-enriched features.
  4. 5-fold stratified CV on synthetic; log-bias coord ascent.

Baseline to beat: LGBM+DGP tuned OOF 0.97271.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import xlogy
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
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]), 2, 0,
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


log("loading synthetic train/test + original")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
orig = pd.read_csv("data/archive.zip")
log(f"synthetic train={len(tr)}  test={len(te)}  original={len(orig)}")

# Shared integer-encoding of categoricals across all three sources.
base_cat = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
for c in base_cat:
    vocab = sorted(set(tr[c].unique()) | set(te[c].unique()) | set(orig[c].unique()))
    mapping = {v: i for i, v in enumerate(vocab)}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")
    orig[c] = orig[c].map(mapping).astype("int32")


# ------------------------------------------------------------------ oracle ----
log("training LGBM oracle on 10k original (one-shot, no CV)")
num_cols_for_oracle = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
oracle_feats = num_cols_for_oracle + base_cat
X_orig = orig[oracle_feats].copy()
y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
log(f"  oracle train size: {len(X_orig)}  priors: "
    f"{(np.bincount(y_orig) / len(y_orig)).round(4).tolist()}")

oracle_params = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=20,
    verbose=-1,
    seed=SEED,
)
# Use a tiny internal CV on the 10k just to pick n_rounds; then retrain on
# all 10k with that many rounds so the final oracle is consistent.
log("  picking n_rounds via 5-fold CV on the 10k original")
skf_o = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
best_iters = []
for fold, (tr_idx, va_idx) in enumerate(skf_o.split(X_orig, y_orig)):
    dtr = lgb.Dataset(X_orig.iloc[tr_idx], label=y_orig[tr_idx], categorical_feature=base_cat)
    dva = lgb.Dataset(X_orig.iloc[va_idx], label=y_orig[va_idx],
                      categorical_feature=base_cat, reference=dtr)
    m = lgb.train(
        oracle_params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    best_iters.append(m.best_iteration)
final_rounds = int(np.mean(best_iters)) or 200
log(f"  avg best_iter on 10k = {np.mean(best_iters):.1f}  -> refit rounds={final_rounds}")

dfull = lgb.Dataset(X_orig, label=y_orig, categorical_feature=base_cat)
oracle = lgb.train(oracle_params, dfull, num_boost_round=final_rounds)

log("scoring oracle on synthetic train + test")
oracle_train = oracle.predict(tr[oracle_feats])
oracle_test = oracle.predict(te[oracle_feats])
log(f"  oracle train shape {oracle_train.shape}  test shape {oracle_test.shape}")

# Sanity check: oracle bal_acc on synthetic train (should match transfer check ~0.96)
orc_pred = oracle_train.argmax(axis=1)
y_syn = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
log(f"  oracle argmax bal_acc on synthetic train = "
    f"{balanced_accuracy_score(y_syn, orc_pred):.5f}")


def attach_oracle_cols(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["oracle_p_low"] = probs[:, 0]
    out["oracle_p_medium"] = probs[:, 1]
    out["oracle_p_high"] = probs[:, 2]
    out["oracle_argmax"] = probs.argmax(axis=1).astype(np.int8)
    out["oracle_confidence"] = probs.max(axis=1)
    # softmax entropy
    p = np.clip(probs, 1e-12, 1.0)
    out["oracle_entropy"] = -(xlogy(p, p)).sum(axis=1)
    sorted_probs = -np.sort(-p, axis=1)
    out["oracle_margin"] = sorted_probs[:, 0] - sorted_probs[:, 1]
    return out


log("building DGP + oracle feature set for main model")
tr = add_dgp_features(tr)
te = add_dgp_features(te)
tr = attach_oracle_cols(tr, oracle_train)
te = attach_oracle_cols(te, oracle_test)

# Define features for main model (same split logic as benchmark_dgp.py)
num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
feature_cols = num_cols + cat_cols
log(f"total features ({len(feature_cols)}): {feature_cols}")

X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


# ------------------------------------------------------------- main 5-fold ----
log("running 5-fold stratified LGBM on DGP + oracle features")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

params = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=127,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=200,
    verbose=-1,
    seed=SEED,
)

fold_bal = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                      categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=4000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    fb = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
    fold_bal.append(fb)
    log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fb:.5f}  ({time.time()-t0:.1f}s)")


def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y, pred_idx),
        "cm": confusion_matrix(y, pred_idx).tolist(),
    }


results = [
    bench("LGBM+DGP+ORACLE argmax", oof.argmax(axis=1)),
    bench("LGBM+DGP+ORACLE prior-reweight argmax", (oof / prior).argmax(axis=1)),
]

log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(b: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + b).argmax(axis=1))


bias = -np.log(prior)
best = score_bias(bias)
grid = np.linspace(-2.5, 2.5, 51)
for _ in range(20):
    improved = False
    for k in range(len(CLASSES)):
        base = bias.copy()
        scores = []
        for g in grid:
            base[k] = bias[k] + g
            scores.append(score_bias(base))
        j = int(np.argmax(scores))
        if scores[j] > best + 1e-6:
            bias[k] = bias[k] + grid[j]
            best = scores[j]
            improved = True
    if not improved:
        break
log(f"  best bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={best:.5f}")
results.append(bench("LGBM+DGP+ORACLE tuned log-bias", (log_oof + bias).argmax(axis=1)))

print("\n=== LGBM+DGP+ORACLE summary (OOF bal_acc, 5-fold) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")
print(f"  fold std (argmax) = {np.std(fold_bal):.5f}")

best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"\nbest rule: {best_rule['name']}")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))
print(f"\ndelta vs LGBM+DGP (0.97271): {best_rule['bal_acc'] - 0.97271:+.5f}")

np.save(ART_DIR / "oof_lgbm_dgp_oracle.npy", oof)
np.save(ART_DIR / "test_lgbm_dgp_oracle.npy", test_pred)
with open(ART_DIR / "bench_oracle_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "feature_cols": feature_cols,
            "fold_bal_argmax": fold_bal,
            "oracle_avg_best_iter": float(np.mean(best_iters)),
            "oracle_refit_rounds": final_rounds,
            "oracle_train_bal_acc_on_synth": float(balanced_accuracy_score(y_syn, orc_pred)),
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
            "delta_vs_lgbm_dgp": best_rule["bal_acc"] - 0.97271,
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {ART_DIR}/")

argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_oracle_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_oracle_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
