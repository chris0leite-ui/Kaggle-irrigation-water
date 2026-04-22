"""Extended DGP feature engineering on top of benchmark_dgp.py.

benchmark_dgp.py established that adding the DGP rule indicators +
signed/abs distances to thresholds lifts tuned OOF bal_acc from
0.97097 (vanilla LGBM) to 0.97271 (+0.00174).

This script keeps all those features and adds a second layer aimed at
the boundary-band label noise:

  - dgp_is_boundary         score in {3, 4, 6, 7}
  - dgp_score_sq            non-monotone score effect
  - dgp_score_x_nomulch     interaction flagged in benchmark_dgp docstring
  - dgp_score_x_kc          "
  - dgp_sdist_*             sign(d) * d^2  (signed-squared distance, 4 axes)
  - dgp_min_abs_norm        nearest-axis distance in normalized units
  - dgp_n_axes_close        count of axes with |dist| / sigma < 0.5
  - dgp_score_if_flip_*     counterfactual score if this row's axis flipped
                            its indicator (4 axes)

Hypothesis: the label-flip noise concentrates near boundaries and is a
function of *how close* to the cut + *which cut is nearest*. Giving
the tree explicit boundary-distance geometry + the counterfactual
neighboring score may let it model the flip probability sharply.

Same 5-fold stratified pipeline / LGBM config as benchmark_dgp.py for
apples-to-apples OOF.
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


def add_dgp_features(df: pd.DataFrame, sigma: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float)
    rm = out["Rainfall_mm"].astype(float)
    tc = out["Temperature_C"].astype(float)
    ws = out["Wind_Speed_kmh"].astype(float)

    # --- base DGP features (same as benchmark_dgp.py) ---
    dry = (sm < 25).astype(np.int8)
    norain = (rm < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str) == "No").astype(np.int8)
    kc = np.where(
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]),
        2, 0,
    ).astype(np.int8)
    out["dgp_dry"] = dry
    out["dgp_norain"] = norain
    out["dgp_hot"] = hot
    out["dgp_windy"] = windy
    out["dgp_nomulch"] = nomulch
    out["dgp_kc"] = kc
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int16)
    out["dgp_score"] = score

    dm = sm - 25.0
    dr = rm - 300.0
    dt = tc - 30.0
    dw = ws - 10.0
    out["dgp_dist_moist"] = dm
    out["dgp_dist_rain"] = dr
    out["dgp_dist_temp"] = dt
    out["dgp_dist_wind"] = dw
    out["dgp_abs_moist"] = dm.abs()
    out["dgp_abs_rain"] = dr.abs()
    out["dgp_abs_temp"] = dt.abs()
    out["dgp_abs_wind"] = dw.abs()

    # --- extended FE (new) ---
    out["dgp_is_boundary"] = score.isin([3, 4, 6, 7]).astype(np.int8)
    out["dgp_score_sq"] = (score.astype(np.int32) ** 2).astype(np.int32)
    out["dgp_score_x_nomulch"] = (score * nomulch).astype(np.int16)
    out["dgp_score_x_kc"] = (score * kc).astype(np.int16)

    out["dgp_sdist_moist"] = np.sign(dm) * dm.abs() ** 2
    out["dgp_sdist_rain"] = np.sign(dr) * dr.abs() ** 2
    out["dgp_sdist_temp"] = np.sign(dt) * dt.abs() ** 2
    out["dgp_sdist_wind"] = np.sign(dw) * dw.abs() ** 2

    if sigma is None:
        sigma = {
            "moist": float(dm.abs().std()) or 1.0,
            "rain": float(dr.abs().std()) or 1.0,
            "temp": float(dt.abs().std()) or 1.0,
            "wind": float(dw.abs().std()) or 1.0,
        }
    nm = dm.abs() / sigma["moist"]
    nr = dr.abs() / sigma["rain"]
    nt = dt.abs() / sigma["temp"]
    nw = dw.abs() / sigma["wind"]
    out["dgp_norm_moist"] = nm
    out["dgp_norm_rain"] = nr
    out["dgp_norm_temp"] = nt
    out["dgp_norm_wind"] = nw
    out["dgp_min_abs_norm"] = np.minimum(np.minimum(nm, nr), np.minimum(nt, nw))
    out["dgp_n_axes_close"] = (
        (nm < 0.5).astype(np.int8)
        + (nr < 0.5).astype(np.int8)
        + (nt < 0.5).astype(np.int8)
        + (nw < 0.5).astype(np.int8)
    )

    # counterfactual: score if THIS axis indicator flipped (others stay)
    #   dry contributes 2*dry; flipping dry -> score ± 2 depending on current value
    out["dgp_score_if_flip_moist"] = (score + (1 - 2 * dry) * 2).astype(np.int16)
    out["dgp_score_if_flip_rain"] = (score + (1 - 2 * norain) * 2).astype(np.int16)
    out["dgp_score_if_flip_temp"] = (score + (1 - 2 * hot) * 1).astype(np.int16)
    out["dgp_score_if_flip_wind"] = (score + (1 - 2 * windy) * 1).astype(np.int16)

    return out, sigma


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

tr, sigma = add_dgp_features(tr, sigma=None)
te, _ = add_dgp_features(te, sigma=sigma)
log(f"sigma (train): {sigma}")

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")

feature_cols = num_cols + cat_cols
log(f"features ({len(feature_cols)}): {feature_cols}")

X = tr[feature_cols].copy()
X_test = te[feature_cols].copy()
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


log("running 5-fold stratified LGBM on extended DGP-enriched features")
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

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    t0 = time.time()
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(
        X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols, reference=dtr,
    )
    model = lgb.train(
        params,
        dtr,
        num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
    fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
    log(
        f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
        f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)"
    )


def bench(name: str, pred_idx: np.ndarray) -> dict:
    return {
        "name": name,
        "bal_acc": balanced_accuracy_score(y, pred_idx),
        "cm": confusion_matrix(y, pred_idx).tolist(),
    }


results = [
    bench("LGBM+DGP-FE2 argmax", oof.argmax(axis=1)),
    bench("LGBM+DGP-FE2 prior-reweight argmax", (oof / prior).argmax(axis=1)),
]

log("coord-ascent over per-class log-bias")
log_oof = np.log(np.clip(oof, 1e-9, 1.0))


def score_bias(bias: np.ndarray) -> float:
    return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))


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
results.append(bench("LGBM+DGP-FE2 tuned log-bias", (log_oof + bias).argmax(axis=1)))

print("\n=== LGBM+DGP-FE2 summary (OOF balanced accuracy) ===")
w = max(len(r["name"]) for r in results)
for r in results:
    print(f"  {r['name']:<{w}}  {r['bal_acc']:.5f}")

print("\nconfusion matrix (rows=true, cols=pred) for best rule:")
best_rule = max(results, key=lambda r: r["bal_acc"])
print(f"best: {best_rule['name']}")
print(pd.DataFrame(best_rule["cm"], index=CLASSES, columns=CLASSES))

np.save(ART_DIR / "oof_lgbm_dgp_fe2.npy", oof)
np.save(ART_DIR / "test_lgbm_dgp_fe2.npy", test_pred)
with open(ART_DIR / "bench_dgp_fe2_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "feature_cols": feature_cols,
            "sigma": sigma,
            "results": [{"name": r["name"], "bal_acc": r["bal_acc"]} for r in results],
            "best_rule": best_rule["name"],
        },
        f,
        indent=2,
    )
log(f"OOF + test probs saved to {ART_DIR}/")

argmax_test_idx = test_pred.argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_fe2_argmax.csv", index=False
)
tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_lgbm_dgp_fe2_tuned.csv", index=False
)
log(f"submissions written to {OUT_DIR}/")
