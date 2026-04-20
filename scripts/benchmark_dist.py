"""LGBM + distance-to-threshold features (brainstorm idea #2).

Adds signed and absolute distances to each DGP threshold
(Soil_Moisture=25, Rainfall_mm=300, Temperature_C=30,
Wind_Speed_kmh=10), their signs, and a handful of boundary-aware
interactions. Adds the reverse-engineered DGP `score` and `rule_pred`
columns too — these let the booster condition on "how close to the
class boundary is this row?", which is where the synthetic label
noise lives.

Hypothesis: the baseline LGBM has already discovered the threshold
rule, but it has no direct access to *distance-to-boundary*, so it
cannot learn that the flip probability scales with that distance.
Giving the model explicit distance features should let it model the
noise gradient, lifting OOF above 0.97097.

Pipeline mirrors scripts/benchmark.py: 5-fold stratified, LGBM
multiclass, log-bias coord-ascent, save OOF + test probs + tuned
submission.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

OUT_DIR = Path("submissions")
OOF_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append signed/abs distance-to-threshold features and DGP score-related cols."""
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    # signed distances (positive = above threshold)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)

    # absolute distances (smaller = more ambiguous)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)

    # binary indicator cols (trees benefit from explicit splits near thresholds)
    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    # DGP score + rule prediction (ordinal 0..10 score, 0..2 label)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    # distance to the nearest score-band boundary (3/4 and 6/7)
    #   score=3 and score=4 are symmetric around 3.5; likewise 6/7 around 6.5.
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)

    # minimum-axis distance: how "safe" is this row across ALL four continuous thresholds
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)

    # a handful of pairwise interaction terms
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)

    return out


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} ({len(num_cols)} numeric + {len(cat_cols)} categorical)")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    log("running 5-fold stratified LGBM")
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
        dtr = lgb.Dataset(
            X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_cols
        )
        dva = lgb.Dataset(
            X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_cols,
            reference=dtr,
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
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))

    log("coord-ascent over per-class log-bias")
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
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
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={best:.5f}")

    cm = confusion_matrix(y, (log_oof + bias).argmax(axis=1))
    log(f"OOF confusion matrix (rows=true, cols=pred):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== LGBM + distance-to-threshold features (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  prior-reweight       : {reweight_bal:.5f}")
    print(f"  tuned log-bias       : {best:.5f}")

    np.save(OOF_DIR / "oof_lgbm_dist.npy", oof)
    np.save(OOF_DIR / "test_lgbm_dist.npy", test_pred)
    with open(OOF_DIR / "bench_dist_results.json", "w") as f:
        json.dump(
            {
                "seed": SEED,
                "n_folds": N_FOLDS,
                "n_features": len(feat_cols),
                "class_priors": prior.tolist(),
                "log_bias": bias.tolist(),
                "argmax_bal_acc": float(argmax_bal),
                "reweight_bal_acc": float(reweight_bal),
                "tuned_bal_acc": float(best),
            },
            f,
            indent=2,
        )

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_lgbm_dist_tuned.csv", index=False
    )
    argmax_test_idx = test_pred.argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in argmax_test_idx]}).to_csv(
        OUT_DIR / "submission_lgbm_dist_argmax.csv", index=False
    )
    log(f"OOF + test probs saved to {OOF_DIR}/; submissions to {OUT_DIR}/")


if __name__ == "__main__":
    main()
