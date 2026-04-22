"""LGBM with Generalized Cross-Entropy loss (brainstorm idea #5).

GCE (Zhang & Sabuncu 2018) bounds the contribution of any single
example, making training robust to label noise. For soft-max
probability p_t of the true class t and a parameter q ∈ (0, 1]:

    L = (1 - p_t^q) / q

    q → 0  ⇒ CE (sensitive to noisy labels)
    q = 1  ⇒ MAE (very noise-robust but hard to optimise)

We pick q = 0.7 as a moderate default. The ~2 % boundary-band
flips in synthetic train are precisely the case GCE was designed
for — we hope it reduces overfitting to the flipped rows and
lifts OOF on the (cleaner) validation folds.

Pipeline mirrors scripts/benchmark_dist.py. LGBM custom multiclass
objective, same 5-fold stratified CV, same log-bias tuning on top.
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
K = len(CLASSES)
Q = 0.7
OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    out["dry"] = (sm < 25).astype(np.int8)
    out["norain"] = (rf < 300).astype(np.int8)
    out["hot"] = (tc > 30).astype(np.int8)
    out["windy"] = (ws > 10).astype(np.int8)
    out["nomulch"] = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    kc = np.where(
        np.isin(out["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0
    ).astype(np.int8)
    out["kc"] = kc
    out["dgp_score"] = (
        2 * (out["dry"].values + out["norain"].values)
        + (out["hot"].values + out["windy"].values + out["nomulch"].values)
        + out["kc"].values
    ).astype(np.int8)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def make_gce_objective(q: float = Q):
    """Return (obj_fn, metric_fn) implementing GCE multiclass.

    LGBM multiclass custom obj: preds is flat (n*K,), class-major
    (first n = class 0 logits, next n = class 1, etc). Returns
    grad, hess in the same flat layout.
    """
    def fobj(preds: np.ndarray, dtrain: lgb.Dataset):
        y = dtrain.get_label().astype(int)
        n = len(y)
        # LGBM multiclass preds layout: (K*n,) class-major.
        z = preds.reshape(K, n).T  # (n, K)
        p = softmax(z)
        p_t = p[np.arange(n), y]
        p_t_q = np.power(np.clip(p_t, 1e-9, 1.0), q)

        one_hot = np.zeros((n, K), dtype=np.float64)
        one_hot[np.arange(n), y] = 1.0

        # dL/dz_k = -p_t^q * (1{k=t} - p_k)  (minimise L; gradient for
        #  LGBM is dL/dz)
        grad = -p_t_q[:, None] * (one_hot - p)
        # Hessian positive approximation: scale CE hess by p_t^q. Keeps
        # LGBM Newton step well-conditioned even near p_t → 0.
        hess = p_t_q[:, None] * (p * (1.0 - p))
        hess = np.maximum(hess, 1e-6)

        # flatten back to class-major
        return grad.T.ravel(), hess.T.ravel()

    def feval(preds: np.ndarray, dtrain: lgb.Dataset):
        y = dtrain.get_label().astype(int)
        n = len(y)
        z = preds.reshape(K, n).T
        p = softmax(z)
        p_t = p[np.arange(n), y]
        loss = ((1.0 - np.power(np.clip(p_t, 1e-9, 1.0), q)) / q).mean()
        return "gce", float(loss), False

    return fobj, feval


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(K):
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


def main() -> None:
    log(f"loading data (GCE q={Q})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

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

    fobj, feval = make_gce_objective(Q)

    log("running 5-fold stratified LGBM with GCE objective")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    params = dict(
        objective=fobj,
        num_class=K,
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
            num_boost_round=600,
            valid_sets=[dva],
            feval=feval,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        # LGBM with custom obj returns raw logits from model.predict()
        # (since it doesn't know how to softmax). We softmax manually.
        raw_va = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        raw_te = model.predict(X_test, num_iteration=model.best_iteration)
        # raw shape: (n, K) already for multiclass custom obj
        oof[va_idx] = softmax(raw_va)
        test_pred += softmax(raw_te) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"bal_acc(argmax)={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    log("coord-ascent over per-class log-bias")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={tuned_bal:.5f}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== LGBM + GCE loss (OOF bal_acc) ===")
    print(f"  q = {Q}")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  prior-reweight       : {reweight_bal:.5f}")
    print(f"  tuned log-bias       : {tuned_bal:.5f}")

    np.save(ART_DIR / "oof_lgbm_gce.npy", oof)
    np.save(ART_DIR / "test_lgbm_gce.npy", test_pred)
    with open(ART_DIR / "lgbm_gce_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "q": Q,
            "n_features": len(feat_cols),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_lgbm_gce_tuned.csv", index=False
    )
    log(f"artifacts written to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
