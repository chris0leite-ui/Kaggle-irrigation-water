"""Brainstorm #1: dedicated binary 'is High?' head, blended with hybrid.

Under balanced accuracy, High-class recall has 3x leverage (1/3 of the
score), so a model that specialises on P(High | x) may boost the
hybrid's High posterior. Same 43-feature dist set, same 5-fold
stratified split (on 3-class y, so fold indices match existing OOFs),
XGBoost binary:logistic. Then blend the binary head's P(High) into
the hybrid's High posterior and retune log-bias.

Artefacts:
  scripts/artifacts/oof_xgb_bin_high.npy
  scripts/artifacts/test_xgb_bin_high.npy
  scripts/artifacts/binary_high_head_results.json
  submissions/submission_hybrid_binhigh_tuned.csv (if lift > 0)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from benchmark_xgb_dist import (
    CLASSES, CLS2IDX, IDX2CLS, ID, TARGET,
    add_distance_features, tune_log_bias,
)


SEED = 42
N_FOLDS = 5
OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blend_high(p3: np.ndarray, phigh: np.ndarray, w: float) -> np.ndarray:
    """Mix the binary P(High) into 3-class probs' High column.

    blend = (1-w) * p3[:, 2] + w * phigh
    Low/Medium probs are rescaled so each row sums to 1.
    """
    new = p3.copy()
    new_high = np.clip((1 - w) * p3[:, 2] + w * phigh, 1e-9, 1 - 1e-9)
    denom = np.clip(1 - p3[:, 2], 1e-9, 1.0)
    scale = (1 - new_high) / denom
    new[:, 0] = p3[:, 0] * scale
    new[:, 1] = p3[:, 1] * scale
    new[:, 2] = new_high
    new /= new.sum(1, keepdims=True)
    return new


def geo_blend_high(p3: np.ndarray, phigh: np.ndarray, w: float) -> np.ndarray:
    """Geometric-mean blend on the High column, same rescale."""
    new = p3.copy()
    ph = np.clip(p3[:, 2], 1e-9, 1 - 1e-9) ** (1 - w) * np.clip(phigh, 1e-9, 1 - 1e-9) ** w
    ph = np.clip(ph, 1e-9, 1 - 1e-9)
    denom = np.clip(1 - p3[:, 2], 1e-9, 1.0)
    scale = (1 - ph) / denom
    new[:, 0] = p3[:, 0] * scale
    new[:, 1] = p3[:, 1] * scale
    new[:, 2] = ph
    new /= new.sum(1, keepdims=True)
    return new


def logit_add(p3: np.ndarray, phigh: np.ndarray, lam: float) -> np.ndarray:
    """Add lam * logit(phigh) to the High logit; renormalise via softmax."""
    logp = np.log(np.clip(p3, 1e-9, 1.0))
    lg = np.log(np.clip(phigh, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - phigh, 1e-9, 1.0))
    logp[:, 2] += lam * lg
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


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
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y3 = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    y_bin = (y3 == 2).astype(np.int32)
    prior = np.bincount(y3) / len(y3)
    pos_rate = y_bin.mean()
    log(f"features: {len(feat_cols)}; class 'High' rate = {pos_rate:.4f}")

    log("training 5-fold binary 'is High?' XGB")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_bin = np.zeros(len(tr), dtype=np.float64)
    test_bin = np.zeros(len(te), dtype=np.float64)

    xgb_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )

    dte = xgb.DMatrix(X_test, enable_categorical=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y3)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y_bin[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y_bin[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof_bin[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_bin += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        fold_auc = roc_auc_score(y_bin[va_idx], oof_bin[va_idx])
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  AUC={fold_auc:.5f}  ({time.time()-t0:.1f}s)")

    auc = roc_auc_score(y_bin, oof_bin)
    log(f"binary-head OOF AUC = {auc:.5f}")
    np.save(ART_DIR / "oof_xgb_bin_high.npy", oof_bin)
    np.save(ART_DIR / "test_xgb_bin_high.npy", test_bin)

    log("loading hybrid OOF + test probs")
    hyb_oof = np.load(ART_DIR / "oof_hybrid_lgbmxgb_blend.npy")
    hyb_test = np.load(ART_DIR / "test_hybrid_lgbmxgb_blend.npy")

    bias_base, tuned_base = tune_log_bias(hyb_oof, y3, prior)
    log(f"baseline hybrid tuned bal_acc = {tuned_base:.5f}")

    results = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "bin_auc_oof": float(auc),
        "bin_oof_mean": float(oof_bin.mean()),
        "bin_oof_pos_rate": float(pos_rate),
        "baseline_hybrid_tuned": float(tuned_base),
        "sweeps": {},
    }

    sweep_prob = []
    for w in np.linspace(0.0, 1.0, 21):
        blend = blend_high(hyb_oof, oof_bin, w)
        _, tuned = tune_log_bias(blend, y3, prior)
        sweep_prob.append({"w": float(w), "tuned": float(tuned)})
        log(f"  prob-mix   w={w:.2f}  tuned={tuned:.5f}")
    results["sweeps"]["prob_mix"] = sweep_prob

    sweep_geo = []
    for w in np.linspace(0.0, 1.0, 21):
        blend = geo_blend_high(hyb_oof, oof_bin, w)
        _, tuned = tune_log_bias(blend, y3, prior)
        sweep_geo.append({"w": float(w), "tuned": float(tuned)})
        log(f"  geo-mix    w={w:.2f}  tuned={tuned:.5f}")
    results["sweeps"]["geo_mix"] = sweep_geo

    sweep_lgt = []
    for lam in np.linspace(-1.0, 2.0, 31):
        blend = logit_add(hyb_oof, oof_bin, lam)
        _, tuned = tune_log_bias(blend, y3, prior)
        sweep_lgt.append({"lam": float(lam), "tuned": float(tuned)})
        log(f"  logit-add  lam={lam:+.2f}  tuned={tuned:.5f}")
    results["sweeps"]["logit_add"] = sweep_lgt

    best_prob = max(sweep_prob, key=lambda d: d["tuned"])
    best_geo = max(sweep_geo, key=lambda d: d["tuned"])
    best_lgt = max(sweep_lgt, key=lambda d: d["tuned"])
    log(f"best prob-mix  w={best_prob['w']:.2f}   tuned={best_prob['tuned']:.5f}")
    log(f"best geo-mix   w={best_geo['w']:.2f}    tuned={best_geo['tuned']:.5f}")
    log(f"best logit-add lam={best_lgt['lam']:+.2f}  tuned={best_lgt['tuned']:.5f}")

    candidates = [
        ("prob_mix", best_prob["w"], blend_high(hyb_oof, oof_bin, best_prob["w"]),
                                     blend_high(hyb_test, test_bin, best_prob["w"]),
                                     best_prob["tuned"]),
        ("geo_mix",  best_geo["w"],  geo_blend_high(hyb_oof, oof_bin, best_geo["w"]),
                                     geo_blend_high(hyb_test, test_bin, best_geo["w"]),
                                     best_geo["tuned"]),
        ("logit_add", best_lgt["lam"], logit_add(hyb_oof, oof_bin, best_lgt["lam"]),
                                       logit_add(hyb_test, test_bin, best_lgt["lam"]),
                                       best_lgt["tuned"]),
    ]
    name, param, best_oof, best_test, best_tuned = max(candidates, key=lambda c: c[4])
    results["best"] = {
        "kind": name,
        "param": float(param),
        "tuned_bal_acc": float(best_tuned),
        "delta_vs_hybrid": float(best_tuned - tuned_base),
    }
    log(f"best overall: {name} at {param:+.3f} tuned={best_tuned:.5f} "
        f"Delta={best_tuned - tuned_base:+.5f}")

    if best_tuned > tuned_base + 1e-5:
        bias, _ = tune_log_bias(best_oof, y3, prior)
        cm = confusion_matrix(
            y3,
            (np.log(np.clip(best_oof, 1e-9, 1.0)) + bias).argmax(axis=1),
        )
        log(f"OOF confusion matrix:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
        preds = (np.log(np.clip(best_test, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub = pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]})
        sub.to_csv(OUT_DIR / "submission_hybrid_binhigh_tuned.csv", index=False)
        log(f"wrote submission -> {OUT_DIR}/submission_hybrid_binhigh_tuned.csv")
        np.save(ART_DIR / "oof_hybrid_binhigh.npy", best_oof)
        np.save(ART_DIR / "test_hybrid_binhigh.npy", best_test)

    with open(ART_DIR / "binary_high_head_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART_DIR}/binary_high_head_results.json")


if __name__ == "__main__":
    main()
