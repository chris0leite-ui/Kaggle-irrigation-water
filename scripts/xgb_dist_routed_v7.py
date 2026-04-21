"""XGBoost-dist with decoupled routing: train on ALL rows, route at inference.

Tests the anchor-training hypothesis from the v6 null:
  v3 {0,1,2}        train excludes {0,1,2}, infer routes {0,1,2}   OOF 0.97332
  v6 {0,1,2,5}      train excludes {0,1,2,5}, infer routes same    OOF 0.97320 (null)
  v7 {0,1,2,5}      train on ALL 630k, infer routes {0,1,2,5}      <-- this

If v7 > v3, the v6 loss was pure training-anchor removal (score 5
Medium rows were structural for the Medium<->High boundary on {6,7,8}).
In that case decoupled routing is a new lever: deterministic rule at
inference without the data-stripping cost.

Pipeline mirrors benchmark_xgb_dist.py (no routing in training) +
per-row rule override at inference for scores in ROUTED_SCORES.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from xgb_specialist_678 import add_distance_features


SEED = 42
N_FOLDS = 5
# score -> rule class mapping (Low=0, Med=1, High=2)
ROUTE_MAP = {0: 0, 1: 0, 2: 0, 5: 1}
ROUTED_SCORES = tuple(sorted(ROUTE_MAP.keys()))
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
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


def build_rule_probs(scores: np.ndarray) -> np.ndarray:
    out = np.zeros((len(scores), 3), dtype=np.float64)
    for i, s in enumerate(scores):
        c = ROUTE_MAP.get(int(s), None)
        if c is not None:
            out[i, c] = 1.0 - 2e-9
            for k in range(3):
                if k != c:
                    out[i, k] = 1e-9
    return out


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log("dist features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_mask = np.isin(te_scores, ROUTED_SCORES)
    log(f"train mask (inference-only route): {tr_mask.sum()} / {len(tr)}")
    log(f"test  mask: {te_mask.sum()} / {len(te)}")
    log(f"routed scores -> class: "
        f"{ {s: CLASSES[c] for s, c in ROUTE_MAP.items()} }")

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

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)}  priors: {dict(zip(CLASSES, prior.round(4)))}")

    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", learning_rate=0.05,
        max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", enable_categorical=True,
        verbosity=0, seed=SEED,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_xgb = np.zeros((len(tr), 3), dtype=np.float64)  # vanilla XGB OOF
    test_pred_xgb = np.zeros((len(te), 3), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        # TRAIN ON ALL rows — no score filtering
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        oof_xgb[va_idx] = booster.predict(dva, iteration_range=(0, best_iter + 1))
        test_pred_xgb += booster.predict(dte, iteration_range=(0, best_iter + 1)) / N_FOLDS
        fb_xgb = balanced_accuracy_score(y[va_idx], oof_xgb[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={best_iter}  "
            f"bal(xgb)={fb_xgb:.5f}  ({time.time()-t0:.1f}s)")

    # Build routed OOF: overlay rule onto ROUTED_SCORES rows at inference
    tr_rule_probs = build_rule_probs(tr_scores)
    te_rule_probs = build_rule_probs(te_scores)
    oof_routed = oof_xgb.copy()
    oof_routed[tr_mask] = tr_rule_probs[tr_mask]
    test_pred = test_pred_xgb.copy()
    test_pred[te_mask] = te_rule_probs[te_mask]

    # Evaluate both: vanilla XGB (train-on-all, no inference routing) and routed
    argmax_xgb = balanced_accuracy_score(y, oof_xgb.argmax(axis=1))
    bias_xgb, tuned_xgb = tune_log_bias(oof_xgb, y, prior)
    argmax_rt = balanced_accuracy_score(y, oof_routed.argmax(axis=1))
    bias_rt, tuned_rt = tune_log_bias(oof_routed, y, prior)

    cm_rt = confusion_matrix(
        y, (np.log(np.clip(oof_routed, 1e-9, 1.0)) + bias_rt).argmax(axis=1))

    print(f"\n=== v7 decoupled-routing {ROUTED_SCORES} ===")
    print(f"  vanilla XGB (no route):        tuned {tuned_xgb:.5f}")
    print(f"  v7 routed at inference only:   tuned {tuned_rt:.5f}")
    print(f"  v3 reference (train+infer rt): 0.97332")
    print(f"  v6 reference (train+infer {{0,1,2,5}}): 0.97320")
    print(f"  Δ v7 vs v3: {tuned_rt - 0.97332:+.5f}")
    print(f"  Δ v7 vs v6: {tuned_rt - 0.97320:+.5f}")
    print(f"  bias_routed = {bias_rt.round(3).tolist()}")
    print(f"  OOF confusion matrix (routed):\n"
          f"{pd.DataFrame(cm_rt, index=CLASSES, columns=CLASSES)}")

    np.save(ART_DIR / "oof_xgb_vanilla_dist.npy", oof_xgb)
    np.save(ART_DIR / "test_xgb_vanilla_dist.npy", test_pred_xgb)
    np.save(ART_DIR / "oof_xgb_dist_routed_v7.npy", oof_routed)
    np.save(ART_DIR / "test_xgb_dist_routed_v7.npy", test_pred)
    with open(ART_DIR / "xgb_dist_routed_v7_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "routed_scores": list(ROUTED_SCORES),
            "route_map": {str(k): CLASSES[v] for k, v in ROUTE_MAP.items()},
            "tr_routed_at_inference": int(tr_mask.sum()),
            "te_routed_at_inference": int(te_mask.sum()),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "vanilla_xgb_tuned": float(tuned_xgb),
            "v7_routed_tuned": float(tuned_rt),
            "delta_vs_v3": float(tuned_rt - 0.97332),
            "delta_vs_v6": float(tuned_rt - 0.97320),
            "log_bias_routed": bias_rt.tolist(),
        }, f, indent=2)
    tuned_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias_rt).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        OUT_DIR / "submission_xgb_dist_routed_v7_tuned.csv", index=False)
    log(f"artefacts saved (vanilla + routed)")


if __name__ == "__main__":
    main()
