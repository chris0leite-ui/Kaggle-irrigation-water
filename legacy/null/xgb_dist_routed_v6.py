"""XGBoost-dist with rule-routing for {0, 1, 2, 5}.

Extends v3 ({0, 1, 2}) by adding score 5 to the routing set. Score 5
has 0.35% rule-error rate on 79k rows (rule predicts Medium, 99.65%
correct). Both heuristic conditions satisfied:
  - rule ≥ 99.5% on score
  - predicted class (Medium) is 38% of train, abundant in remainder

Unlike v3 which routes all rows to Low, v6 routes rows to DIFFERENT
classes depending on the score:
  - score in {0, 1, 2} -> Low  (rule predicts Low for score <= 3)
  - score in {5}       -> Medium  (rule predicts Medium for 4 <= score <= 6)

After routing: main XGB trains on ~283k rows concentrated on scores
{3, 4, 6, 7, 8, 9} — the error-dense bins.

Check:
  earlier v5 routing {0,1,2,5,9} was -0.00049 vs v1 {1,2} (0.97333).
  The null was attributed to routing 9 (strips 15% of High). This v6
  tests whether removing 9 from the set restores / improves lift.

Baseline: xgb_dist_routed_v3 (routes {0,1,2}) at OOF tuned 0.97332.
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
# route scores mapped to their rule class (Low=0, Medium=1, High=2)
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    """Per-row rule prob: one-hot at the rule's class for this score."""
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

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_mask = np.isin(te_scores, ROUTED_SCORES)
    log(f"train routed (scores in {ROUTED_SCORES}): "
        f"{tr_mask.sum()} / {len(tr)} ({tr_mask.mean()*100:.2f}%)")
    log(f"test  routed: {te_mask.sum()} / {len(te)}")

    # show per-score route mapping
    for s in sorted(ROUTED_SCORES):
        n = (tr_scores == s).sum()
        log(f"  score={s} -> {CLASSES[ROUTE_MAP[s]]}  (n_tr={n})")

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
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_pred_xgb = np.zeros((len(te), 3), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []

    tr_rule_probs = build_rule_probs(tr_scores)  # shape (len(tr), 3)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_filt = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        va_filt = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]
        dtr = xgb.DMatrix(X.iloc[tr_filt], label=y[tr_filt],
                          enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_filt], label=y[va_filt],
                          enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)
        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred_all = booster.predict(dva_full, iteration_range=(0, best_iter + 1))
        va_m = tr_mask[va_idx]
        oof[va_idx[~va_m]] = val_pred_all[~va_m]
        oof[va_idx[va_m]] = tr_rule_probs[va_idx[va_m]]
        test_pred_xgb += booster.predict(dte, iteration_range=(0, best_iter + 1)) / N_FOLDS
        fb = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={best_iter}  "
            f"dropped_tr={len(tr_idx)-len(tr_filt)} dropped_va={len(va_idx)-len(va_filt)}  "
            f"bal_all={fb:.5f}  ({time.time()-t0:.1f}s)")

    # route test
    te_rule_probs = build_rule_probs(te_scores)
    test_pred = test_pred_xgb.copy()
    test_pred[te_mask] = te_rule_probs[te_mask]

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))

    # rule accuracy per routed score (diagnostic)
    print("\n=== XGB-dist routed {0,1,2,5} (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  tuned log-bias       : {tuned:.5f}")
    print(f"  v3 (routed 0,1,2)    : 0.97332")
    print(f"  Δ vs v3              : {tuned - 0.97332:+.5f}")
    print(f"  bias                 : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # per-routed-score rule accuracy
    for s in sorted(ROUTED_SCORES):
        m = tr_scores == s
        rule_c = ROUTE_MAP[s]
        raw = (y[m] == rule_c).mean()
        log(f"  score={s} rule_class={CLASSES[rule_c]} "
            f"raw_acc={raw:.5f} (n={m.sum()})")

    np.save(ART_DIR / "oof_xgb_dist_routed_v6.npy", oof)
    np.save(ART_DIR / "test_xgb_dist_routed_v6.npy", test_pred)
    with open(ART_DIR / "xgb_dist_routed_v6_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "routed_scores": list(ROUTED_SCORES),
            "route_map": {str(k): CLASSES[v] for k, v in ROUTE_MAP.items()},
            "train_rows_routed": int(tr_mask.sum()),
            "test_rows_routed": int(te_mask.sum()),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "argmax_bal_acc": float(argmax_bal),
            "tuned_bal_acc": float(tuned),
            "delta_vs_v3": float(tuned - 0.97332),
            "log_bias": bias.tolist(),
        }, f, indent=2)
    tuned_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        OUT_DIR / "submission_xgb_dist_routed_v6_tuned.csv", index=False
    )
    log(f"artefacts saved")


if __name__ == "__main__":
    main()
