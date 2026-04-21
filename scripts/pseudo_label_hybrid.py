"""Pseudo-labeling on test using current-best hybrid predictions.

Pipeline:
  1. Load test probabilities from the current-best hybrid
     (routed-{0,1,2} main XGB + spec-{6,7,8}, OOF 0.97352).
  2. Apply saved tuned log-bias to get per-row argmax labels + max_prob.
  3. Select test rows with max_prob > TAU (default 0.95).
  4. Append those pseudo-labeled rows to training data.
  5. Retrain main routed-v3 XGB + spec-{6,7,8} XGB on augmented training.
  6. Rebuild hybrid OOF (from pseudo-augmented main + spec) — OOF eval
     still on ORIGINAL train only (pseudo rows never enter val folds).
  7. Compare to baseline hybrid OOF 0.97352.

Key design point: OOF integrity.
  - Each fold's training set = (original train rows in fold) + (all
    pseudo-labeled test rows)
  - Each fold's val set = (original train rows held out)
  - Test rows never appear in val (they're test, distinct IDs).
  - Pseudo labels come from a model trained on the ORIGINAL synthetic
    labels. Their OOF estimate for test-rows is 0.973 accurate, so
    using them as training is essentially free-but-noisy data
    augmentation.

Confidence threshold choice:
  - tau=0.95 captures ~80-85% of test rows (~220k) at ~98%+ accuracy
  - tau=0.90 captures ~90% (~245k) at ~97% accuracy
  - tau=0.99 captures ~50% (~135k) at ~99.5% accuracy
Default tau=0.95 for baseline.
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
TAU = 0.95
ROUTED_SCORES = (0, 1, 2)
SPEC_SCORES = (6, 7, 8)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
SUB = Path("submissions")


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


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log("dist features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values

    # 1. Build pseudo-labels from saved hybrid test probs
    log(f"loading hybrid test probs and deriving pseudo-labels (τ={TAU})")
    test_main = np.load(ART / "test_xgb_dist_routed_v3.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")
    te_spec_mask = np.isin(te_scores, SPEC_SCORES)
    test_hybrid = test_main.copy()
    test_hybrid[te_spec_mask] = test_spec[te_spec_mask]

    # Use the bias saved from hybrid_routed_spec_aug hybrid_base fit
    # (rebuild quickly: tune on saved OOF)
    oof_main = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    tr_spec_mask = np.isin(tr_scores, SPEC_SCORES)
    oof_hybrid = oof_main.copy()
    oof_hybrid[tr_spec_mask] = oof_spec[tr_spec_mask]

    y_tr = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y_tr) / len(y_tr)
    bias, ref_bal = tune_log_bias(oof_hybrid, y_tr, prior)
    log(f"hybrid OOF bal (reference): {ref_bal:.5f}")
    log(f"saved hybrid log-bias: {bias.round(3).tolist()}")

    # Apply bias to TEST
    log_test = np.log(np.clip(test_hybrid, 1e-9, 1.0))
    test_biased_probs = np.exp(log_test + bias)
    test_biased_probs /= test_biased_probs.sum(axis=1, keepdims=True)
    test_pred = test_biased_probs.argmax(axis=1)
    test_max = test_biased_probs.max(axis=1)

    mask_high_conf = test_max > TAU
    n_pseudo = int(mask_high_conf.sum())
    log(f"pseudo-labeled test rows (max_prob > {TAU}): {n_pseudo} / {len(te)} "
        f"({100*n_pseudo/len(te):.1f}%)")
    dist = np.bincount(test_pred[mask_high_conf], minlength=3)
    log(f"pseudo-label class distribution: {dict(zip(CLASSES, dist.tolist()))}")

    # 2. Build augmented training: original train + pseudo rows
    pseudo_labels = test_pred[mask_high_conf].astype(np.int32)

    # Prepare feature set the same way as v3 / spec-678
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

    pseudo_X = X_test.iloc[mask_high_conf].copy()
    pseudo_y = pseudo_labels
    pseudo_scores = te_scores[mask_high_conf]
    for c in cat_cols:
        pseudo_X[c] = pseudo_X[c].astype("category")

    # 3. Retrain MAIN routed-v3 with augmented training
    log(f"retraining main routed-v3 with +{n_pseudo} pseudo-labeled rows")
    xgb_params = dict(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", learning_rate=0.05,
        max_depth=7, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", enable_categorical=True,
        verbosity=0, seed=SEED,
    )
    oof_main_new = np.zeros((len(tr), 3), dtype=np.float64)
    test_main_new = np.zeros((len(te), 3), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_rule_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_rule_mask = np.isin(te_scores, ROUTED_SCORES)
    rule_prob_low = np.array([1 - 2e-9, 1e-9, 1e-9], dtype=np.float64)
    best_iters_main = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_tr)):
        t0 = time.time()
        # Training: original train (non-routed) + all non-routed pseudo rows
        tr_filt = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        pseudo_nonroute_mask = ~np.isin(pseudo_scores, ROUTED_SCORES)

        X_tr_aug = pd.concat([X.iloc[tr_filt],
                              pseudo_X.iloc[pseudo_nonroute_mask]],
                             ignore_index=True)
        for c in cat_cols:
            X_tr_aug[c] = X_tr_aug[c].astype("category")
        y_tr_aug = np.concatenate(
            [y_tr[tr_filt], pseudo_y[pseudo_nonroute_mask]]).astype(np.int32)
        va_filt = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]

        dtr = xgb.DMatrix(X_tr_aug, label=y_tr_aug, enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_filt], label=y_tr[va_filt], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters_main.append(bi)

        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred = booster.predict(dva_full, iteration_range=(0, bi + 1))
        vm = tr_rule_mask[va_idx]
        oof_main_new[va_idx[~vm]] = val_pred[~vm]
        oof_main_new[va_idx[vm]] = rule_prob_low
        test_main_new += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        log(f"  main fold {fold+1}  n_tr_aug={len(y_tr_aug)}  "
            f"best_iter={bi}  ({time.time()-t0:.1f}s)")
    test_main_new[te_rule_mask] = rule_prob_low

    # 4. Retrain SPEC-{6,7,8} with augmented {6,7,8} training
    log(f"retraining spec-{{6,7,8}} with +{int((pseudo_scores >= 6) & (pseudo_scores <= 8)).sum()} pseudo-spec rows")
    pseudo_spec_mask = np.isin(pseudo_scores, SPEC_SCORES)
    log(f"  pseudo rows in {{6,7,8}}: {int(pseudo_spec_mask.sum())}")
    oof_spec_new = np.zeros((len(tr), 3), dtype=np.float64)
    test_spec_new = np.zeros((len(te), 3), dtype=np.float64)
    dte_spec = xgb.DMatrix(X_test.iloc[te_spec_mask], enable_categorical=True)
    best_iters_spec = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_tr)):
        t0 = time.time()
        tr_spec = tr_idx[np.isin(tr_scores[tr_idx], SPEC_SCORES)]
        va_spec = va_idx[np.isin(tr_scores[va_idx], SPEC_SCORES)]
        if len(tr_spec) == 0 or len(va_spec) == 0:
            continue

        X_tr_spec = pd.concat([X.iloc[tr_spec],
                               pseudo_X.iloc[pseudo_spec_mask]],
                              ignore_index=True)
        for c in cat_cols:
            X_tr_spec[c] = X_tr_spec[c].astype("category")
        y_tr_spec = np.concatenate(
            [y_tr[tr_spec], pseudo_y[pseudo_spec_mask]]).astype(np.int32)

        dtr = xgb.DMatrix(X_tr_spec, label=y_tr_spec, enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_spec], label=y_tr[va_spec], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100, verbose_eval=0,
        )
        bi = booster.best_iteration
        best_iters_spec.append(bi)
        val_pred = booster.predict(dva, iteration_range=(0, bi + 1))
        oof_spec_new[va_spec] = val_pred
        spec_idx = np.where(te_spec_mask)[0]
        test_spec_pred = booster.predict(dte_spec, iteration_range=(0, bi + 1))
        for i, pos in enumerate(spec_idx):
            test_spec_new[pos] += test_spec_pred[i] / N_FOLDS
        log(f"  spec fold {fold+1}  n_tr_aug={len(y_tr_spec)}  "
            f"best_iter={bi}  ({time.time()-t0:.1f}s)")

    # 5. Assemble pseudo-hybrid and compare
    oof_hybrid_new = oof_main_new.copy()
    oof_hybrid_new[tr_spec_mask] = oof_spec_new[tr_spec_mask]
    test_hybrid_new = test_main_new.copy()
    test_hybrid_new[te_spec_mask] = test_spec_new[te_spec_mask]

    bias_new, tuned_new = tune_log_bias(oof_hybrid_new, y_tr, prior)
    argmax_new = balanced_accuracy_score(y_tr, oof_hybrid_new.argmax(axis=1))
    cm = confusion_matrix(
        y_tr, (np.log(np.clip(oof_hybrid_new, 1e-9, 1.0)) + bias_new).argmax(axis=1))

    print(f"\n=== Pseudo-label hybrid (τ={TAU}) ===")
    print(f"  baseline hybrid OOF   : 0.97352")
    print(f"  pseudo-hybrid argmax  : {argmax_new:.5f}")
    print(f"  pseudo-hybrid tuned   : {tuned_new:.5f}")
    print(f"  Δ vs baseline         : {tuned_new - 0.97352:+.5f}")
    print(f"  bias                  : {bias_new.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / f"oof_pseudo_hybrid_tau{int(TAU*100)}.npy", oof_hybrid_new)
    np.save(ART / f"test_pseudo_hybrid_tau{int(TAU*100)}.npy", test_hybrid_new)
    with open(ART / f"pseudo_hybrid_tau{int(TAU*100)}_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS, "tau": TAU,
            "n_pseudo_rows": n_pseudo,
            "pseudo_class_dist": dist.tolist(),
            "baseline_tuned": 0.97352,
            "pseudo_tuned": float(tuned_new),
            "delta": float(tuned_new - 0.97352),
            "log_bias": bias_new.tolist(),
            "best_iters_main": [int(x) for x in best_iters_main],
            "best_iters_spec": [int(x) for x in best_iters_spec],
        }, f, indent=2)
    tuned_idx = (np.log(np.clip(test_hybrid_new, 1e-9, 1.0)) + bias_new).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / f"submission_pseudo_hybrid_tau{int(TAU*100)}.csv", index=False)
    log("artefacts saved")


if __name__ == "__main__":
    main()
