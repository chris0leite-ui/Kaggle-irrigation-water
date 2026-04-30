"""#1+#3 fusion: override-success classifier with CFR features.

Adds 5 CFR features to the n1 classifier feature set (25 → 30 features).
Tests whether CFR signal lifts AUC and enables tighter direction filters.

Same fold protocol as n1, same XGB HPs. Compare AUC + filtered OOF lift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def _normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def break_even(prior, a, c):
    return prior[c] / (prior[a] + prior[c])


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    t1b_oof = _normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    t1b_test = _normed(np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32))

    v1_b, _ = tune_log_bias(v1_oof, y, prior)
    raw_b, _ = tune_log_bias(raw_oof, y, prior)
    t1b_b, _ = tune_log_bias(t1b_oof, y, prior)

    def biased_arg(p, b):
        return (np.log(np.clip(p, 1e-9, 1.0)) + b).argmax(1)

    v1_oof_arg = biased_arg(v1_oof, v1_b)
    v1_test_arg = biased_arg(v1_test, v1_b)
    raw_oof_arg = biased_arg(raw_oof, raw_b)
    raw_test_arg = biased_arg(raw_test, raw_b)
    t1b_oof_arg = biased_arg(t1b_oof, t1b_b)
    t1b_test_arg = biased_arg(t1b_test, t1b_b)

    cand_oof = (raw_oof_arg == t1b_oof_arg) & (raw_oof_arg != v1_oof_arg)
    cand_test = (raw_test_arg == t1b_test_arg) & (raw_test_arg != v1_test_arg)

    # CFR features
    cfr_train = np.load(ART / "cfr_features_train.npy")  # (n, 5): sm/rf/tc/ws/total
    cfr_test = np.load(ART / "cfr_features_test.npy")

    def build_features(v1, raw, t1b, raw_arg, t1b_arg, v1_arg, dist_df, cfr):
        n = len(v1)
        consensus = raw_arg
        feat = np.column_stack([
            v1.max(axis=1), entropy(v1),
            raw.max(axis=1), entropy(raw),
            t1b.max(axis=1), entropy(t1b),
            v1[:, 0], v1[:, 1], v1[:, 2],
            raw[:, 0], raw[:, 1], raw[:, 2],
            t1b[:, 0], t1b[:, 1], t1b[:, 2],
            v1[np.arange(n), v1_arg] - v1[np.arange(n), consensus],
            v1_arg, consensus,
            dist_df["dgp_score"].to_numpy(),
            dist_df["sm_dist"].to_numpy(),
            dist_df["rf_dist"].to_numpy(),
            dist_df["tc_dist"].to_numpy(),
            dist_df["ws_dist"].to_numpy(),
            dist_df["sm_abs"].to_numpy(),
            dist_df["rf_abs"].to_numpy(),
            cfr,  # 5 cols
        ])
        return feat

    X_oof = build_features(v1_oof, raw_oof, t1b_oof, raw_oof_arg, t1b_oof_arg, v1_oof_arg, train_dist, cfr_train)
    X_test = build_features(v1_test, raw_test, t1b_test, raw_test_arg, t1b_test_arg, v1_test_arg, test_dist, cfr_test)
    print(f"Feature matrix shape: OOF {X_oof.shape} (was 25, now {X_oof.shape[1]})")

    cand_idx = np.where(cand_oof)[0]
    consensus_oof = raw_oof_arg[cand_idx]
    target = (y[cand_idx] == consensus_oof).astype(np.int32)
    Xc = X_oof[cand_idx]
    print(f"OOF candidates: {len(target)}, positive rate: {target.mean():.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_classifier = np.zeros(len(target), dtype=np.float32)
    fold_best_iters = []
    params = dict(objective="binary:logistic", max_depth=4,
                  learning_rate=0.05, eval_metric="logloss",
                  reg_alpha=1.0, reg_lambda=2.0, subsample=0.9, colsample_bytree=0.9,
                  tree_method="hist", verbosity=0)
    for fold, (tr, va) in enumerate(skf.split(Xc, target)):
        dtr = xgb.DMatrix(Xc[tr], label=target[tr])
        dva = xgb.DMatrix(Xc[va], label=target[va])
        booster = xgb.train(params, dtr, num_boost_round=500,
                            evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
        oof_classifier[va] = booster.predict(dva)
        fold_best_iters.append(int(booster.best_iteration))
        auc = roc_auc_score(target[va], oof_classifier[va])
        print(f"  fold {fold+1}: best_iter={booster.best_iteration}  AUC={auc:.4f}")
    auc = roc_auc_score(target, oof_classifier)
    print(f"OVERALL AUC (n1+CFR): {auc:.4f} (n1 alone was 0.8804)")

    # Per-direction precision sweep
    print(f"\n=== Per-direction τ sweep ===")
    direction_thresh = {}
    direction_n = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            m = (v1_oof_arg[cand_idx] == a) & (consensus_oof == c)
            if m.sum() == 0: continue
            be = break_even(prior, a, c)
            current_prec = target[m].mean()
            if current_prec >= be:
                direction_thresh[(a, c)] = 0.0
                continue
            probs = oof_classifier[m]
            outs = target[m]
            order = np.argsort(-probs)
            cum = np.cumsum(outs[order])
            cumprec = cum / np.arange(1, len(outs) + 1)
            valid = cumprec >= be
            if valid.any():
                k = np.where(valid)[0][-1] + 1
                tau = float(probs[order[k - 1]])
                direction_thresh[(a, c)] = tau
                kept = (probs >= tau).sum()
                kept_correct = ((probs >= tau) & (outs == 1)).sum()
                kept_prec = kept_correct / max(kept, 1)
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: τ={tau:.4f}  "
                      f"keep {kept}/{m.sum()}  prec={kept_prec:.4f} (BE={be:.4f})")
            else:
                direction_thresh[(a, c)] = 1.01
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: NO τ; reject all {m.sum()}")
            direction_n[(a, c)] = int(m.sum())

    # Apply filtered override on OOF
    base_v1 = balanced_accuracy_score(y, v1_oof_arg)
    full = v1_oof_arg.copy()
    full[cand_oof] = consensus_oof
    bal_full = balanced_accuracy_score(y, full)

    filtered = v1_oof_arg.copy()
    n_apply = 0
    for i_loc, idx in enumerate(cand_idx):
        a = v1_oof_arg[idx]
        c = consensus_oof[i_loc]
        tau = direction_thresh.get((a, c), 1.01)
        if oof_classifier[i_loc] >= tau:
            filtered[idx] = c
            n_apply += 1
    bal_filt = balanced_accuracy_score(y, filtered)
    print(f"\n=== OOF results ===")
    print(f"  v1 baseline:                    {base_v1:.5f}")
    print(f"  FULL k=2 unanimous:             {bal_full:.5f}  Δ vs v1={bal_full-base_v1:+.5f}")
    print(f"  FILTERED (n+CFR, n={n_apply}/{cand_oof.sum()}): {bal_filt:.5f}  "
          f"Δ vs FULL={bal_filt-bal_full:+.5f}")

    # Train final + apply on test
    median_best = max(int(np.median(fold_best_iters)), 30)
    final_booster = xgb.train(params, xgb.DMatrix(Xc, label=target),
                              num_boost_round=median_best)
    test_classifier = final_booster.predict(xgb.DMatrix(X_test[cand_test]))

    v1_csv_pred = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")[TARGET].map(CLS2IDX).to_numpy()
    new_test = v1_csv_pred.copy()
    consensus_test = raw_test_arg
    cand_test_idx = np.where(cand_test)[0]
    n_test_kept = 0
    for j, idx in enumerate(cand_test_idx):
        a = v1_test_arg[idx]
        c = consensus_test[idx]
        tau = direction_thresh.get((a, c), 1.01)
        if test_classifier[j] >= tau:
            new_test[idx] = c
            n_test_kept += 1
    winner_pred = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff_w = int((new_test != winner_pred).sum())
    print(f"\nTest: kept {n_test_kept}/{cand_test.sum()}, diff vs LB-best winner: {diff_w}")

    path = SUB / "submission_n13_n1_plus_cfr.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in new_test]}).to_csv(path, index=False)
    print(f"Saved: {path}")

    summary = {
        "OOF_AUC": float(auc),
        "baseline_v1": float(base_v1),
        "FULL_k2": float(bal_full),
        "FILTERED": float(bal_filt),
        "delta_filt_vs_full": float(bal_filt - bal_full),
        "n_oof_kept": int(n_apply),
        "n_test_kept": int(n_test_kept),
        "diff_vs_winner": diff_w,
        "direction_thresholds": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": float(t)
                                 for (a, c), t in direction_thresh.items()},
        "submission": str(path),
    }
    with open(ART / "n13_n1_plus_cfr_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary")


if __name__ == "__main__":
    main()
