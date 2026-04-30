"""#2 Cross-submission disagreement features as standalone XGB classifier.

Compute per-row disagreement metrics across 5 LB-validated bases:
  - argmax_disagreement_count (0-4): how many bases disagree with v1
  - per_class_prob_std (3): std of per-class prob across bases
  - top1_top2_margin_v1 (1): v1 confidence
  - mean_pairwise_jaccard (1): error overlap heuristic
  - cross_max_prob_var (1): variance of max_prob across bases
  - vote_share_per_class (3): fraction of bases voting each class

Features (15 total) → train XGB to predict y on these features alone.
This becomes a standalone component for blending/override.

Different from saturated meta-stacker: uses ENSEMBLE-LEVEL row metadata,
not per-base raw probs. Should be uncorrelated with existing tree-meta.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a, eps=1e-9):
    return a / np.clip(a.sum(1, keepdims=True), eps, None)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)
    n_tr, n_te = len(train), len(test)

    bases = [
        ("v1",     "oof_sklearn_rf_meta_natural_v1_lb98129.npy",
                   "test_sklearn_rf_meta_natural_v1_lb98129.npy"),
        ("raw",    "oof_rawashishsin_2600.npy",
                   "test_rawashishsin_2600.npy"),
        ("t1b",    "oof_tier1b_greedy_meta.npy",
                   "test_tier1b_greedy_meta.npy"),
        ("recipe", "oof_recipe_full_te.npy",
                   "test_recipe_full_te.npy"),
        ("cb",     "oof_recipe_full_te_catboost.npy",
                   "test_recipe_full_te_catboost.npy"),
    ]
    log("Loading bases + computing tuned biases")
    pool = []
    for label, oof_p, test_p in bases:
        oof = _normed(np.load(ART / oof_p).astype(np.float32))
        tst = _normed(np.load(ART / test_p).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        # Apply bias and renormalize
        log_oof = np.log(np.clip(oof, 1e-9, 1.0)) + bias
        log_oof -= log_oof.max(axis=1, keepdims=True)
        oof_b = np.exp(log_oof)
        oof_b /= oof_b.sum(axis=1, keepdims=True)
        log_test = np.log(np.clip(tst, 1e-9, 1.0)) + bias
        log_test -= log_test.max(axis=1, keepdims=True)
        test_b = np.exp(log_test)
        test_b /= test_b.sum(axis=1, keepdims=True)
        pool.append(dict(label=label, oof=oof_b, test=test_b,
                         oof_arg=oof_b.argmax(1), test_arg=test_b.argmax(1)))
        log(f"  {label}: tuned {tuned:.5f}")

    n_bases = len(pool)
    log(f"\nBuilding disagreement features ({n_bases} bases)")

    def build_disagree_feats(probs_list, args_list):
        n = len(probs_list[0])
        # 1. argmax disagreement count vs v1 (assuming pool[0] is v1)
        v1_arg = args_list[0]
        disagree_cnt = np.zeros(n, dtype=np.int32)
        for i in range(1, n_bases):
            disagree_cnt += (args_list[i] != v1_arg).astype(np.int32)

        # 2. per-class prob std across bases (3 features)
        all_probs = np.stack(probs_list, axis=1)  # (n, n_bases, 3)
        prob_std = all_probs.std(axis=1)  # (n, 3)

        # 3. top1-top2 margin of v1 (confidence)
        v1_probs = probs_list[0]
        sort_v1 = np.sort(v1_probs, axis=1)
        margin = sort_v1[:, -1] - sort_v1[:, -2]

        # 4. max_prob variance across bases
        max_probs = np.stack([p.max(axis=1) for p in probs_list], axis=1)  # (n, n_bases)
        max_prob_var = max_probs.var(axis=1)

        # 5. vote share per class (3 features)
        vote_share = np.zeros((n, 3), dtype=np.float32)
        for i in range(n_bases):
            for k in range(3):
                vote_share[:, k] += (args_list[i] == k).astype(np.int32)
        vote_share /= n_bases

        # 6. mean per-class prob across bases (3 features)
        mean_per_class = all_probs.mean(axis=1)  # (n, 3)

        # 7. consensus argmax (most common vote)
        votes = np.zeros((n, 3), dtype=np.int32)
        for i in range(n_bases):
            for k in range(3):
                votes[:, k] += (args_list[i] == k).astype(np.int32)
        consensus_arg = votes.argmax(1)
        consensus_strength = votes.max(axis=1) / n_bases

        feats = np.column_stack([
            disagree_cnt,                      # 1
            prob_std,                           # 3
            margin.reshape(-1, 1),              # 1
            max_prob_var.reshape(-1, 1),        # 1
            vote_share,                         # 3
            mean_per_class,                     # 3
            consensus_arg.reshape(-1, 1),       # 1
            consensus_strength.reshape(-1, 1),  # 1
            v1_probs,                           # 3 (raw v1 probs)
        ])
        return feats

    oof_probs = [p["oof"] for p in pool]
    oof_args = [p["oof_arg"] for p in pool]
    test_probs = [p["test"] for p in pool]
    test_args = [p["test_arg"] for p in pool]

    X_oof = build_disagree_feats(oof_probs, oof_args)
    X_test = build_disagree_feats(test_probs, test_args)
    log(f"Feature matrix: OOF {X_oof.shape}, test {X_test.shape}")

    # Add dist features
    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)
    extra_cols = ["dgp_score", "sm_dist", "rf_dist", "tc_dist", "ws_dist"]
    X_oof_full = np.column_stack([X_oof, train_dist[extra_cols].to_numpy()])
    X_test_full = np.column_stack([X_test, test_dist[extra_cols].to_numpy()])
    log(f"With dist: {X_oof_full.shape}")

    # Train XGB on disagreement features → predict y (3-class)
    log("Training XGB-meta on disagreement features (5-fold seed=42)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((n_tr, 3), dtype=np.float32)
    test_meta = np.zeros((n_te, 3), dtype=np.float32)
    params = dict(objective="multi:softprob", num_class=3, max_depth=4,
                  learning_rate=0.05, eval_metric="mlogloss",
                  reg_alpha=2.0, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", verbosity=0)
    for fold, (tr, va) in enumerate(skf.split(X_oof_full, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X_oof_full[tr], label=y[tr])
        dva = xgb.DMatrix(X_oof_full[va], label=y[va])
        booster = xgb.train(params, dtr, num_boost_round=300,
                            evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
        oof_meta[va] = booster.predict(dva).reshape(-1, 3)
        test_meta += booster.predict(xgb.DMatrix(X_test_full)).reshape(-1, 3) / 5.0
        log(f"  fold {fold+1}: best_iter={booster.best_iteration}  time={time.time()-t0:.1f}s")

    np.save(ART / "oof_n2_disagree.npy", oof_meta)
    np.save(ART / "test_n2_disagree.npy", test_meta)

    bias, tuned = tune_log_bias(oof_meta, y, prior)
    pred_oof = (np.log(np.clip(oof_meta, 1e-9, 1.0)) + bias).argmax(1)
    pcr = per_class_recall(y, pred_oof)
    log(f"\n=== n2 disagreement-XGB standalone ===")
    log(f"  OOF tuned: {tuned:.5f}  bias={bias.round(3).tolist()}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Compare to v1
    v1_oof = pool[0]["oof"]
    v1_arg = pool[0]["oof_arg"]
    v1_bal = balanced_accuracy_score(y, v1_arg)
    diff = (pred_oof != v1_arg).sum()
    log(f"  v1 baseline: {v1_bal:.5f}, diff: {diff} OOF rows ({diff/n_tr*100:.2f}%)")

    # Build submission
    test_arg = (np.log(np.clip(test_meta, 1e-9, 1.0)) + bias).argmax(1)
    path = SUB / "submission_n2_disagree.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_arg]}).to_csv(path, index=False)
    log(f"  Saved: {path}")

    # Use as new OTHER for override (if standalone passes 0.97)
    winner_pred = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff_w = int((test_arg != winner_pred).sum())
    log(f"  Test diff vs LB-best winner: {diff_w}")

    # Also try as 3rd OTHER in k=3 unanimous override
    log(f"\n=== n2 as 3rd OTHER for k=3 unanimous override ===")
    raw_arg_test = pool[1]["test_arg"]
    t1b_arg_test = pool[2]["test_arg"]
    v1_arg_test = pool[0]["test_arg"]
    n2_test_arg = test_arg
    # k=3 unanimous: raw == t1b == n2 ≠ v1
    unanimous = (raw_arg_test == t1b_arg_test) & (raw_arg_test == n2_test_arg) & (raw_arg_test != v1_arg_test)
    n_over = unanimous.sum()
    log(f"  k=3 unanimous overrides: {n_over}")
    if n_over > 0:
        new_pred = v1_arg_test.copy()
        new_pred[unanimous] = raw_arg_test[unanimous]
        diff_w2 = int((new_pred != winner_pred).sum())
        log(f"  k=3 sub vs winner: {diff_w2}")
        path2 = SUB / "submission_n2_k3_unanimous.csv"
        pd.DataFrame({"id": test_ids,
                      TARGET: [IDX2CLS[i] for i in new_pred]}).to_csv(path2, index=False)
        log(f"  Saved: {path2}")

    summary = {
        "OOF_tuned": float(tuned),
        "bias": bias.tolist(),
        "PCR": pcr.tolist(),
        "diff_vs_v1": int(diff),
        "diff_vs_winner": diff_w,
        "n_features": X_oof_full.shape[1],
        "submission": str(path),
    }
    with open(ART / "n2_disagreement_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved summary")


if __name__ == "__main__":
    main()
