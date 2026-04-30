"""#1 Override-success classifier features.

Train a binary XGB on OOF override candidates to predict whether each
would-be override is correct. Use as a SOFT GATE (per-row probability)
to filter overrides at break-even precision.

OOF candidates: rows where (raw_argmax == t1b_argmax) ≠ v1_argmax (k=2 unanimous).
Target: 1 if true label == consensus class, 0 otherwise.

Features (per row):
  - v1's max_prob and entropy
  - raw's max_prob and entropy
  - t1b's max_prob and entropy
  - per-class probs of all 3 models (9 features)
  - max_prob_diff between v1 and consensus
  - dgp_score (rule cell)
  - 4 signed dist-to-threshold features
  - rule-axis raw numerics

XGB: depth=4, n_est=500, lr=0.05, no class_weight (target ~75% positive).

Deploy:
  - Train on OOF k=2 unanimous candidates with leak-free 5-fold
  - Apply to test winner's 286 override candidates
  - For each direction, find threshold τ_d such that precision >= break_even
  - Filter: override only if classifier_prob > τ_d

Per CLAUDE.md: emit candidate CSVs only.
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
CLS = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLS)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def _normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def entropy(p, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def break_even_precision(prior, anchor_class, override_class):
    return prior[override_class] / (prior[anchor_class] + prior[override_class])


def build_features(v1, raw, t1b, raw_arg, t1b_arg, v1_arg, dist_df):
    """Per-row feature matrix for override-success classifier."""
    n = len(v1)
    consensus = raw_arg  # since raw_arg == t1b_arg on candidate rows
    feat = np.column_stack([
        v1.max(axis=1),                   # v1_max_prob
        entropy(v1),                       # v1_entropy
        raw.max(axis=1),                   # raw_max_prob
        entropy(raw),                      # raw_entropy
        t1b.max(axis=1),                   # t1b_max_prob
        entropy(t1b),                      # t1b_entropy
        v1[:, 0], v1[:, 1], v1[:, 2],      # v1 per-class probs
        raw[:, 0], raw[:, 1], raw[:, 2],   # raw per-class probs
        t1b[:, 0], t1b[:, 1], t1b[:, 2],   # t1b per-class probs
        v1[np.arange(n), v1_arg] - v1[np.arange(n), consensus],   # confidence diff
        v1_arg, consensus,                 # categorical hints
        dist_df["dgp_score"].to_numpy(),
        dist_df["sm_dist"].to_numpy(),
        dist_df["rf_dist"].to_numpy(),
        dist_df["tc_dist"].to_numpy(),
        dist_df["ws_dist"].to_numpy(),
        dist_df["sm_abs"].to_numpy(),
        dist_df["rf_abs"].to_numpy(),
    ])
    feat_names = [
        "v1_maxp", "v1_entr", "raw_maxp", "raw_entr", "t1b_maxp", "t1b_entr",
        "v1_p_L", "v1_p_M", "v1_p_H",
        "raw_p_L", "raw_p_M", "raw_p_H",
        "t1b_p_L", "t1b_p_M", "t1b_p_H",
        "v1_conf_diff", "v1_arg", "consensus",
        "dgp_score", "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs",
    ]
    return feat, feat_names


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # Build dist-features for both train and test
    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)

    # Load anchor + 2 OTHERS
    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    t1b_oof = _normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    t1b_test = _normed(np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32))

    # Tuned argmaxes
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

    # Override candidate masks
    cand_oof = (raw_oof_arg == t1b_oof_arg) & (raw_oof_arg != v1_oof_arg)
    cand_test = (raw_test_arg == t1b_test_arg) & (raw_test_arg != v1_test_arg)
    print(f"OOF candidates:  {cand_oof.sum()}")
    print(f"Test candidates: {cand_test.sum()}")

    # Build features for all rows then subset
    X_oof, names = build_features(v1_oof, raw_oof, t1b_oof,
                                  raw_oof_arg, t1b_oof_arg, v1_oof_arg, train_dist)
    X_test, _ = build_features(v1_test, raw_test, t1b_test,
                               raw_test_arg, t1b_test_arg, v1_test_arg, test_dist)
    print(f"Feature matrix shape: OOF {X_oof.shape}, test {X_test.shape}")
    print(f"Features: {names}")

    # ===== Train binary XGB on OOF candidates with 5-fold leak-free =====
    cand_idx = np.where(cand_oof)[0]
    consensus_oof = raw_oof_arg[cand_idx]   # consensus class on candidates
    target = (y[cand_idx] == consensus_oof).astype(np.int32)
    Xc = X_oof[cand_idx]
    print(f"\nClassifier target stats: positive rate = {target.mean():.3f} "
          f"(n_positive={target.sum()}, n_total={len(target)})")

    # Per-direction precision (= positive rate) on OOF
    print(f"\n=== OOF override precision per direction ===")
    direction_prec = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            m = (v1_oof_arg[cand_idx] == a) & (consensus_oof == c)
            n = m.sum()
            if n == 0: continue
            prec = target[m].mean()
            be = break_even_precision(prior, a, c)
            direction_prec[(a, c)] = (int(n), float(prec), float(be))
            print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: n={n:4d}  prec={prec:.4f}  BE={be:.4f}  margin={prec-be:+.4f}")

    # 5-fold OOF predictions on candidates (stratified by target)
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
        pred = booster.predict(dva)
        oof_classifier[va] = pred
        fold_best_iters.append(int(booster.best_iteration))
        auc = roc_auc_score(target[va], pred)
        print(f"  fold {fold+1}: best_iter={booster.best_iteration}  AUC={auc:.4f}")
    overall_auc = roc_auc_score(target, oof_classifier)
    print(f"OVERALL OOF AUC: {overall_auc:.4f}")

    # ===== Train final classifier on all OOF candidates for test inference =====
    median_best = int(np.median(fold_best_iters))
    print(f"Final classifier n_round = median best_iter = {median_best}")
    dall = xgb.DMatrix(Xc, label=target)
    final_booster = xgb.train(params, dall, num_boost_round=max(median_best, 30))
    test_classifier = final_booster.predict(xgb.DMatrix(X_test[cand_test]))

    # ===== Per-direction threshold sweep on OOF =====
    print(f"\n=== Threshold sweep per direction (find τ such that filtered_precision >= BE) ===")
    direction_thresh = {}
    for a in range(3):
        for c in range(3):
            if (a, c) not in direction_prec: continue
            m = (v1_oof_arg[cand_idx] == a) & (consensus_oof == c)
            be = direction_prec[(a, c)][2]
            if direction_prec[(a, c)][1] >= be:
                # Direction already above BE; no threshold needed
                direction_thresh[(a, c)] = 0.0  # accept all
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: already above BE; τ=0.0 "
                      f"(keep all {m.sum()})")
                continue
            # Sweep τ ascending; find smallest τ such that filtered precision >= BE
            probs = oof_classifier[m]
            outcomes = target[m]
            # Sort by probs descending; cumulative precision at each cutoff
            order = np.argsort(-probs)
            sorted_outcomes = outcomes[order]
            sorted_probs = probs[order]
            cumcorrect = np.cumsum(sorted_outcomes)
            cumprec = cumcorrect / np.arange(1, len(sorted_outcomes) + 1)
            # Find largest k such that cumprec[k-1] >= BE
            valid = cumprec >= be
            if valid.any():
                k = np.where(valid)[0][-1] + 1
                tau = sorted_probs[k - 1]
                direction_thresh[(a, c)] = float(tau)
                kept = (probs >= tau).sum()
                kept_correct = ((probs >= tau) & (outcomes == 1)).sum()
                kept_prec = kept_correct / max(kept, 1)
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: τ={tau:.4f}  "
                      f"keep {kept}/{m.sum()}  prec={kept_prec:.4f} (BE={be:.4f})")
            else:
                # No threshold yields BE; reject all
                direction_thresh[(a, c)] = 1.01
                print(f"  {IDX2CLS[a]:<7}->{IDX2CLS[c]:<7}: NO τ reaches BE; "
                      f"reject all {m.sum()}")

    # ===== Build OOF-side filtered override =====
    base_v1 = balanced_accuracy_score(y, v1_oof_arg)
    # FULL = unfiltered k=2 unanimous
    full_oof = v1_oof_arg.copy()
    full_oof[cand_oof] = consensus_oof
    full_bal = balanced_accuracy_score(y, full_oof)

    # FILTERED: per-direction τ
    filtered_oof = v1_oof_arg.copy()
    n_filtered_apply = 0
    for i_loc, idx in enumerate(cand_idx):
        a = v1_oof_arg[idx]
        c = consensus_oof[i_loc]
        tau = direction_thresh.get((a, c), 1.01)
        if oof_classifier[i_loc] >= tau:
            filtered_oof[idx] = c
            n_filtered_apply += 1
    filt_bal = balanced_accuracy_score(y, filtered_oof)
    print(f"\n=== OOF results ===")
    print(f"  v1 baseline               : {base_v1:.5f}")
    print(f"  FULL k=2 unanimous        : {full_bal:.5f}  Δ vs v1={full_bal-base_v1:+.5f}")
    print(f"  FILTERED (n={n_filtered_apply}/{cand_oof.sum()}): {filt_bal:.5f}  "
          f"Δ vs v1={filt_bal-base_v1:+.5f}  Δ vs FULL={filt_bal-full_bal:+.5f}")

    # ===== Build test-side filtered override =====
    # winner = LB-best 0.98140 (already on disk); use v1 CSV as anchor
    v1_csv_pred = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")[TARGET].map(CLS2IDX).to_numpy()
    new_test = v1_csv_pred.copy()
    consensus_test = raw_test_arg
    n_test_kept = 0
    cand_test_idx = np.where(cand_test)[0]
    for j, idx in enumerate(cand_test_idx):
        a = v1_test_arg[idx]
        c = consensus_test[idx]
        tau = direction_thresh.get((a, c), 1.01)
        if test_classifier[j] >= tau:
            new_test[idx] = c
            n_test_kept += 1

    # Diff vs LB-best winner
    winner_pred = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff_winner = int((new_test != winner_pred).sum())
    print(f"\nTest-side filtered override:")
    print(f"  Total candidates: {cand_test.sum()}")
    print(f"  Kept after filter: {n_test_kept}")
    print(f"  Diff vs LB-best winner (0.98140): {diff_winner}")

    path = SUB / "submission_n1_override_classifier_filtered.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in new_test]}).to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Save summary
    summary = {
        "OOF_AUC": float(overall_auc),
        "baseline_v1_OOF": float(base_v1),
        "FULL_k2_OOF": float(full_bal),
        "FILTERED_OOF": float(filt_bal),
        "delta_FILTERED_vs_FULL_OOF": float(filt_bal - full_bal),
        "delta_FILTERED_vs_v1_OOF": float(filt_bal - base_v1),
        "n_oof_kept": int(n_filtered_apply),
        "n_oof_candidates": int(cand_oof.sum()),
        "n_test_kept": int(n_test_kept),
        "n_test_candidates": int(cand_test.sum()),
        "diff_vs_winner": diff_winner,
        "direction_precision": {
            f"{IDX2CLS[a]}->{IDX2CLS[c]}": dict(n=n, prec=p, BE=be)
            for (a, c), (n, p, be) in direction_prec.items()
        },
        "direction_thresholds": {
            f"{IDX2CLS[a]}->{IDX2CLS[c]}": float(t) for (a, c), t in direction_thresh.items()
        },
        "submission": str(path),
    }
    with open(ART / "n1_override_classifier_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: scripts/artifacts/n1_override_classifier_results.json")


if __name__ == "__main__":
    main()
