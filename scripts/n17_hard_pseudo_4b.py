"""Option C: Hard-pseudo from 4b (LB 0.98150) labeler.

Mechanism:
  1. Use 4b's test argmax as candidate pseudo-labels
  2. Filter to high-confidence rows: raw_max_prob ≥ 0.99 AND raw_argmax == 4b
     (4b is just a CSV; using rawashishsin's max_prob as confidence proxy)
  3. Augment train with filtered test rows
  4. Retrain recipe XGB on augmented train
  5. Use as 3rd OTHER for k=2 plurality on top of 4b

Why this might add independent signal:
  - Strongest labeler ever attempted (4b LB 0.98150)
  - Hard pseudo at τ=0.99 (vs prior soft-distill which was NULL)
  - Different training distribution → different decision surface
  - Could detect rows where 4b is right but trained on those rows
    helps the new model handle adjacent boundary rows

Compute: ~50 min CPU full 5-fold retrain. To save time, do 1-fold
quick smoke first; if standalone OOF on the smoke-fold is competitive,
skip full 5-fold and go straight to full-train test inference.

This is a TIME-SAVING variant: no full 5-fold retrain. Instead:
  - Add high-confidence pseudo rows to train
  - Retrain ONE XGB on augmented data (no CV)
  - Predict on test
  - Use as 3rd voter on top of 4b
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


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    log("Loading data")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # 4b labels (LB 0.98150)
    labels_4b = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    log(f"4b label dist on test: L={(labels_4b==0).sum()} M={(labels_4b==1).sum()} H={(labels_4b==2).sum()}")

    # rawashishsin test probs (for confidence)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    raw_test = raw_test / raw_test.sum(1, keepdims=True).clip(1e-9)
    raw_max = raw_test.max(axis=1)
    raw_arg = raw_test.argmax(1)

    # Filter: keep rows where raw_max ≥ tau AND raw_arg == 4b's label
    tau = 0.99
    high_conf = (raw_max >= tau) & (raw_arg == labels_4b)
    n_keep = high_conf.sum()
    log(f"Filter τ={tau}: {n_keep} test rows kept ({n_keep/len(labels_4b)*100:.1f}%)")
    log(f"  Class dist of kept rows: L={((labels_4b==0)&high_conf).sum()} "
        f"M={((labels_4b==1)&high_conf).sum()} H={((labels_4b==2)&high_conf).sum()}")

    # Build dist features
    log("Building dist features")
    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)
    feat_cols = [c for c in train_dist.columns
                 if pd.api.types.is_numeric_dtype(train_dist[c])
                 and train_dist[c].dtype != bool]
    feat_cols = [c for c in feat_cols if train_dist[c].dtype.kind in "fiub"]
    log(f"  {len(feat_cols)} features")
    Xtr = train_dist[feat_cols].to_numpy().astype(np.float32)
    Xte = test_dist[feat_cols].to_numpy().astype(np.float32)

    # Augment train with high-confidence test rows
    Xte_keep = Xte[high_conf]
    yte_keep = labels_4b[high_conf]
    Xaug = np.vstack([Xtr, Xte_keep])
    yaug = np.concatenate([y, yte_keep])
    log(f"Augmented train: {Xaug.shape} (+{n_keep} rows = {n_keep/len(Xtr)*100:.1f}% augmentation)")

    # ===== 5-fold OOF on REAL train rows only (pseudo rows always in tr_idx) =====
    log("Training XGB with 5-fold OOF (real rows only for val)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_tr = len(y)
    oof = np.zeros((n_tr, 3), dtype=np.float32)
    test_pred = np.zeros((len(Xte), 3), dtype=np.float32)
    params = dict(objective="multi:softprob", num_class=3, max_depth=4,
                  learning_rate=0.05, eval_metric="mlogloss",
                  reg_alpha=2.0, reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", verbosity=0)

    for fold, (tr, va) in enumerate(skf.split(Xtr, y)):
        t0 = time.time()
        # tr_idx in REAL train + ALL pseudo rows
        Xtr_aug = np.vstack([Xtr[tr], Xte_keep])
        ytr_aug = np.concatenate([y[tr], yte_keep])
        dtr = xgb.DMatrix(Xtr_aug, label=ytr_aug)
        dva = xgb.DMatrix(Xtr[va], label=y[va])
        booster = xgb.train(params, dtr, num_boost_round=500,
                            evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
        oof[va] = booster.predict(dva).reshape(-1, 3)
        test_pred += booster.predict(xgb.DMatrix(Xte)).reshape(-1, 3) / 5.0
        log(f"  fold {fold+1}: best_iter={booster.best_iteration}  time={time.time()-t0:.1f}s")

    np.save(ART / "oof_n17_pseudo_4b.npy", oof)
    np.save(ART / "test_n17_pseudo_4b.npy", test_pred)

    bias, tuned = tune_log_bias(oof, y, prior)
    pred_tuned = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1)
    pcr = per_class_recall(y, pred_tuned)
    bal_argmax = balanced_accuracy_score(y, oof.argmax(1))
    log(f"\n=== n17 pseudo-4b retrain ===")
    log(f"  OOF argmax: {bal_argmax:.5f}")
    log(f"  OOF tuned:  {tuned:.5f}  bias={bias.round(3).tolist()}")
    log(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Compare to v1 baseline
    v1 = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1 = v1 / v1.sum(1, keepdims=True).clip(1e-9)
    bv1, _ = tune_log_bias(v1, y, prior)
    v1_arg = (np.log(np.clip(v1, 1e-9, 1.0)) + bv1).argmax(1)
    diff_v1 = (pred_tuned != v1_arg).sum()
    log(f"  vs v1: {diff_v1} OOF rows differ ({diff_v1/n_tr*100:.2f}%)")

    # Use as 3rd voter on top of 4b
    log("\n=== Use as 3rd voter for 2-of-3 plurality on top of 4b ===")
    test_arg = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(1)
    # Load raw, t1b for OTHERS pool
    t1b = np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32)
    t1b = t1b / t1b.sum(1, keepdims=True).clip(1e-9)
    t1b_o = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    t1b_o = t1b_o / t1b_o.sum(1, keepdims=True).clip(1e-9)
    bt1b, _ = tune_log_bias(t1b_o, y, prior)
    t1btt = (np.log(np.clip(t1b, 1e-9, 1.0)) + bt1b).argmax(1)

    raw_t = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    raw_t = raw_t / raw_t.sum(1, keepdims=True).clip(1e-9)
    raw_o = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_o = raw_o / raw_o.sum(1, keepdims=True).clip(1e-9)
    braw, _ = tune_log_bias(raw_o, y, prior)
    rawtt = (np.log(np.clip(raw_t, 1e-9, 1.0)) + braw).argmax(1)

    # 2-of-3 plurality (raw, t1b, n17) on 4b anchor
    n_te = len(test_ids)
    oargs = np.stack([rawtt, t1btt, test_arg], axis=1)
    votes = np.zeros((n_te, 3), dtype=np.int32)
    for c in range(3):
        votes[:, c] = (oargs == c).sum(axis=1)
    not_anchor = (np.arange(3)[None, :] != labels_4b[:, None])
    elig = (votes >= 2) & not_anchor
    votes_elig = np.where(elig, votes, -1)
    any_elig = elig.any(axis=1)
    chosen = votes_elig.argmax(axis=1)
    n_over = int(any_elig.sum())
    log(f"  Test 2-of-3 overrides on 4b: {n_over}")
    print(f"  Direction breakdown:")
    for src in [0, 1, 2]:
        for dst in [0, 1, 2]:
            if src == dst: continue
            cnt = ((labels_4b == src) & (chosen == dst) & any_elig).sum()
            if cnt > 0:
                log(f"    {IDX2CLS[src]}->{IDX2CLS[dst]}: {cnt}")

    new_pred = labels_4b.copy()
    new_pred[any_elig] = chosen[any_elig]
    diff_4b = int((new_pred != labels_4b).sum())
    log(f"  vs 4b: {diff_4b} differ")
    path = SUB / "submission_n17_pseudo4b_layered.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in new_pred]}).to_csv(path, index=False)
    log(f"  Saved: {path}")

    summary = {
        "tau": tau,
        "n_pseudo_kept": int(n_keep),
        "n_pseudo_pct": float(n_keep / len(labels_4b)),
        "OOF_argmax": float(bal_argmax),
        "OOF_tuned": float(tuned),
        "PCR": pcr.tolist(),
        "diff_oof_vs_v1": int(diff_v1),
        "n_test_overrides_on_4b": n_over,
        "diff_test_vs_4b": diff_4b,
        "submission": str(path),
    }
    with open(ART / "n17_pseudo_4b_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved summary")


if __name__ == "__main__":
    main()
