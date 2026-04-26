"""P2 bucket-aware specialist trainer.

Two binary heads, 5-fold StratifiedKFold(seed=42) ALIGNED with every saved
OOF (so the resulting OOF probs can be soft-logit-added into the LB-best
4-stack at the same row indices).

Important: each specialist is trained ONLY on rows in its bucket, but the
stratified split is on the FULL training set's y. We get the bucket's row
indices via dgp_score, then take the intersection with each fold's tr_idx
and va_idx. That keeps the held-out rows aligned with the global OOF split.

For each row in the FULL training set:
  - If row's score == bucket_score: oof prediction = specialist's val pred
  - Else: oof prediction = nan (signaling "not in bucket")

Test side: predict ALL test rows; consumer masks by score later.

Saves:
  scripts/artifacts/oof_p2_score3.npy  (n_train, fp32 P(y=Med | score=3))
                                        — nan outside bucket
  scripts/artifacts/oof_p2_score6.npy  (similar, P(y=High | score=6))
  scripts/artifacts/test_p2_score3.npy (n_test fp32)
  scripts/artifacts/test_p2_score6.npy
  scripts/artifacts/p2_specialists_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, add_distance_features  # noqa: E402
from p2_features import build_features  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
SMOKE = bool(int(os.environ.get("SMOKE", "0")))

BUCKETS = {
    "score3": dict(score=3, target_class=1, target_name="Medium"),
    "score6": dict(score=6, target_class=2, target_name="High"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if SMOKE:
        train = train.head(20_000).reset_index(drop=True)
        test = test.head(5_000).reset_index(drop=True)

    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    # Compute dgp_score for both train and test
    log("computing dgp_score for train + test")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    score_tr = tr_d["dgp_score"].to_numpy().astype(np.int8)
    score_te = te_d["dgp_score"].to_numpy().astype(np.int8)

    # Per-bucket
    summary = {}
    for bucket, cfg in BUCKETS.items():
        log(f"\n=========== {bucket} (score=={cfg['score']}, target={cfg['target_name']}) ===========")
        bucket_mask_tr = (score_tr == cfg["score"])
        bucket_mask_te = (score_te == cfg["score"])
        n_in_bucket = int(bucket_mask_tr.sum())
        target_full = (y == cfg["target_class"]).astype(np.int8)
        n_pos_in_bucket = int(target_full[bucket_mask_tr].sum())
        log(f"  rows in bucket: train {n_in_bucket}, test {bucket_mask_te.sum()}")
        log(f"  positive class ({cfg['target_name']}) in bucket: {n_pos_in_bucket} "
            f"({100*n_pos_in_bucket/max(n_in_bucket,1):.2f}%)")

        Xtr, Xte, feat_names = build_features(train, test, score_tr, score_te, bucket)
        log(f"  features: {len(feat_names)} cols")

        oof = np.full(len(train), np.nan, dtype=np.float32)
        test_pred = np.zeros(len(test), dtype=np.float32)

        skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
        fold_aucs = []
        for fold, (tr_idx_full, va_idx_full) in enumerate(skf.split(Xtr, y), 1):
            # Restrict each fold to bucket rows
            tr_mask = bucket_mask_tr[tr_idx_full]
            va_mask = bucket_mask_tr[va_idx_full]
            tr_idx_b = tr_idx_full[tr_mask]
            va_idx_b = va_idx_full[va_mask]
            n_tr_b = len(tr_idx_b); n_va_b = len(va_idx_b)
            n_pos_tr = int(target_full[tr_idx_b].sum())
            spw = max((n_tr_b - n_pos_tr) / max(n_pos_tr, 1), 1.0)

            t1 = time.time()
            dtr = xgb.DMatrix(Xtr.iloc[tr_idx_b].values, label=target_full[tr_idx_b])
            dva = xgb.DMatrix(Xtr.iloc[va_idx_b].values, label=target_full[va_idx_b])
            dte = xgb.DMatrix(Xte.iloc[bucket_mask_te.nonzero()[0]].values)
            params = dict(
                objective="binary:logistic", eval_metric="auc",
                max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=2.0, reg_lambda=2.0,
                scale_pos_weight=spw,
                tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
            )
            bst = xgb.train(
                params, dtr, num_boost_round=3000 if not SMOKE else 200,
                evals=[(dva, "val")],
                early_stopping_rounds=200 if not SMOKE else 30,
                verbose_eval=0,
            )
            bi = bst.best_iteration
            vp = bst.predict(dva, iteration_range=(0, bi + 1))
            oof[va_idx_b] = vp.astype(np.float32)
            # Test side: predict only bucket-test-rows once, average across folds
            te_idx_b = bucket_mask_te.nonzero()[0]
            tp = bst.predict(dte, iteration_range=(0, bi + 1))
            test_pred[te_idx_b] += tp.astype(np.float32) / N_FOLDS

            try:
                auc = roc_auc_score(target_full[va_idx_b], vp)
            except ValueError:
                auc = np.nan
            fold_aucs.append(float(auc))
            log(f"  fold {fold}/{N_FOLDS}  it={bi}  "
                f"n_tr={n_tr_b}/{len(tr_idx_full)}  n_va={n_va_b}/{len(va_idx_full)}  "
                f"auc={auc:.5f}  wall={time.time()-t1:.1f}s")

        # Save (oof has NaN outside bucket; consumer must mask)
        np.save(ART / f"oof_p2_{bucket}.npy", oof)
        np.save(ART / f"test_p2_{bucket}.npy", test_pred)
        log(f"  saved oof_p2_{bucket}.npy + test_p2_{bucket}.npy")

        # Aggregate AUC over the bucket
        bucket_oof = oof[bucket_mask_tr]
        bucket_target = target_full[bucket_mask_tr]
        try:
            agg_auc = roc_auc_score(bucket_target, bucket_oof)
        except ValueError:
            agg_auc = np.nan
        log(f"  bucket-OOF AUC={agg_auc:.5f}  fold-mean AUC={np.nanmean(fold_aucs):.5f}")

        summary[bucket] = dict(
            n_train_in_bucket=n_in_bucket, n_pos=n_pos_in_bucket,
            n_test_in_bucket=int(bucket_mask_te.sum()),
            fold_aucs=fold_aucs, agg_auc=float(agg_auc),
            features=feat_names,
        )

    (ART / "p2_specialists_results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {ART / 'p2_specialists_results.json'}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
