"""F — Data-scaling diagnostic for the recipe pipeline.

Goal: determine if LB 0.98094 ceiling is data-limited or model-limited.
Method: load recipe FE once, then sweep training-row subsamples on
held-out fold 0 (126k rows) with same XGB HPs.

Outputs:
  scripts/artifacts/F_data_scaling_results.json

Wall budget: ~25 min (3 min FE + 5 sequential XGB fits at growing N).

SMOKE=1: train=20k, fold split = 1 fold, fractions [0.5, 1.0] only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

# Force NO non-default recipe FE knobs so we get the canonical V10 set.
for k in ("EXTRA_FE", "DROP_SCORES", "ANCHOR_WEIGHT_ALPHA", "TTA_BOUNDARY",
          "EXTRA_OOD", "EXTRA_KNN10K", "GBY", "DAE_EMBED_PATH",
          "INSTAB", "EXTRA_OOD9", "EXTRA_AV", "EXTRA_NNDIST",
          "CLEANLAB_TREATMENT", "DROP_DETERMINISTIC", "THREE_WAY_OTE"):
    os.environ.pop(k, None)
os.environ["FOLD_SEED"] = "42"
os.environ["XGB_BOOSTER"] = "gbtree"
os.environ.setdefault("OTE_ALPHA", "1.0")

import recipe_full_te as R  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

ART = Path(__file__).parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_smoke" if SMOKE else ""
OUT_JSON = ART / f"F_data_scaling{SUFFIX}_results.json"
SEED = 42

# Subsample fractions of the (504k) fold-0 training set.
FRACS = [0.50, 1.00] if SMOKE else [0.10, 0.20, 0.40, 0.80, 1.00]

# XGB HPs: identical to recipe_full_te's CPU production config.
XGB_PARAMS = dict(
    n_estimators=300 if SMOKE else 3000,
    max_depth=4, max_leaves=30,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2, reg_alpha=5, reg_lambda=5,
    max_bin=256 if SMOKE else 1024,
    objective="multi:softprob", tree_method="hist",
    eval_metric="mlogloss",
    enable_categorical=False, n_jobs=-1, random_state=SEED,
    early_stopping_rounds=50 if SMOKE else 200, verbosity=0, booster="gbtree",
)


def log(msg: str) -> None:
    print(f"[F {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    log("loading + engineering recipe FE (one-shot)")
    train, test, info, _ = R.load_and_engineer()
    y = train[R.TARGET].to_numpy()
    log(f"train={len(train):,}  feature_cols={len(train.columns)}")

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    te_targets = info["te_cols"]
    log(f"numeric_feats={len(numeric_feats)}  te_target_cats={len(te_targets)}")

    # StratifiedKFold seed=42 — fold 0 split aligned with all OOFs on disk.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))
    tr_idx0, va_idx0 = splits[0]
    log(f"fold 0: train_rows={len(tr_idx0):,}  val_rows={len(va_idx0):,}")

    # Per-fold OTE fit on FULL fold-0 training set; transform val once and reuse.
    # NOTE: OTE statistics are from full fold-0 train (504k); subsample only at
    # XGB training step. This means small-N XGB models see "leaked" OTE statistics
    # built from the full training set. This is acceptable for the diagnostic
    # purpose (relative-shape of OOF vs N is still informative; we measure model
    # capacity-vs-data interaction, not absolute generalization).
    log("fitting OrderedTE on full fold-0 train (one-shot, applies to all subsamples)")
    rng_ote = np.random.default_rng(SEED)
    perm = rng_ote.permutation(len(tr_idx0))
    df_tr = train.iloc[tr_idx0[perm]].reset_index(drop=True)
    df_va = train.iloc[va_idx0].reset_index(drop=True)
    ote = OrderedTE(a=R.OTE_ALPHA)
    df_tr_with_ote = ote.fit(df_tr, cat_cols=te_targets, target=R.TARGET)
    df_va_with_ote = ote.transform(df_va)
    # Unshuffle train back to original order.
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    df_tr_with_ote = df_tr_with_ote.iloc[inv].reset_index(drop=True)
    log(f"  OTE done in {time.time() - t0:.0f}s")

    feat_cols = numeric_feats + ote.te_col_names()
    X_tr_full = df_tr_with_ote[feat_cols].to_numpy(dtype=np.float32)
    y_tr_full = y[tr_idx0]
    X_va = df_va_with_ote[feat_cols].to_numpy(dtype=np.float32)
    y_va = y[va_idx0]
    log(f"X_tr_full={X_tr_full.shape}  X_va={X_va.shape}  feat_cols={len(feat_cols)}")

    rng = np.random.default_rng(SEED)
    results = []

    for frac in FRACS:
        # Stratified subsample to preserve class balance.
        sub_idx = []
        for cls in [0, 1, 2]:
            cls_idx = np.where(y_tr_full == cls)[0]
            n_cls_sub = int(round(frac * len(cls_idx)))
            picked = rng.choice(cls_idx, size=n_cls_sub, replace=False)
            sub_idx.append(picked)
        sub_idx = np.sort(np.concatenate(sub_idx))
        log(f"--- frac={frac:.2f} N={len(sub_idx):,} ---")

        X_tr = X_tr_full[sub_idx]
        y_tr = y_tr_full[sub_idx]
        sw = compute_sample_weight("balanced", y_tr)

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        t1 = time.time()
        clf.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
        wall = time.time() - t1

        proba = clf.predict_proba(X_va)
        argmax_pred = proba.argmax(axis=1)
        argmax_bal = balanced_accuracy_score(y_va, argmax_pred)

        # Tune log-bias on val (matches recipe pipeline's decision rule).
        # Use train-prior from y_tr (recipe convention).
        prior = np.bincount(y_tr, minlength=3) / len(y_tr)
        bias, tuned_bal = tune_log_bias(proba, y_va, prior)

        log(f"  best_iter={clf.best_iteration}  argmax={argmax_bal:.5f}  "
            f"tuned={tuned_bal:.5f}  bias={[round(b, 3) for b in bias.tolist()]}  wall={wall:.0f}s")

        results.append({
            "frac": frac,
            "n_train": int(len(sub_idx)),
            "best_iter": int(clf.best_iteration),
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned_bal),
            "bias": [float(b) for b in bias.tolist()],
            "wall_seconds": float(wall),
        })

        OUT_JSON.write_text(json.dumps({
            "smoke": SMOKE,
            "fold": 0, "n_val": int(len(va_idx0)),
            "n_train_full": int(len(tr_idx0)),
            "results": results,
            "elapsed_seconds": time.time() - t0,
        }, indent=2))
        log(f"  saved {OUT_JSON.name}")

    log("=" * 60)
    log(f"SUMMARY (fold 0, val_rows={len(va_idx0):,})")
    log(f"{'frac':>6} {'N':>9} {'argmax':>10} {'tuned':>10}")
    for r in results:
        log(f"{r['frac']:>6.2f} {r['n_train']:>9,} {r['argmax_bal']:>10.5f} {r['tuned_bal']:>10.5f}")
    log(f"total wall = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
