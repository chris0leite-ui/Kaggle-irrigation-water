"""#3 multi-task auxiliary XGBs.

Train binary aux heads on the dist feature set targeting:
  AUX1: is_flipped_from_rule  = (y != rule_pred)  — bool of NN flip vs rule
  AUX2: is_missed_high        = (y == 2) & (rule_pred != 2)
  AUX3: is_missed_medium      = (y == 1) & (rule_pred != 1)

These targets are NEW supervision signals — none of the existing OOFs in
the meta-stacker bank target them directly. recipe_full_te / pseudolabel
target the multiclass y; specialists targeted (y == 2) only on score=6
domain. AUX1 is the global flip detector (every score), AUX2/AUX3 are
class-conditional flip detectors.

Hypothesis: a meta-stacker that consumes these aux OOFs can route
boundary rows differently than a stacker that only sees y posteriors.

Saves three (oof, test) pairs as new bank components for the combined
eval (rebuild meta-stacker, blend gate, decide on LB probe).
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

ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE", "0") == "1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_aux(name: str, X: np.ndarray, X_te: np.ndarray, y_bin: np.ndarray,
                strat_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one binary head 5-fold OOF, return (oof, test, metrics)."""
    oof = np.zeros(len(y_bin), dtype=np.float32)
    test = np.zeros(len(X_te), dtype=np.float32)
    pos = int(y_bin.sum())
    neg = len(y_bin) - pos
    spw = neg / max(pos, 1)
    log(f"  [{name}] pos={pos}/{len(y_bin)}  scale_pos_weight={spw:.2f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    n_round = 200 if SMOKE else 2000

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X[tr_idx], label=y_bin[tr_idx])
        dva = xgb.DMatrix(X[va_idx], label=y_bin[va_idx])
        dte = xgb.DMatrix(X_te)
        params = dict(
            objective="binary:logistic", eval_metric="auc",
            max_depth=5, learning_rate=0.05,
            min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=1.0, reg_lambda=1.0,
            scale_pos_weight=spw,
            tree_method="hist", verbosity=0, seed=SEED,
        )
        booster = xgb.train(params, dtr, num_boost_round=n_round,
                              evals=[(dva, "v")], early_stopping_rounds=100,
                              verbose_eval=0)
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        auc = roc_auc_score(y_bin[va_idx], oof[va_idx])
        aucs.append(auc)
        log(f"    fold {fold+1}: it={bi}  auc={auc:.4f}  wall={time.time()-t0:.1f}s")

    overall_auc = roc_auc_score(y_bin, oof)
    log(f"  [{name}] overall OOF AUC = {overall_auc:.5f}")
    return oof, test, {"overall_auc": float(overall_auc),
                         "fold_aucs": [float(a) for a in aucs],
                         "pos": pos, "neg": neg, "spw": float(spw)}


def main() -> None:
    log(f"loading train + test (SMOKE={SMOKE})")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    if SMOKE:
        idx = np.random.RandomState(SEED).choice(len(tr), 30_000, replace=False)
        tr = tr.iloc[idx].reset_index(drop=True)
        te = te.iloc[:10_000].copy().reset_index(drop=True)

    tr = add_distance_features(tr)
    te = add_distance_features(te)
    target = "Irrigation_Need"

    y = tr[target].map(CLS2IDX).to_numpy().astype(np.int8)
    rule = tr["rule_pred"].to_numpy().astype(np.int8)

    cat_cols = [c for c in tr.columns
                  if not pd.api.types.is_numeric_dtype(tr[c]) and c != target]
    log(f"factorizing {len(cat_cols)} cats: {cat_cols}")
    for c in cat_cols:
        s_tr = tr[c].astype(str)
        s_te = te[c].astype(str)
        m = {v: i for i, v in enumerate(sorted(set(s_tr) | set(s_te)))}
        tr[c] = s_tr.map(m).astype(np.int32)
        te[c] = s_te.map(m).fillna(-1).astype(np.int32)

    drop = [target, "id"]
    feats = [c for c in tr.columns if c not in drop]
    X = tr[feats].astype(np.float32).to_numpy()
    X_te = te[feats].astype(np.float32).to_numpy()
    log(f"feature count: {X.shape[1]}")

    aux_targets = {
        "aux_flipped_from_rule": (y != rule).astype(np.int8),
        "aux_missed_high": ((y == 2) & (rule != 2)).astype(np.int8),
        "aux_missed_medium": ((y == 1) & (rule != 1)).astype(np.int8),
    }

    out_meta = {}
    for name, y_bin in aux_targets.items():
        log(f"=== {name} ===")
        oof, test, meta = train_aux(name, X, X_te, y_bin, strat_y=y)
        suffix = "_smoke" if SMOKE else ""
        np.save(ART / f"oof_{name}{suffix}.npy", oof)
        np.save(ART / f"test_{name}{suffix}.npy", test)
        out_meta[name] = meta

    suffix = "_smoke" if SMOKE else ""
    with open(ART / f"multitask_aux{suffix}_results.json", "w") as f:
        json.dump({"smoke": SMOKE, "n_folds": N_FOLDS, "seed": SEED,
                    "aux": out_meta}, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
