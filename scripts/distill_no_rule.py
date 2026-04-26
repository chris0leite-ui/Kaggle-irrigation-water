"""Experiment C: feature-restricted distillation student.

Forces the tree ensemble to discover the LB-best 4-stack decision surface
from a feature basis that EXCLUDES rule-derived signals (the 4 binary
threshold flags `soil_lt_25/temp_gt_30/rain_lt_300/wind_gt_10` and the 3
LR-formula logits `logit_P_{Low,Medium,High}`). Reuses the recipe pipeline's
FE infrastructure for everything else (cats, digits, num_as_cat, freq,
orig_stats, OTE on cats+combos+digits+num_as_cat).

Two target modes via env var DISTILL_TARGET:
  ""  | "true"      — use original train labels y (default, cleanest test)
  "lb_pseudo"      — use argmax(LB-best 4-stack OOF) as hard pseudo-labels

5-fold StratifiedKFold(seed=42), aligned with every other OOF on disk.
Outputs:
  oof_distill_no_rule[_<target>].npy
  test_distill_no_rule[_<target>].npy
  distill_no_rule[_<target>]_results.json

SMOKE=1 → 20k train, 2 folds, 300 rounds (~3 min CPU).
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, CLS_MAP  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402
from tier1b_helpers import build_lbbest_stack, BIAS  # noqa: E402

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
DISTILL_TARGET = os.environ.get("DISTILL_TARGET", "true")
assert DISTILL_TARGET in ("true", "lb_pseudo"), DISTILL_TARGET
SUFFIX = "" if DISTILL_TARGET == "true" else f"_{DISTILL_TARGET}"

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"config: DISTILL_TARGET={DISTILL_TARGET!r}  SMOKE={SMOKE}  N_FOLDS={N_FOLDS}")
    train, test, info, test_ids = load_and_engineer()
    y_true = train[TARGET].to_numpy().astype(np.int32)
    n = len(train)

    if DISTILL_TARGET == "lb_pseudo":
        if SMOKE:
            log("SMOKE+lb_pseudo: synthesising fake pseudo (random) for plumbing test")
            rng = np.random.default_rng(SEED)
            y = rng.integers(0, 3, size=n).astype(np.int32)
        else:
            log("loading LB-best 4-stack hard pseudo-argmax labels")
            lb_oof, _ = build_lbbest_stack(y_true)
            y = lb_oof.argmax(1).astype(np.int32)
            log(f"  pseudo vs true: agreement={(y == y_true).mean():.5f} "
                f"({(y != y_true).sum():,} disagree)")
    else:
        y = y_true

    # Feature subset: drop tres + logits from numeric_feats AND from te_cols.
    # Rationale: tres are 4 rule-flag booleans, logits are LR-formula posterior
    # scores — both directly encode the rule. Removing them forces XGB to
    # rediscover rule-equivalent splits via cats + digits + nums alone.
    drop_num_feats = set(info["tres"]) | set(info["logits"])
    drop_te_cols = set(info["tres"])  # tres also appear in te_cols (encoded as cats)
    numeric_feats = [c for c in (
        info["nums"] + info["freq"] + info["orig_stats"]
    ) if c not in drop_num_feats]
    te_cols = [c for c in info["te_cols"] if c not in drop_te_cols]
    log(f"feature subset: numeric={len(numeric_feats)}  te_cols={len(te_cols)}  "
        f"(dropped tres={len(info['tres'])} + logits={len(info['logits'])})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((n, 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores: list[float] = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    # Per-fold checkpoint pattern (rehydrate-resilient): saves each fold's
    # val OOF + test pred immediately after training. On rerun, completed folds
    # are loaded from disk and skipped.
    ck_prefix = f"distill_no_rule{SUFFIX}"
    cached: set[int] = set()
    for fold_check in range(1, N_FOLDS + 1):
        ck_oof = ART / f"oof_{ck_prefix}_fold{fold_check}.npy"
        ck_test = ART / f"test_{ck_prefix}_fold{fold_check}.npy"
        if ck_oof.exists() and ck_test.exists():
            cached.add(fold_check)
    if cached:
        log(f"  resume: {len(cached)} fold(s) cached: {sorted(cached)}")

    # Stratify on TRUE y so val-fold class composition is consistent with the
    # downstream blend gate (which evaluates against true labels).
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_true), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        if fold in cached:
            ck_oof = ART / f"oof_{ck_prefix}_fold{fold}.npy"
            ck_test = ART / f"test_{ck_prefix}_fold{fold}.npy"
            vp = np.load(ck_oof)
            tp = np.load(ck_test)
            oof[va_idx] = vp
            test_pred += tp / N_FOLDS
            bal = balanced_accuracy_score(y_true[va_idx], vp.argmax(1))
            fold_scores.append(bal)
            log(f"  fold {fold} CACHED  argmax_bal_acc = {bal:.5f}")
            continue
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)
        # Override the TARGET column on tr/va/te for OrderedTE — it must see
        # the (pseudo or true) label we're distilling on, not the raw y.
        X_tr[TARGET] = y[tr_idx]
        X_va[TARGET] = y[va_idx]  # only used as a placeholder; OTE doesn't read va
        X_te[TARGET] = -1  # unused

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=te_cols, target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr = y[tr_idx].copy()
        sw = compute_sample_weight("balanced", y_tr)

        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y_tr,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y_true[va_idx])],
            verbose=500,
        )
        vp = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        tp = model.predict_proba(X_te[feat_cols]).astype(np.float32)
        oof[va_idx] = vp
        test_pred += tp / N_FOLDS
        # checkpoint immediately for rehydrate resilience
        np.save(ART / f"oof_{ck_prefix}_fold{fold}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold}.npy", tp)
        bal = balanced_accuracy_score(y_true[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y_true, oof.argmax(1))
    log(f"=== OOF argmax bal_acc (vs TRUE y) = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    prior = np.bincount(y_true, minlength=3) / len(y_true)
    bias, tuned = tune_log_bias(oof, y_true, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_distill_no_rule{SUFFIX}.npy", oof)
    np.save(ART / f"test_distill_no_rule{SUFFIX}.npy", test_pred)
    out = dict(
        config=dict(target=DISTILL_TARGET, smoke=SMOKE, n_folds=N_FOLDS,
                    n_features=len(feat_cols), seed=SEED),
        fold_scores=fold_scores,
        overall_argmax=float(overall),
        tuned_bal=float(tuned),
        tuned_bias=bias.tolist(),
    )
    with open(ART / f"distill_no_rule{SUFFIX}_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote oof_distill_no_rule{SUFFIX}.npy + test + results.json")


if __name__ == "__main__":
    main()
