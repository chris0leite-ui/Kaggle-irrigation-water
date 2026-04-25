"""5-fold CV with per-fold SMOTE-NC + redrive + OTE + XGB + promise gate.

Persists OOF/test/JSON after every fold so abort or rehydrate retains progress.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from recipe_ote import OrderedTE  # noqa
from smote_local.redrive import smote_nc_on_raw, redrive_fe  # noqa
from smote_local.gate import evaluate as gate_eval  # noqa
from smote_local.load_engineer import TARGET, log  # noqa


SEED = 42


def run_cv(train, test, raw_train, info, maps, *,
           n_folds=5, max_folds=None,
           smote_target=42000, smote_k=5,
           xgb_params=None, total_kill_sec=None,
           art_dir="scripts/artifacts", suffix="smote_v2"):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    art = Path(art_dir); art.mkdir(exist_ok=True, parents=True)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    cats = info["cats"]; nums = info["nums"]
    if max_folds is None:
        max_folds = n_folds

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores, fold1_metrics = [], None
    gate_decision, folds_completed = None, 0
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if fold > max_folds:
            log(f"reached max_folds={max_folds}, stopping")
            break
        if total_kill_sec is not None and (time.time() - t0) > total_kill_sec:
            log(f"!! TOTAL_KILL ({total_kill_sec}s) reached, partial save")
            break

        log(f"=== fold {fold}/{n_folds} (max={max_folds}) ===")
        t_fold = time.time()
        raw_tr = raw_train.iloc[tr_idx].copy().reset_index(drop=True)
        y_tr_full = raw_tr[TARGET].to_numpy()
        raw_tr_nolab = raw_tr.drop(columns=[TARGET])
        log(f"  raw fold-tr: {len(raw_tr_nolab):,} × "
            f"{raw_tr_nolab.shape[1]} (8 cats + 11 nums)")

        # 1. SMOTE-NC on raw 19 cols
        t1 = time.time()
        try:
            raw_aug, y_aug = smote_nc_on_raw(
                raw_tr_nolab, y_tr_full, smote_target,
                k=smote_k, random_state=SEED + fold)
        except Exception as e:
            log(f"  SMOTE failed: {e}; skip aug this fold")
            raw_aug, y_aug = raw_tr_nolab.copy(), y_tr_full.copy()
        n_h_pre = int((y_tr_full == 2).sum())
        n_h_post = int((y_aug == 2).sum())
        log(f"  SMOTE-NC: {len(raw_tr_nolab):,} → {len(raw_aug):,}  "
            f"(H {n_h_pre:,} → {n_h_post:,})  wall={time.time()-t1:.1f}s")

        # 2. FE re-derive on aug
        t1 = time.time()
        train_aug_fe = redrive_fe(
            raw_aug, cats=cats, nums=nums,
            combo_pairs=maps["combo_pairs"], combo_maps=maps["combo_maps"],
            nac_maps=maps["nac_maps"], freq_maps=maps["freq_maps"],
            orig_stat_maps=maps["orig_stat_maps"], cat_maps=maps["cat_maps"],
            digit_cols_keep=maps["digit_cols"])
        train_aug_fe[TARGET] = y_aug
        log(f"  FE redrive: {len(train_aug_fe):,} × {train_aug_fe.shape[1]}  "
            f"wall={time.time()-t1:.1f}s")

        # 3. OTE on aug train (per-fold leak-free w.r.t. val)
        t1 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(train_aug_fe))
        X_tr_shuf = train_aug_fe.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr_aug = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(train.iloc[va_idx].reset_index(drop=True))
        X_te = te.transform(test.copy().reset_index(drop=True))
        log(f"  OTE: wall={time.time()-t1:.1f}s")

        # 4. XGB
        feat_cols = numeric_feats + te.te_col_names()
        y_aug_arr = X_tr_aug[TARGET].to_numpy()
        sw = compute_sample_weight("balanced", y_aug_arr)
        log(f"  XGB train: {len(feat_cols)} feat × {len(X_tr_aug):,} rows")
        t1 = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr_aug[feat_cols], y_aug_arr, sample_weight=sw,
                  eval_set=[(X_va[feat_cols], y[va_idx])], verbose=500)
        log(f"  XGB done best_iter={model.best_iteration} wall={time.time()-t1:.1f}s")

        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / n_folds
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        folds_completed += 1
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  total={time.time()-t_fold:.1f}s")

        # Persist after every fold
        np.save(art / f"oof_{suffix}.npy", oof)
        np.save(art / f"test_{suffix}.npy",
                test_pred * n_folds / max(folds_completed, 1))

        # Promise gate after fold 1
        if fold == 1:
            decision, fold1_metrics = gate_eval(
                oof[va_idx], y[va_idx], len(va_idx) // 5)
            log(f"\n=== fold-1 PROMISE GATE ===")
            for k, v in fold1_metrics.items():
                log(f"  {k:20s} = {v}")
            gate_decision = decision
            (art / f"{suffix}_fold1_gate.json").write_text(
                json.dumps(fold1_metrics, indent=2))
            if decision == "ABORT":
                log("=== ABORTING after fold 1 ===")
                break

    if folds_completed > 0:
        test_pred = test_pred * n_folds / max(folds_completed, 1)
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                folds_completed=folds_completed,
                fold1_metrics=fold1_metrics,
                gate_decision=gate_decision)
