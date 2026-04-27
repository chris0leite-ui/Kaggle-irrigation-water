"""Phase A — fold-safe residual Target Encoding on top of recipe FE.

Pipeline:
  1. Reuse recipe load_and_engineer (full recipe FE pipeline).
  2. Per fold, compute 3 binary residual targets on tr_idx (raw y, score,
     rule_pred all available pre-fold from raw features).
  3. Fit residual-OrderedTE per (target × key) on tr_idx with n_shuffles=8,
     alpha=10. Apply to val + test via full-train-key lookup. ~45 features.
  4. Concatenate with recipe features. Train recipe heavy-reg XGB.
  5. Tune log-bias, save OOF + test + submission CSV.

Output paths suffixed _residte. SMOKE=1 → 20k train + 2 folds + smaller XGB.
RUN_FOLD=N → run only fold N (for rehydrate-resilient sequencing).
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
from recipe_full_te import (  # noqa: E402
    CLS_MAP, IDX2CLS, TARGET, load_and_engineer,
)
from recipe_ote import OrderedTE  # noqa: E402
from residual_te_helpers import (  # noqa: E402
    build_residual_targets, compute_rule_pred_score,
    default_key_specs, fit_residual_ote_block,
)

SEED = 42
N_FOLDS = 5
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2
RUN_FOLD = int(os.environ.get("RUN_FOLD", "0"))  # 0 = all folds

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)
SUFFIX = "_residte"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Phase A residual TE  smoke={SMOKE}  run_fold={RUN_FOLD or 'all'}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    # Pre-fold derivation of dgp_score / rule_pred from RAW columns; recipe
    # has factorized cats so re-load raw. Subsample matched to load_and_engineer
    # so row indices align.
    raw_train = pd.read_csv("data/train.csv")
    raw_test = pd.read_csv("data/test.csv")
    if SMOKE:
        raw_train = raw_train.sample(20_000, random_state=SEED).reset_index(drop=True)
        raw_test = raw_test.sample(10_000, random_state=SEED).reset_index(drop=True)
    assert len(raw_train) == len(train), (len(raw_train), len(train))
    assert len(raw_test) == len(test), (len(raw_test), len(test))
    dgp_score, rule_pred = compute_rule_pred_score(raw_train)
    score_te, rule_te = compute_rule_pred_score(raw_test)
    # Attach as columns so OTE _key_strings can lookup by name on engineered df.
    if "dgp_score" not in train.columns:
        train["dgp_score"] = dgp_score
    if "dgp_score" not in test.columns:
        test["dgp_score"] = score_te
    targets_full = build_residual_targets(y, dgp_score, rule_pred)
    for name, vec in targets_full.items():
        log(f"  target {name}: {int(vec.sum()):,} positives "
            f"({100*vec.mean():.3f}%)")

    keys = default_key_specs(info["combos"], info["digits"])
    log(f"  residual TE keys ({len(keys)}): "
        f"{[' x '.join(k) for k in keys]}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores: list[float] = []

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1, random_state=SEED, verbosity=0,
        early_stopping_rounds=50 if SMOKE else 200,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if RUN_FOLD and fold != RUN_FOLD:
            continue
        log(f"=== fold {fold}/{N_FOLDS} ===")
        ck_oof = ART / f"oof_recipe_full_te{SUFFIX}_fold{fold}.npy"
        ck_te = ART / f"test_recipe_full_te{SUFFIX}_fold{fold}.npy"
        if ck_oof.exists() and ck_te.exists():
            vp = np.load(ck_oof); tp = np.load(ck_te)
            if vp.shape[0] == len(va_idx) and tp.shape[0] == len(test):
                oof[va_idx] = vp; test_pred += tp / N_FOLDS
                bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
                fold_scores.append(bal)
                log(f"  fold {fold} CACHED  bal={bal:.5f}")
                continue
            log(f"  fold {fold} checkpoint shape mismatch ({vp.shape} vs "
                f"{len(va_idx)}, {tp.shape} vs {len(test)}); re-running")
            ck_oof.unlink(); ck_te.unlink()

        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)
        targets_tr = {k: v[tr_idx] for k, v in targets_full.items()}

        # Standard recipe OrderedTE (multi-class y) on cat cols.
        log("  fitting recipe OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        ote = OrderedTE(a=1.0)
        X_tr_shuf = ote.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = ote.transform(X_va); X_te = ote.transform(X_te)
        log(f"    recipe OTE done in {time.time()-t0:.1f}s")

        # Phase A residual OTE block.
        log("  fitting residual OTE")
        t0 = time.time()
        tr_b, va_b, te_b, res_cols = fit_residual_ote_block(
            X_tr, X_va, X_te, targets_tr, keys,
            n_shuffles=2 if SMOKE else 8, alpha=10.0, seed=SEED + fold,
        )
        log(f"    residual OTE done in {time.time()-t0:.1f}s ({len(res_cols)} cols)")
        for i, c in enumerate(res_cols):
            X_tr[c] = tr_b[:, i]; X_va[c] = va_b[:, i]; X_te[c] = te_b[:, i]

        feat_cols = numeric_feats + ote.te_col_names() + res_cols
        sw = compute_sample_weight("balanced", y[tr_idx])
        log(f"  training XGB on {len(feat_cols)} feats, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        vp = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        tp = model.predict_proba(X_te[feat_cols]).astype(np.float32)
        np.save(ck_oof, vp); np.save(ck_te, tp)
        oof[va_idx] = vp; test_pred += tp / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} bal={bal:.5f} best_iter={model.best_iteration}")

    if RUN_FOLD:
        log(f"RUN_FOLD={RUN_FOLD} — partial run, exiting before aggregate")
        return

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax={overall:.5f}  tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_recipe_full_te{SUFFIX}.npy", oof)
    np.save(ART / f"test_recipe_full_te{SUFFIX}.npy", test_pred)
    eps = 1e-9
    test_idx = (np.log(np.clip(test_pred, eps, 1.0)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_idx]})
    sub_path = SUB / f"submission_recipe_full_te{SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    summary = dict(
        n_folds=N_FOLDS, smoke=SMOKE,
        residual_te_keys=[" x ".join(k) for k in keys],
        residual_te_n_features=3 * len(keys),
        target_pos_rates={k: float(v.mean()) for k, v in targets_full.items()},
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
    )
    with open(ART / f"recipe_full_te{SUFFIX}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote results JSON")


if __name__ == "__main__":
    main()
