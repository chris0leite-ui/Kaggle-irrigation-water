"""B2 GroupKFold diagnostic for recipe_full_te.

Re-splits the 630k training set by Region (default) or Crop_Type
(via GROUP=crop env var), to test whether our StratifiedKFold(seed=42)
baseline is leaking across region/crop via OTE group statistics.

Hypothesis: the recipe's OTE encodings use leave-one-fold-out means
over the train pool. If Region/Crop_Type columns correlate with y
in a way StratifiedKFold fails to hold out (e.g. "Region=South"
consistently over-predicts High), OOF could be optimistic. Splitting
so that every val fold has entirely unseen region values forces a
harder generalization check.

Expected outcomes:
  - If OOF drops materially (≥0.005 tuned bal_acc): we have leakage,
    the 0.98005 LB-best ladder overstates the true frontier, and the
    apparent stacking-inflation ceiling is partly CV artifact.
  - If OOF holds (<0.002 drop): StratifiedKFold is honest; our
    ceiling is real structural saturation.

5 regions × 5-fold GroupKFold → each fold validates exactly 1 region.
6 crops × 5-fold → some folds get 1 crop, some get 2 (via the sklearn
allocator). Either is a stricter split than StratifiedKFold.

SMOKE=1 → 20k rows, 2 folds (which means 2 groups → simple).
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
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
GROUP = os.environ.get("GROUP", "region").lower()  # 'region' | 'crop'
GROUP_COL = {"region": "Region", "crop": "Crop_Type"}[GROUP]

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)

SUFFIX = f"_{GROUP}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"B2 GroupKFold diagnostic  group={GROUP} col={GROUP_COL} smoke={SMOKE}")

    # Load raw group labels BEFORE FE (recipe_features factorizes cats).
    group_raw = pd.read_csv("data/train.csv", usecols=[GROUP_COL])[GROUP_COL].values
    log(f"group cardinality = {len(set(group_raw))}  "
        f"dist = {pd.Series(group_raw).value_counts().to_dict()}")

    train, test, info, test_ids = load_and_engineer()

    if SMOKE:
        # load_and_engineer internally respects SMOKE; re-align groups
        # by loading the same SMOKE rows.
        sub_n = len(train)
        group = group_raw[:sub_n]
    else:
        assert len(train) == len(group_raw), (len(train), len(group_raw))
        group = group_raw

    y = train[TARGET].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"]
                     + info.get("dae_embed", [])
                     + info.get("extra_domain", [])
                     + info.get("extra_decimal", []))

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss",
        enable_categorical=False, n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, group), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        log(f"  tr={len(tr_idx):,}  va={len(va_idx):,}  "
            f"val groups={sorted(set(group[va_idx]))}")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr[feat_cols], y[tr_idx], sample_weight=sw,
                  eval_set=[(X_va[feat_cols], y[va_idx])], verbose=500)
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    oof_path = ART / f"oof_b2_groupkfold{SUFFIX}.npy"
    test_path = ART / f"test_b2_groupkfold{SUFFIX}.npy"
    np.save(oof_path, oof)
    np.save(test_path, test_pred)
    log(f"wrote {oof_path} + {test_path}")

    # Compare vs StratifiedKFold baseline on disk.
    baseline = json.loads((ART / "recipe_full_te_results.json").read_text())
    baseline_tuned = baseline["tuned_log_bias_bal_acc"]
    log(f"Δ tuned OOF vs StratifiedKFold baseline "
        f"({baseline_tuned:.5f}) = {tuned - baseline_tuned:+.5f}")

    eps = 1e-9
    test_log = np.log(np.clip(test_pred, eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / f"submission_b2_groupkfold{SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, group=GROUP, group_col=GROUP_COL,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        baseline_tuned=baseline_tuned,
        delta_vs_baseline=tuned - baseline_tuned,
    )
    results_path = ART / f"b2_groupkfold{SUFFIX}_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {results_path}")


if __name__ == "__main__":
    main()
