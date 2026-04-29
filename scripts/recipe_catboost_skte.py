"""Pick 2b: CatBoost natural-cal rebuild with sklearn TargetEncoder(cv=5)
replacing OrderedTE.

Triggered by Phase 1 (recipe_catboost_natural.py) producing bias_H ≈ 2.6
even with depth=3 + no reg + lr=0.05 + ORIG_ROW_WEIGHT=0.5 — confirming
that the natural-calibration mechanism in rawashishsin v3 (LB 0.98109,
bias_H = 0.00) comes from sklearn's CV-shuffled smoothing rather than
the training-regime knobs alone.

OrderedTE: per-row exclusive cumulative on shuffled data, no internal
CV. Pure noise injection.

sklearn TargetEncoder(target_type='multiclass', cv=5, smooth='auto'):
internal 5-fold cross-fitting on training data; smooth='auto' applies
empirical-Bayes shrinkage per category. Different mechanism: each
training row gets a TE value computed from the OTHER 4/5 of training
data (no row-self-leak), and the smoothing replaces the function L1/L2
reg plays in our recipe.

Inputs: same 117 cat-tuples (cats + combos + digits + num_as_cat + tres)
that OrderedTE encoded → 117 × 3 = 351 numeric features. Combined with
85 numerics (raws + tres + logits + freq + orig_stats) → 436 total.

XGB-style heavy-reg dropped (no reg_alpha/reg_lambda). depth=3, lr=0.05,
iter=2600. ORIG_ROW_WEIGHT=0.5. Same 5-fold StratifiedKFold(seed=42).

Diagnostic gate: bias_H ∈ [-0.5, +1.0] after tune_log_bias. PASS expected
based on rawashishsin parity hypothesis.

Wall budget: ~50 min CPU (5 folds × ~10 min, sklearn TE adds ~3-5 min/fold
vs OrderedTE).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

ORIG_ROW_WEIGHT = 0.5

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

RUN_FOLD = os.environ.get("RUN_FOLD")
RUN_FOLD = int(RUN_FOLD) if RUN_FOLD else None

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)

SUFFIX = "_catboost_skte" + ("_smoke" if SMOKE else "")
FINAL_OOF_NAME = ("recipe_full_te_catboost_skte"
                  + ("_smoke" if SMOKE else ""))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_engineer():
    log("loading train / test / orig")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    log("adding threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    log("adding cat x cat pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)

    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)

    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)

    log("adding ORIG mean/std per col")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)
    log("adding ORIG mean/std to orig itself")
    for c in nums + cats:
        stats = orig.groupby(c)[TARGET].agg(["mean", "std"]).reset_index()
        stats.columns = [c, f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        merged = orig.merge(stats, on=c, how="left")
        orig[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
        orig[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats_cols)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, orig, info, test_ids


def fold_paths(fold: int):
    return (ART / f"oof{SUFFIX}_fold{fold}.npy",
            ART / f"test{SUFFIX}_fold{fold}.npy",
            ART / f"recipe_full_te{SUFFIX}_fold{fold}.json")


def run_one_fold(fold, tr_idx, va_idx, train, test, orig, info, y, y_orig):
    oof_p, test_p, json_p = fold_paths(fold)
    if oof_p.exists() and test_p.exists() and json_p.exists():
        log(f"  fold {fold} cached, loading checkpoints")
        return  # caller aggregates from disk

    log(f"=== fold {fold}/{N_FOLDS} ===")
    X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
    X_va = train.iloc[va_idx].copy().reset_index(drop=True)
    X_te = test.copy().reset_index(drop=True)
    X_or = orig.copy().reset_index(drop=True)

    te_cols = info["te_cols"]
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    # sklearn TargetEncoder: fit on (synth-train ∪ orig) with combined y.
    # cv=5 internal cross-fitting → each training row gets a TE value
    # from the OTHER 4/5 of training data (no row-self-leak).
    log(f"  fitting sklearn TargetEncoder(target_type='multiclass', cv=5, "
        f"smooth='auto') on {len(te_cols)} cat-tuples")
    t0 = time.time()
    X_tr_combined_cat = pd.concat(
        [X_tr[te_cols], X_or[te_cols]], axis=0, ignore_index=True
    ).to_numpy()
    y_combined = np.concatenate([y[tr_idx], y_orig])
    te = TargetEncoder(
        target_type="multiclass",
        cv=5,
        smooth="auto",
        random_state=SEED + fold,
    )
    te_tr_combined = te.fit_transform(X_tr_combined_cat, y_combined)
    te_va = te.transform(X_va[te_cols].to_numpy())
    te_te = te.transform(X_te[te_cols].to_numpy())
    n_te_out = te_tr_combined.shape[1]  # = len(te_cols) * 3
    log(f"    sklearn TE done in {time.time()-t0:.1f}s  "
        f"(output cols = {n_te_out})")

    # Assemble feature matrices: numeric_feats columns + TE outputs
    X_tr_num = X_tr[numeric_feats].to_numpy(dtype=np.float32)
    X_va_num = X_va[numeric_feats].to_numpy(dtype=np.float32)
    X_te_num = X_te[numeric_feats].to_numpy(dtype=np.float32)
    X_or_num = X_or[numeric_feats].to_numpy(dtype=np.float32)

    n_synth = len(tr_idx)
    te_tr = te_tr_combined[:n_synth]
    te_or = te_tr_combined[n_synth:]
    X_tr_full = np.concatenate([X_tr_num, te_tr], axis=1).astype(np.float32)
    X_va_full = np.concatenate([X_va_num, te_va], axis=1).astype(np.float32)
    X_te_full = np.concatenate([X_te_num, te_te], axis=1).astype(np.float32)
    X_or_full = np.concatenate([X_or_num, te_or], axis=1).astype(np.float32)

    # Concat synth-fold-train + orig (rawashishsin pattern)
    X_combined = np.concatenate([X_tr_full, X_or_full], axis=0)
    sw = compute_sample_weight("balanced", y_combined).astype(np.float32)
    sw[n_synth:] *= ORIG_ROW_WEIGHT
    log(f"  combined train rows: {len(X_combined)} "
        f"(synth {n_synth} + orig {len(X_or_full)})  "
        f"orig sw multiplier = {ORIG_ROW_WEIGHT}")

    cb_params = dict(
        iterations=300 if SMOKE else 2600,
        depth=3,
        learning_rate=0.05,
        l2_leaf_reg=0.0,
        subsample=0.8,
        rsm=0.8,
        min_data_in_leaf=2,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED + fold,
        od_type="Iter",
        od_wait=50 if SMOKE else 200,
        bootstrap_type="Bernoulli",
        verbose=False,
        task_type="CPU",
        thread_count=-1,
        allow_writing_files=False,
    )

    log(f"  training CatBoost on {X_tr_full.shape[1]} features")
    model = CatBoostClassifier(**cb_params)
    model.fit(
        X_combined, y_combined,
        sample_weight=sw,
        eval_set=(X_va_full, y[va_idx]),
        use_best_model=True,
        verbose=500,
    )
    oof_va = model.predict_proba(X_va_full).astype(np.float32)
    test_pred = model.predict_proba(X_te_full).astype(np.float32)
    bal = float(balanced_accuracy_score(y[va_idx], oof_va.argmax(1)))
    best_iter = int(model.tree_count_)
    log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={best_iter}")

    np.save(oof_p, oof_va)
    np.save(test_p, test_pred)
    json_p.write_text(json.dumps({
        "fold": fold, "bal_acc": bal, "best_iter": best_iter,
        "n_combined": int(len(X_combined)),
        "n_synth": int(n_synth), "n_orig": int(len(X_or_full)),
        "n_features_total": int(X_tr_full.shape[1]),
        "n_te_out": int(n_te_out),
    }, indent=2))
    log(f"  saved fold-{fold} checkpoint -> {oof_p.name}")


def main():
    train, test, orig, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    y_orig = orig[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    for f, (tr_idx, va_idx) in enumerate(splits, 1):
        if RUN_FOLD is not None and f != RUN_FOLD:
            continue
        run_one_fold(f, tr_idx, va_idx, train, test, orig, info, y, y_orig)

    if RUN_FOLD is not None:
        log(f"finished single fold {RUN_FOLD}; rerun without RUN_FOLD to "
            f"aggregate when all 5 fold checkpoints exist")
        return

    # Aggregate
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    best_iters = []
    for f, (_, va_idx) in enumerate(splits, 1):
        oof_p, test_p, json_p = fold_paths(f)
        if not (oof_p.exists() and test_p.exists() and json_p.exists()):
            log(f"  fold {f} checkpoint missing, aborting aggregation")
            return
        oof[va_idx] = np.load(oof_p)
        test_pred += np.load(test_p) / N_FOLDS
        meta = json.loads(json_p.read_text())
        fold_scores.append(meta["bal_acc"])
        best_iters.append(meta["best_iter"])

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    bias_h = float(bias[2])
    if -0.5 <= bias_h <= 1.0:
        cal_verdict = "PASS — natural calibration achieved"
    elif bias_h <= 1.5:
        cal_verdict = "PARTIAL — bias_H below recipe family but above target"
    elif bias_h <= 2.5:
        cal_verdict = "partial — between recipe and target"
    else:
        cal_verdict = "FAIL — bias_H still high"
    log(f"natural-cal verdict: {cal_verdict}")
    log(f"  vs natural CB (OrderedTE): bias_H ~2.6 (PARTIAL/FAIL)")
    log(f"  vs rawashishsin v3:        bias_H = 0.00 (target)")

    np.save(ART / f"oof_{FINAL_OOF_NAME}.npy", oof)
    np.save(ART / f"test_{FINAL_OOF_NAME}.npy", test_pred)

    eps = 1e-9
    test_log = np.log(np.clip(test_pred, eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = SUB / f"submission_{FINAL_OOF_NAME}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in fold_scores],
        best_iters=best_iters,
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        bias_H=bias_h,
        cal_verdict=cal_verdict,
        ORIG_ROW_WEIGHT=ORIG_ROW_WEIGHT,
        n_train=len(train), n_orig=len(orig), n_test=len(test),
        encoder="sklearn TargetEncoder(multiclass, cv=5, smooth=auto)",
    )
    out_json = ART / f"recipe_full_te{SUFFIX}_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
