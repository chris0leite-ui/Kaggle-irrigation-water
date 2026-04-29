"""rawashishsin XGB clone on our V10 recipe FE.

Mirrors rawashishsin v3's exact XGB HPs (depth=3, lr=0.05, n_est=2600,
no L1/L2 reg, max_bin=1100, subsample=0.9, colsample_bytree=0.8,
class-balanced sample weights + ORIG_ROW_WEIGHT=0.5) but swaps in OUR
V10 recipe FE bank (~440 features: cats + combos + digits + num_as_cat
+ tres + logits + freq + orig_stats + sklearn TE on the 117 cat-tuples).

Hypothesis: rawashishsin's natural calibration property (bias drift
≤ 0.4 from -log(prior)) comes from the HP regime (low depth + no reg
+ low lr + smoothed TE) rather than the FE choice. If true, applying
the same regime to a richer FE bank should produce:
  - Same natural-cal bias profile
  - HIGHER OOF (more features = more capacity)
  - Cleaner orthogonality with rawashishsin v3 (different FE)

Per-fold StratifiedKFold(seed=42) aligned with every saved OOF.
sklearn TargetEncoder(cv=5, smooth='auto') applied per outer fold on
(synth_train ∪ orig) with internal cross-fitting. Test transform uses
the full per-outer-fold TE.

Wall budget: ~50-60 min CPU (5 folds × ~10 min, XGB hist + max_bin=1100
is a bit slow on CPU but doable). Per-fold checkpointing.
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
import xgboost as xgb
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

SUFFIX = "_xgb_skte" + ("_smoke" if SMOKE else "")
FINAL_OOF_NAME = ("recipe_full_te_xgb_skte"
                  + ("_smoke" if SMOKE else ""))


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    log("recipe FE: threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)
    log("recipe FE: cat-pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)
    log("recipe FE: digit features")
    digits = add_digit_features(train, test, orig, nums)
    log("recipe FE: num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    log("recipe FE: FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)
    log("recipe FE: ORIG mean/std")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # Add ORIG mean/std to orig itself (for training-row concat)
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
    log(f"  feature groups: cats={len(cats)} combos={len(combos)} "
        f"digits={len(digits)} num_as_cat={len(num_as_cat)} "
        f"tres={len(tres)} logits={len(logits)} freq={len(freq)} "
        f"orig_stats={len(orig_stats_cols)} te_cols={len(info['te_cols'])}")
    return train, test, orig, info, test_ids


def fold_paths(fold):
    return (ART / f"oof{SUFFIX}_fold{fold}.npy",
            ART / f"test{SUFFIX}_fold{fold}.npy",
            ART / f"recipe_full_te{SUFFIX}_fold{fold}.json")


def run_one_fold(fold, tr_idx, va_idx, train, test, orig, info, y, y_orig):
    oof_p, test_p, json_p = fold_paths(fold)
    if oof_p.exists() and test_p.exists() and json_p.exists():
        log(f"  fold {fold} cached, skipping")
        return

    log(f"=== fold {fold}/{N_FOLDS} ===")
    X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
    X_va = train.iloc[va_idx].copy().reset_index(drop=True)
    X_te = test.copy().reset_index(drop=True)
    X_or = orig.copy().reset_index(drop=True)

    te_cols = info["te_cols"]
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    # sklearn TargetEncoder fit on (synth-train ∪ orig)
    log(f"  fitting sklearn TargetEncoder(multiclass, cv=5) on "
        f"{len(te_cols)} cat-tuples")
    t0 = time.time()
    X_tr_combined_cat = pd.concat(
        [X_tr[te_cols], X_or[te_cols]], axis=0, ignore_index=True
    ).to_numpy()
    y_combined = np.concatenate([y[tr_idx], y_orig])
    te = TargetEncoder(
        target_type="multiclass", cv=5, smooth="auto",
        random_state=SEED + fold,
    )
    te_tr_combined = te.fit_transform(X_tr_combined_cat, y_combined)
    te_va = te.transform(X_va[te_cols].to_numpy())
    te_te = te.transform(X_te[te_cols].to_numpy())
    n_te_out = te_tr_combined.shape[1]
    log(f"    sklearn TE done in {time.time()-t0:.1f}s  ({n_te_out} cols)")

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

    X_combined = np.concatenate([X_tr_full, X_or_full], axis=0)
    sw = compute_sample_weight("balanced", y_combined).astype(np.float32)
    sw[n_synth:] *= ORIG_ROW_WEIGHT
    log(f"  combined train: {len(X_combined)} rows  "
        f"(synth {n_synth} + orig {len(X_or_full)})  "
        f"orig sw_mult={ORIG_ROW_WEIGHT}")

    # rawashishsin's exact XGB HPs
    n_est = 300 if SMOKE else 2600
    params = dict(
        objective="multi:softprob", num_class=3,
        n_estimators=n_est, learning_rate=0.05,
        max_depth=3, subsample=0.9, colsample_bytree=0.8,
        max_bin=1100, eval_metric="mlogloss",
        n_jobs=-1, random_state=SEED + fold,
        tree_method="hist",  # CPU-compatible (no device='cuda')
    )

    log(f"  training XGB (depth=3, lr=0.05, max_bin=1100) on "
        f"{X_tr_full.shape[1]} features")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_combined, y_combined,
        sample_weight=sw,
        eval_set=[(X_va_full, y[va_idx])],
        verbose=500,
    )
    oof_va = model.predict_proba(X_va_full).astype(np.float32)
    test_pred = model.predict_proba(X_te_full).astype(np.float32)
    bal = float(balanced_accuracy_score(y[va_idx], oof_va.argmax(1)))
    log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
        f"best_iter={model.best_iteration if hasattr(model, 'best_iteration') else n_est}")

    np.save(oof_p, oof_va)
    np.save(test_p, test_pred)
    json_p.write_text(json.dumps({
        "fold": fold, "bal_acc": bal,
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
        log(f"finished single fold {RUN_FOLD}")
        return

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    for f, (_, va_idx) in enumerate(splits, 1):
        oof_p, test_p, json_p = fold_paths(f)
        if not (oof_p.exists() and test_p.exists() and json_p.exists()):
            log(f"  fold {f} checkpoint missing, aborting aggregation")
            return
        oof[va_idx] = np.load(oof_p)
        test_pred += np.load(test_p) / N_FOLDS
        meta = json.loads(json_p.read_text())
        fold_scores.append(meta["bal_acc"])

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    # Bias drift diagnostic
    drift = bias - (-np.log(prior))
    drift_max = float(np.abs(drift).max())
    log(f"  bias drift from -log(prior): {drift.round(4).tolist()}")
    log(f"  max drift magnitude: {drift_max:.3f}")
    if drift_max <= 0.3:
        cal_verdict = "PASS — natural calibration achieved (uniform small drift)"
    elif drift_max <= 0.7:
        cal_verdict = "PARTIAL — drift still meaningful but better than recipe-family"
    else:
        cal_verdict = "FAIL — drift > 0.7 on at least one class"
    log(f"natural-cal verdict: {cal_verdict}")

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
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        bias_drift=drift.tolist(),
        drift_max=drift_max,
        cal_verdict=cal_verdict,
        ORIG_ROW_WEIGHT=ORIG_ROW_WEIGHT,
        encoder="sklearn TargetEncoder(multiclass, cv=5, smooth=auto)",
        model="XGBoost depth=3, lr=0.05, max_bin=1100, no reg (rawashishsin parity)",
    )
    out_json = ART / f"{FINAL_OOF_NAME}_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
