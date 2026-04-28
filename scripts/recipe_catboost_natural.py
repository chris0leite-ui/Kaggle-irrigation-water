"""CatBoost natural-calibration rebuild on the V10 recipe FE bank.

Mirror of `recipe_full_te_catboost.py` but adopts the rawashishsin v3
training regime (LB 0.98109, tuned bias [−1.36, −1.19, 0.00]) which
produces NATURALLY CALIBRATED raw probabilities. Knob diff:

| knob          | existing CB (LB 0.97935)    | natural-cal rebuild         |
|---------------|-----------------------------|-----------------------------|
| depth         | 4                           | 3 (rawashishsin parity)     |
| l2_leaf_reg   | 10.0                        | 0.0 (no reg)                |
| iterations    | 2000                        | 2600                        |
| learning_rate | 0.1                         | 0.05                        |
| orig rows     | features-only via mean/std  | CONCAT into training pool   |
| sample_weight | balanced(y_synth)           | balanced(y_combined),       |
|               |                             |   sw[orig_idx] *= 0.5       |
| od_wait       | 200                         | 200 (unchanged)             |
| bootstrap     | Bernoulli (CPU)             | Bernoulli (CPU)             |

Diagnostic gate: bias_H ∈ [−0.5, +1.0] after `tune_log_bias` ⇒ natural
calibration achieved (vs existing CB's bias_H = +2.80).

Per-fold checkpointing for rehydrate resilience (RUN_FOLD env var or
auto-detect cached folds). All artefacts use `_catboost_natural` suffix.

Wall budget: ~30 min CPU (5 folds × ~6 min each at depth=3).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

ORIG_ROW_WEIGHT = 0.5  # rawashishsin parity

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

RUN_FOLD = os.environ.get("RUN_FOLD")
RUN_FOLD = int(RUN_FOLD) if RUN_FOLD else None  # 1-based; None = run all

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)

SUFFIX = "_catboost_natural" + ("_smoke" if SMOKE else "")
# Final aggregate is saved under the canonical recipe-family name so
# downstream blend / meta scripts can find it via the standard pattern.
FINAL_OOF_NAME = ("recipe_full_te_catboost_natural"
                  + ("_smoke" if SMOKE else ""))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_engineer() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
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

    # add_orig_mean_std only writes ORIG_*_mean/std to train+test; we need
    # the same columns on orig itself so we can concat orig as training rows.
    # The lookup is self-referential (orig groupby itself) but bounded by
    # ORIG_ROW_WEIGHT=0.5 in sample_weight. Each cat group has many rows so
    # the per-row mean/std is stable, not a row-self-leak.
    log("adding ORIG mean/std to orig itself (for training-row concat)")
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


def fold_paths(fold: int) -> tuple[Path, Path, Path]:
    return (ART / f"oof{SUFFIX}_fold{fold}.npy",
            ART / f"test{SUFFIX}_fold{fold}.npy",
            ART / f"recipe_full_te{SUFFIX}_fold{fold}.json")


def run_one_fold(fold: int, tr_idx: np.ndarray, va_idx: np.ndarray,
                 train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame,
                 info: dict, y: np.ndarray, y_orig: np.ndarray,
                 a_ote: float = 1.0) -> dict:
    oof_p, test_p, json_p = fold_paths(fold)
    if oof_p.exists() and test_p.exists() and json_p.exists():
        log(f"  fold {fold} cached, loading checkpoints")
        oof_va = np.load(oof_p)
        test_pred = np.load(test_p)
        meta = json.loads(json_p.read_text())
        return dict(va_idx=va_idx, oof_va=oof_va, test_pred=test_pred,
                    bal_acc=meta["bal_acc"], best_iter=meta["best_iter"])

    log(f"=== fold {fold}/{N_FOLDS} ===")
    X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
    X_va = train.iloc[va_idx].copy().reset_index(drop=True)
    X_te = test.copy().reset_index(drop=True)
    X_or = orig.copy().reset_index(drop=True)

    log("  fitting OrderedTE on shuffled X_tr")
    t0 = time.time()
    rng = np.random.default_rng(SEED + fold)
    perm = rng.permutation(len(X_tr))
    X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
    te = OrderedTE(a=a_ote)
    X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
    X_va = te.transform(X_va)
    X_te = te.transform(X_te)
    X_or = te.transform(X_or)
    log(f"    OTE done in {time.time()-t0:.1f}s")

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    feat_cols = numeric_feats + te.te_col_names()

    # Concat synth-fold-train + orig — rawashishsin pattern
    X_combined = pd.concat(
        [X_tr[feat_cols], X_or[feat_cols]], axis=0, ignore_index=True
    )
    y_combined = np.concatenate([y[tr_idx], y_orig])
    sw = compute_sample_weight("balanced", y_combined).astype(np.float32)
    sw[len(tr_idx):] *= ORIG_ROW_WEIGHT
    log(f"  combined train rows: {len(X_combined)} "
        f"(synth {len(tr_idx)} + orig {len(X_or)})  "
        f"orig sw multiplier = {ORIG_ROW_WEIGHT}")

    cb_params = dict(
        iterations=300 if SMOKE else 2600,
        depth=3,                # natural-cal: lower depth (rawashishsin)
        learning_rate=0.05,     # natural-cal: lower lr
        l2_leaf_reg=0.0,        # natural-cal: NO reg
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

    log(f"  training CatBoost (depth=3, no reg, lr=0.05) on "
        f"{len(feat_cols)} features")
    model = CatBoostClassifier(**cb_params)
    model.fit(
        X_combined, y_combined,
        sample_weight=sw,
        eval_set=(X_va[feat_cols], y[va_idx]),
        use_best_model=True,
        verbose=500,
    )
    oof_va = model.predict_proba(X_va[feat_cols]).astype(np.float32)
    test_pred = model.predict_proba(X_te[feat_cols]).astype(np.float32)
    bal = float(balanced_accuracy_score(y[va_idx], oof_va.argmax(1)))
    best_iter = int(model.tree_count_)
    log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={best_iter}")

    # Atomic checkpoint write
    np.save(oof_p, oof_va)
    np.save(test_p, test_pred)
    json_p.write_text(json.dumps({
        "fold": fold, "bal_acc": bal, "best_iter": best_iter,
        "n_combined": int(len(X_combined)),
        "n_synth": int(len(tr_idx)), "n_orig": int(len(X_or)),
        "feat_cols_count": len(feat_cols),
    }, indent=2))
    log(f"  saved fold-{fold} checkpoint -> {oof_p.name}")
    return dict(va_idx=va_idx, oof_va=oof_va, test_pred=test_pred,
                bal_acc=bal, best_iter=best_iter)


def main():
    train, test, orig, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    y_orig = orig[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))

    fold_results = {}
    for f, (tr_idx, va_idx) in enumerate(splits, 1):
        if RUN_FOLD is not None and f != RUN_FOLD:
            continue
        fold_results[f] = run_one_fold(f, tr_idx, va_idx, train, test, orig,
                                        info, y, y_orig)

    if RUN_FOLD is not None:
        log(f"finished single fold {RUN_FOLD}; rerun without RUN_FOLD to "
            f"aggregate when all 5 fold checkpoints exist")
        return

    # Aggregate all folds (require all 5 to exist)
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

    # Calibration diagnostic
    bias_h = float(bias[2])
    if -0.5 <= bias_h <= 1.0:
        cal_verdict = "PASS — natural calibration achieved (bias_H near 0)"
    elif bias_h <= 1.5:
        cal_verdict = "PARTIAL — bias_H below recipe family but above target"
    else:
        cal_verdict = "FAIL — bias_H still high; sklearn TE may be needed (Pick 2b)"
    log(f"natural-cal verdict: {cal_verdict}")

    # Compare vs existing CB and recipe XGB
    old_cb_bias_h = 2.80  # documented in CLAUDE.md / existing results JSON
    log(f"  bias_H drift: existing CB {old_cb_bias_h} -> natural CB {bias_h:.3f}")

    np.save(ART / f"oof_{FINAL_OOF_NAME}.npy", oof)
    np.save(ART / f"test_{FINAL_OOF_NAME}.npy", test_pred)

    # Build submission CSV
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
    )
    out_json = ART / f"recipe_full_te{SUFFIX}_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
