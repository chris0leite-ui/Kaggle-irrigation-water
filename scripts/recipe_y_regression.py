"""A3 stacked-regression: XGBRegressor on y_ord ∈ {0,1,2} with rmse loss.

Mirrors recipe_full_te_lgbm.py FE pipeline (cat-pair combos + digits +
num-as-cat + threshold flags + LR-formula logits + FREQ + ORIG mean/std
+ OrderedTE on ~117 cats). Same 5-fold StratifiedKFold(seed=42) for
OOF alignment with every saved OOF.

Difference from recipe_full_te.py:
  - Output: continuous prediction in ℝ (instead of multi:softprob)
  - Loss: reg:squarederror on y_ord (instead of multi-class CE on one-hot)
  - Per-leaf optimization minimizes MSE of leaf mean vs ordinal y
    (instead of softmax-CE-optimal class probabilities)

Hypothesis: the rmse-on-ordinal-y gradient produces a structurally
different decision surface than multi-class softprob. This tests whether
the LB-best ceiling is bounded by the OUTPUT FORMULATION rather than
just FE / model class.

Outputs (continuous OOF + test):
  scripts/artifacts/oof_recipe_y_regression.npy   shape (n_train,)  float32
  scripts/artifacts/test_recipe_y_regression.npy  shape (n_test,)   float32
  Plus 3-class derived probs via Gaussian kernel for blend gate eval:
  scripts/artifacts/oof_recipe_y_regression_3cls.npy   (n_train, 3)
  scripts/artifacts/test_recipe_y_regression_3cls.npy  (n_test, 3)
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
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


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
        log("SMOKE=1 — subsampling 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]
    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")
    log("adding threshold flags + LR logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)
    log("adding cat x cat pair combos / digits / num_as_cat / FREQ / ORIG_mean_std")
    combos = add_cat_pair_combos(train, test, orig, cats)
    digits = add_digit_features(train, test, orig, nums)
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    freq = add_freq_features(train, test, orig, cats + combos)
    orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]; test[c] = codes[s:t]; orig[c] = codes[t:]
    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} combos={len(combos)} "
        f"digits={len(digits)} num_as_cat={len(num_as_cat)} tres={len(tres)} "
        f"logits={len(logits)} freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


def run_cv(train, test, info, a_ote=1.0):
    y = train[TARGET].to_numpy()  # int {0, 1, 2} = ordinal target
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    oof = np.zeros(len(train), dtype=np.float32)
    test_pred = np.zeros(len(test), dtype=np.float32)
    fold_rmses, best_iters = [], []
    feat_cols: list[str] = []  # populated when a fold is actually trained
    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="reg:squarederror", tree_method="hist",
        eval_metric="rmse", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        # Per-fold rehydrate-resilient checkpoint: skip if both files exist.
        fold_oof_path = ART / f"oof_recipe_y_regression_fold{fold}.npy"
        fold_test_path = ART / f"test_recipe_y_regression_fold{fold}.npy"
        if fold_oof_path.exists() and fold_test_path.exists():
            log(f"  checkpoint exists — loading {fold_oof_path.name}")
            fold_oof = np.load(fold_oof_path).astype(np.float32)
            fold_test = np.load(fold_test_path).astype(np.float32)
            assert fold_oof.shape[0] == len(va_idx), \
                f"fold {fold} OOF shape mismatch: {fold_oof.shape[0]} vs {len(va_idx)}"
            assert fold_test.shape[0] == len(test), \
                f"fold {fold} test shape mismatch: {fold_test.shape[0]} vs {len(test)}"
            oof[va_idx] = fold_oof
            test_pred += fold_test  # already divided by N_FOLDS at save time
            y_va = y[va_idx].astype(np.float32)
            rmse = float(np.sqrt(((oof[va_idx] - y_va) ** 2).mean()))
            fold_rmses.append(rmse)
            best_iters.append(-1)  # unknown from cached fold
            log(f"  fold {fold}  rmse={rmse:.5f}  (from cache)")
            continue
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)
        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va); X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")
        feat_cols = numeric_feats + te.te_col_names()
        log(f"  training XGBRegressor on {len(feat_cols)} features")
        t1 = time.time()
        y_tr = y[tr_idx].astype(np.float32)
        y_va = y[va_idx].astype(np.float32)
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr[feat_cols], y_tr,
                  eval_set=[(X_va[feat_cols], y_va)], verbose=False)
        bi = int(getattr(model, "best_iteration", model.n_estimators))
        best_iters.append(bi)
        fold_oof = model.predict(X_va[feat_cols]).astype(np.float32)
        fold_test = (model.predict(X_te[feat_cols]).astype(np.float32) / N_FOLDS)
        oof[va_idx] = fold_oof
        test_pred += fold_test
        rmse = float(np.sqrt(((oof[va_idx] - y_va) ** 2).mean()))
        fold_rmses.append(rmse)
        log(f"  fold {fold}  best_iter={bi}  rmse={rmse:.5f}  "
            f"wall={time.time()-t1:.1f}s")
        # Atomic per-fold save: write to .tmp then rename so a partial
        # write doesn't fool the resume check.
        for arr, path in ((fold_oof, fold_oof_path), (fold_test, fold_test_path)):
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, arr); tmp.rename(path)
        log(f"  fold {fold} checkpoint saved")
    log(f"=== OOF rmse mean={np.mean(fold_rmses):.5f} ± {np.std(fold_rmses):.5f}")
    return dict(oof=oof, test=test_pred, fold_rmses=fold_rmses,
                best_iters=best_iters, feat_cols=feat_cols)


def threshold_bal_acc(y, oof, t1, t2):
    pred = np.where(oof < t1, 0, np.where(oof < t2, 1, 2))
    return balanced_accuracy_score(y, pred), pred


def tune_thresholds(y, oof):
    """Coord-ascent on (t1, t2) thresholds to maximize macro-recall."""
    grid = np.linspace(0.0, 2.0, 41)
    best_t1, best_t2 = 0.5, 1.5
    best, _ = threshold_bal_acc(y, oof, best_t1, best_t2)
    for _ in range(15):
        improved = False
        for t1 in grid:
            if t1 >= best_t2:
                continue
            s, _ = threshold_bal_acc(y, oof, t1, best_t2)
            if s > best + 1e-7:
                best, best_t1 = s, t1; improved = True
        for t2 in grid:
            if t2 <= best_t1:
                continue
            s, _ = threshold_bal_acc(y, oof, best_t1, t2)
            if s > best + 1e-7:
                best, best_t2 = s, t2; improved = True
        if not improved:
            break
    return best_t1, best_t2, best


def to_3cls_gaussian(yhat, sigma):
    """Soft 3-class probs via Gaussian kernel `p_k ∝ exp(-(yhat - k)^2 / 2σ²)`."""
    diffs = yhat[:, None] - np.array([0, 1, 2])[None, :]
    logp = -0.5 * (diffs / sigma) ** 2
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def tune_sigma(y, oof, prior):
    """Pick σ that maximizes tuned-log-bias bal_acc on OOF."""
    best_sigma, best = 0.5, 0.0
    for sigma in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
        p = to_3cls_gaussian(oof, sigma)
        _, score = tune_log_bias(p, y, prior)
        if score > best:
            best, best_sigma = score, sigma
    return best_sigma, best


def main():
    train, test, info, test_ids = load_and_engineer()
    res = run_cv(train, test, info)
    y = train[TARGET].to_numpy()
    log("\n=== threshold tuning on OOF ===")
    t1, t2, bal_thr = tune_thresholds(y, res["oof"])
    log(f"  best thresholds = ({t1:.3f}, {t2:.3f})  bal_acc = {bal_thr:.5f}")
    log("\n=== Gaussian-kernel σ tuning on OOF ===")
    prior = np.bincount(y, minlength=3) / len(y)
    sigma, bal_g = tune_sigma(y, res["oof"], prior)
    log(f"  best σ = {sigma:.3f}  tuned bal_acc = {bal_g:.5f}")
    np.save(ART / "oof_recipe_y_regression.npy", res["oof"])
    np.save(ART / "test_recipe_y_regression.npy", res["test"])
    p_oof = to_3cls_gaussian(res["oof"], sigma)
    p_test = to_3cls_gaussian(res["test"], sigma)
    np.save(ART / "oof_recipe_y_regression_3cls.npy", p_oof)
    np.save(ART / "test_recipe_y_regression_3cls.npy", p_test)
    log(f"  saved continuous OOF + test + 3cls (σ={sigma}) probs")
    bias, bal_lb = tune_log_bias(p_oof, y, prior)
    log(f"\n  3cls @ tuned log-bias: bal_acc={bal_lb:.5f}  bias={bias.round(4).tolist()}")
    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_rmses=res["fold_rmses"], best_iters=res["best_iters"],
        oof_rmse_mean=float(np.mean(res["fold_rmses"])),
        threshold_t1=float(t1), threshold_t2=float(t2),
        bal_acc_thresholds=float(bal_thr),
        gaussian_sigma=float(sigma),
        bal_acc_gaussian_logbias=float(bal_lb),
        bal_acc_gaussian_argmax=float(bal_g),
        log_bias=bias.tolist(),
        n_features=len(res["feat_cols"]),
    )
    with open(ART / "recipe_y_regression_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote scripts/artifacts/recipe_y_regression_results.json")


if __name__ == "__main__":
    main()
