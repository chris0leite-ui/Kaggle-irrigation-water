"""Per-score expert LGBM models (brainstorm idea #8).

Strategy: partition rows by the DGP score s ∈ {0..10}. Train one
binary/3-class LGBM per score value on rows with that score,
predicting y_obs from *non-rule* features (everything except the
6 rule cols and score itself — those are constant within a score
bin and carry no intra-bin signal).

For each row at prediction time we compute score, look up the
expert trained on that score, and read its probability vector
out. This forces each expert to specialise on the exact noise
pattern of its score bin:
  - scores 0, 5, 6, 10: rule is noise-free, predict rule label.
  - scores 1, 2, 3: rule=Low, some rows are actually Medium.
    Binary Low-vs-Medium expert.
  - score 4: rule=Medium, some rows are Low or High. 3-class
    expert.
  - scores 7, 8, 9: rule=High, some rows are actually Medium.
    Binary Medium-vs-High expert.

OOF: 5-fold stratified on `y` (same split as baseline). For each
fold, we train every required expert on the training portion and
predict on the validation portion, routing each val row by its
score. We then tune a global per-class log-bias on the combined
OOF probs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

RULE_COLS = (
    "Soil_Moisture",
    "Rainfall_mm",
    "Temperature_C",
    "Wind_Speed_kmh",
    "Mulching_Used",
    "Crop_Growth_Stage",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dgp_score_series(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE_STAGES), 2, 0)
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def rule_label(score: np.ndarray) -> np.ndarray:
    return np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int32)


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def train_expert(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    allowed_classes: list[int],
    cat_cols: list[str],
    n_classes: int,
) -> np.ndarray:
    """Train a binary or 3-class LGBM and return prob predictions on X_va.

    allowed_classes: list of class indices that appear in this bin.
    Returned probs are dense over all 3 classes (zeros for classes
    not seen in this bin).
    """
    # remap training labels to a dense range over allowed_classes
    remap = {c: i for i, c in enumerate(allowed_classes)}
    y_local = np.array([remap[int(y)] for y in y_tr], dtype=np.int32)

    if n_classes == 2:
        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            num_leaves=127,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=100,
            verbose=-1,
            seed=SEED,
        )
        dtr = lgb.Dataset(X_tr, label=y_local, categorical_feature=cat_cols)
        model = lgb.train(params, dtr, num_boost_round=400)
        p1 = model.predict(X_va)
        local_probs = np.stack([1.0 - p1, p1], axis=1)
    else:
        params = dict(
            objective="multiclass",
            num_class=n_classes,
            metric="multi_logloss",
            learning_rate=0.05,
            num_leaves=127,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=100,
            verbose=-1,
            seed=SEED,
        )
        dtr = lgb.Dataset(X_tr, label=y_local, categorical_feature=cat_cols)
        model = lgb.train(params, dtr, num_boost_round=400)
        local_probs = model.predict(X_va)

    # expand to 3 classes
    full = np.zeros((len(X_va), len(CLASSES)), dtype=np.float64)
    for local_idx, global_cls in enumerate(allowed_classes):
        full[:, global_cls] = local_probs[:, local_idx]
    return full


def one_hot_from_rule(score: np.ndarray) -> np.ndarray:
    """Convert rule score directly into a 3-class prob vector.

    Used for score bins where the rule is noise-free so training
    an expert is redundant. We still emit a near-hard vector so
    log-bias tuning can nudge it.
    """
    rule = rule_label(score)
    probs = np.full((len(score), len(CLASSES)), 1e-3, dtype=np.float64)
    probs[np.arange(len(score)), rule] = 1.0 - 2e-3
    return probs


# ------------------------------------------------------------------ data ----
def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    score_tr = dgp_score_series(tr)
    score_te = dgp_score_series(te)
    tr["_score"] = score_tr
    te["_score"] = score_te

    # features: everything EXCEPT rule cols (Soil_Moisture, Rainfall_mm,
    # Temperature_C, Wind_Speed_kmh, Mulching_Used, Crop_Growth_Stage).
    # Also drop id, target, _score.
    feat_cols = [c for c in tr.columns
                 if c not in RULE_COLS + (ID, TARGET, "_score")]
    # distance-to-threshold features (since rule cols are dropped we
    # still want the "how close to boundary" info from them).
    def add_dist(df):
        df["_sm_dist"] = df["Soil_Moisture"].astype(float) - 25.0
        df["_rf_dist"] = df["Rainfall_mm"].astype(float) - 300.0
        df["_tc_dist"] = df["Temperature_C"].astype(float) - 30.0
        df["_ws_dist"] = df["Wind_Speed_kmh"].astype(float) - 10.0
        df["_sm_abs"] = df["_sm_dist"].abs()
        df["_rf_abs"] = df["_rf_dist"].abs()
        df["_tc_abs"] = df["_tc_dist"].abs()
        df["_ws_abs"] = df["_ws_dist"].abs()
        return df

    tr = add_dist(tr)
    te = add_dist(te)
    dist_cols = ["_sm_dist", "_rf_dist", "_tc_dist", "_ws_dist",
                 "_sm_abs", "_rf_abs", "_tc_abs", "_ws_abs"]
    feat_cols = feat_cols + dist_cols

    # categorical encoding
    cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(tr[c])]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    log(f"features ({len(feat_cols)}): {feat_cols}")
    log(f"cat cols: {cat_cols}")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    # describe score bins
    for s in range(11):
        n = int((score_tr == s).sum())
        if n == 0:
            continue
        classes_in_bin = sorted(set(y[score_tr == s].tolist()))
        log(f"  score={s:2d}  n={n:7d}  classes={[CLASSES[c] for c in classes_in_bin]}")

    # assign each score bin to a model type
    bin_config: dict[int, dict] = {}
    for s in range(11):
        mask = score_tr == s
        if mask.sum() == 0:
            continue
        classes_in_bin = sorted(set(y[mask].tolist()))
        if len(classes_in_bin) == 1:
            bin_config[s] = {"type": "hard", "class": classes_in_bin[0]}
        elif len(classes_in_bin) == 2:
            bin_config[s] = {"type": "binary", "classes": classes_in_bin}
        else:
            bin_config[s] = {"type": "multiclass", "classes": classes_in_bin}

    log("bin_config:")
    for s, cfg in bin_config.items():
        log(f"  score={s}: {cfg}")

    # --- 5-fold CV ---
    log("running 5-fold stratified CV with per-score experts")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(tr, y)):
        t0 = time.time()
        score_fold_tr = score_tr[tr_idx]
        score_fold_va = score_tr[va_idx]
        X_fold_tr = tr[feat_cols].iloc[tr_idx]
        X_fold_va = tr[feat_cols].iloc[va_idx]
        y_fold_tr = y[tr_idx]

        for s, cfg in bin_config.items():
            m_tr = score_fold_tr == s
            m_va = score_fold_va == s
            if m_va.sum() == 0:
                continue
            if cfg["type"] == "hard":
                oof_idx = va_idx[m_va]
                oof[oof_idx] = 0.0
                oof[oof_idx, cfg["class"]] = 1.0 - 2e-3
                oof[oof_idx, :] = np.where(oof[oof_idx] > 0, oof[oof_idx], 1e-3)
                continue
            if m_tr.sum() < 30:
                # too few training rows in this fold; fall back to rule-label
                oof_idx = va_idx[m_va]
                rule_cls = rule_label(np.array([s]))[0]
                oof[oof_idx, :] = 1e-3
                oof[oof_idx, rule_cls] = 1.0 - 2e-3
                continue
            X_tr_s = X_fold_tr[m_tr]
            y_tr_s = y_fold_tr[m_tr]
            X_va_s = X_fold_va[m_va]
            full = train_expert(
                X_tr_s, y_tr_s, X_va_s,
                allowed_classes=cfg["classes"],
                cat_cols=cat_cols,
                n_classes=len(cfg["classes"]),
            )
            oof[va_idx[m_va]] = full

        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  argmax_bal_acc={fold_bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    log("tuning log-bias on OOF")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}")
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # --- train full experts on all train, predict on test ---
    log("training full-train experts for test predictions")
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    X_all = tr[feat_cols]
    X_te = te[feat_cols]
    for s, cfg in bin_config.items():
        m_all = score_tr == s
        m_te = score_te == s
        if m_te.sum() == 0:
            continue
        if cfg["type"] == "hard":
            test_pred[m_te, :] = 1e-3
            test_pred[m_te, cfg["class"]] = 1.0 - 2e-3
            continue
        if m_all.sum() < 30:
            rule_cls = rule_label(np.array([s]))[0]
            test_pred[m_te, :] = 1e-3
            test_pred[m_te, rule_cls] = 1.0 - 2e-3
            continue
        full = train_expert(
            X_all[m_all], y[m_all], X_te[m_te],
            allowed_classes=cfg["classes"],
            cat_cols=cat_cols,
            n_classes=len(cfg["classes"]),
        )
        test_pred[m_te] = full

    print("\n=== per-score expert LGBM (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  prior-reweight       : {reweight_bal:.5f}")
    print(f"  tuned log-bias       : {tuned_bal:.5f}")

    np.save(ART_DIR / "oof_score_experts.npy", oof)
    np.save(ART_DIR / "test_score_experts.npy", test_pred)
    with open(ART_DIR / "score_experts_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "bin_config": {int(k): v for k, v in bin_config.items()},
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_score_experts_tuned.csv", index=False
    )
    log(f"artifacts written to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
