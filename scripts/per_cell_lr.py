"""Per-cell logistic regression diagnostic.

The 2026-04-21 DGP residual EDA showed non-rule continuous features
(Humidity, Previous_Irrigation_mm, EC, Field_Area) have significant
Cohen's d between flipped and non-flipped rows. Trees at 127 leaves
already split on rule features to form cells; within a cell they run
out of leaves to separate on continuous non-rule features (8 available,
all weak individually). A cell-local linear model has no such budget
constraint — it sees the non-rule axes directly.

This script:
  - Partitions 630k synthetic train into 128 rule cells (same scheme
    as empirical_bayes_cell.py: stage * 32 + dry*16 + norain*8 +
    hot*4 + windy*2 + nomulch).
  - For each of 5 folds x 128 cells, fits a multinomial LR on 7
    non-rule continuous features (Humidity, Previous_Irrigation_mm,
    EC, Field_Area_hectare, Soil_pH, Organic_Carbon, Sunlight_Hours).
  - Fallback for cells with <100 rows or <2 unique classes: Laplace-
    smoothed empirical-Bayes over that cell's class distribution.
  - Saves OOF probs + test probs + JSON summary.

Baselines to beat:
  EB-cell (6 rule features only)   0.96339  -- ceiling for rule-only
  LGBM-dist (43 features)          0.97266  -- tree with non-rule
  Hybrid (current best)            0.97352

Diagnostic value:
  - If standalone > 0.963, non-rule continuous signal is real
    at cell level (above pure rule).
  - Blend with hybrid is the real test — orthogonality, not
    standalone score.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

NON_RULE_FEATS = [
    "Humidity",
    "Previous_Irrigation_mm",
    "Electrical_Conductivity",
    "Field_Area_hectare",
    "Soil_pH",
    "Organic_Carbon",
    "Sunlight_Hours",
]

MIN_CELL_SIZE = 100   # cells smaller than this fall back to EB
MIN_CLASSES = 2       # cells with <2 classes in train fall back to EB
LR_C = 1.0            # L2 regularization strength
LR_MAX_ITER = 1000

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell_id(df: pd.DataFrame) -> np.ndarray:
    """Pack (stage, dry, norain, hot, windy, nomulch) into int in [0, 128)."""
    dry = (df["Soil_Moisture"].astype(float).values < 25).astype(int)
    norain = (df["Rainfall_mm"].astype(float).values < 300).astype(int)
    hot = (df["Temperature_C"].astype(float).values > 30).astype(int)
    windy = (df["Wind_Speed_kmh"].astype(float).values > 10).astype(int)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(int)
    stage_cats = ["Flowering", "Harvest", "Sowing", "Vegetative"]
    stage_vals = pd.Categorical(
        df["Crop_Growth_Stage"].astype(str).values,
        categories=stage_cats,
        ordered=False,
    ).codes.astype(int)
    if (stage_vals < 0).any():
        raise ValueError(
            f"unseen stage: "
            f"{set(df['Crop_Growth_Stage'].astype(str)) - set(stage_cats)}"
        )
    return (
        stage_vals * 32
        + dry * 16
        + norain * 8
        + hot * 4
        + windy * 2
        + nomulch
    ).astype(np.int32)


def eb_probs(y_train: np.ndarray) -> np.ndarray:
    """Laplace-smoothed class distribution for a cell (shape 3)."""
    alpha = 1.0 / len(CLASSES)
    counts = np.full(len(CLASSES), alpha, dtype=np.float64)
    for yy in y_train:
        counts[yy] += 1.0
    return counts / counts.sum()


def fit_cell(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
) -> np.ndarray:
    """Return (n_va, 3) probs. Falls back to EB if unfit-worthy."""
    n_classes = len(np.unique(y_tr))
    if len(y_tr) < MIN_CELL_SIZE or n_classes < MIN_CLASSES:
        prob = eb_probs(y_tr)
        return np.tile(prob[None, :], (len(X_va), 1))

    # standardize per-cell
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    clf = LogisticRegression(
        solver="lbfgs",
        C=LR_C,
        max_iter=LR_MAX_ITER,
        # no class_weight: let LR learn true per-cell posteriors.
        # macro-recall optimization lives in the pipeline log-bias
        # stage, not per-cell.
    )
    clf.fit(X_tr_s, y_tr)
    # LR only knows the classes it saw; pad missing ones with ~0.
    proba_raw = clf.predict_proba(X_va_s)
    proba = np.full((len(X_va), len(CLASSES)), 1e-6, dtype=np.float64)
    for i, c in enumerate(clf.classes_):
        proba[:, c] = proba_raw[:, i]
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def fit_and_predict_fold(
    cell_tr: np.ndarray, cell_va: np.ndarray,
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray,
) -> np.ndarray:
    """Per-cell LR over a single fold's (train, val) split."""
    proba_va = np.zeros((len(X_va), len(CLASSES)), dtype=np.float64)
    uniq_cells = np.unique(np.concatenate([cell_tr, cell_va]))
    n_fit = 0
    n_fallback = 0
    for c in uniq_cells:
        mask_tr = cell_tr == c
        mask_va = cell_va == c
        if not mask_va.any():
            continue
        if not mask_tr.any():
            # unseen cell at val time: uniform prior
            proba_va[mask_va] = 1.0 / len(CLASSES)
            n_fallback += 1
            continue
        X_tr_c = X_tr[mask_tr]
        y_tr_c = y_tr[mask_tr]
        X_va_c = X_va[mask_va]
        proba_va[mask_va] = fit_cell(X_tr_c, y_tr_c, X_va_c)
        if len(y_tr_c) < MIN_CELL_SIZE or len(np.unique(y_tr_c)) < MIN_CLASSES:
            n_fallback += 1
        else:
            n_fit += 1
    return proba_va, n_fit, n_fallback


def tune_log_bias(
    oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
) -> tuple[np.ndarray, float]:
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
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    cell_tr_all = cell_id(tr)
    cell_te_all = cell_id(te)
    log(f"cells used in train: {len(np.unique(cell_tr_all))}/128 "
        f"(test: {len(np.unique(cell_te_all))}/128)")

    # Per-cell size + class-count diagnostic
    ctr_counts = pd.Series(cell_tr_all).value_counts().sort_index()
    small = (ctr_counts < MIN_CELL_SIZE).sum()
    log(f"cells with <{MIN_CELL_SIZE} train rows: {small}")
    # Cell class-entropy diagnostic
    multi_class_cells = 0
    for c in ctr_counts.index:
        if len(np.unique(y[cell_tr_all == c])) >= MIN_CLASSES:
            multi_class_cells += 1
    log(f"cells with ≥{MIN_CLASSES} classes present: "
        f"{multi_class_cells}/{len(ctr_counts)}")

    X = tr[NON_RULE_FEATS].astype(float).values
    X_te = te[NON_RULE_FEATS].astype(float).values

    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_bals = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(cell_tr_all, y)):
        t0 = time.time()
        proba_va, n_fit, n_fb = fit_and_predict_fold(
            cell_tr_all[tr_idx], cell_tr_all[va_idx],
            X[tr_idx], y[tr_idx], X[va_idx],
        )
        oof[va_idx] = proba_va
        fold_bal = balanced_accuracy_score(y[va_idx], proba_va.argmax(axis=1))
        fold_bals.append(fold_bal)
        log(f"  fold {fold+1}/{N_FOLDS}  argmax_bal={fold_bal:.5f}  "
            f"fit={n_fit} fallback={n_fb}  {time.time()-t0:.1f}s")

    # full-train fit for test predictions
    log("fitting all cells on full train for test predictions")
    t0 = time.time()
    uniq_all = np.unique(np.concatenate([cell_tr_all, cell_te_all]))
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    for c in uniq_all:
        m_tr = cell_tr_all == c
        m_te = cell_te_all == c
        if not m_te.any():
            continue
        if not m_tr.any():
            test_pred[m_te] = 1.0 / len(CLASSES)
            continue
        test_pred[m_te] = fit_cell(X[m_tr], y[m_tr], X_te[m_te])
    log(f"test fit done in {time.time()-t0:.1f}s")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"bias = {dict(zip(CLASSES, bias.round(4)))}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== per-cell LR summary (OOF bal_acc) ===")
    print(f"  argmax                : {argmax_bal:.5f}")
    print(f"  prior-reweight        : {reweight_bal:.5f}")
    print(f"  tuned log-bias        : {tuned_bal:.5f}")
    print(f"  fold std              : {np.std(fold_bals):.5f}")

    np.save(ART / "oof_per_cell_lr.npy", oof)
    np.save(ART / "test_per_cell_lr.npy", test_pred)
    with open(ART / "per_cell_lr_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "features": NON_RULE_FEATS,
            "min_cell_size": MIN_CELL_SIZE,
            "lr_C": LR_C,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
            "fold_bals": [float(b) for b in fold_bals],
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_per_cell_lr_tuned.csv", index=False
    )
    log(f"artifacts -> {ART}/; submission -> "
        f"{SUB/'submission_per_cell_lr_tuned.csv'}")


if __name__ == "__main__":
    main()
