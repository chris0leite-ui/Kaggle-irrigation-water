"""N2 — kNN diversity-leg for the meta-stacker bank.

KNN(k=50) on the 43-feature dist set, 5-fold StratifiedKFold(seed=42).
Distance-based model — fundamentally different geometry from every
existing tree/NN bank component. wguesdon's kernel includes `knn_ote`
as a diversity weak-learner.

Performance trick: fit on a stratified 80k subsample of train (full
504k k-NN is impractical at k=50). Predict on the full 130k val
fold and the full 270k test set. The subsample preserves per-class
density structure; k=50 gives stable per-class probability estimates
with the small sample.

Output: oof_n2_knn.npy + test_n2_knn.npy + JSON.
SMOKE=1 → 1 fold, smaller subsample.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)

# Same dist columns as N2 ExtraTrees for apples-to-apples bank addition.
# 11 raw numerics + 24 derived from add_distance_features() = 35.
DIST_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
    "dry", "norain", "hot", "windy", "nomulch", "kc_active",
    "dgp_score", "rule_pred",
    "sm_dist", "rf_dist", "tc_dist", "ws_dist",
    "sm_abs", "rf_abs", "tc_abs", "ws_abs",
    "min_boundary_dist", "min_axis_abs",
    "score_dist_low_mid", "score_dist_mid_high",
    "sm_x_kc", "sm_x_rf", "rf_x_kc", "tc_x_ws",
]

K_NN = 50
SUBSAMPLE_SIZE = 80_000  # stratified per fold


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def stratified_subsample(X, y, n, rng):
    """Per-class proportional subsample of size n."""
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    take = (counts * n / counts.sum()).astype(int)
    take[-1] = n - take[:-1].sum()  # exact total
    idx = []
    for c, k in zip(classes, take):
        c_idx = np.where(y == c)[0]
        idx.append(rng.choice(c_idx, size=k, replace=False))
    return np.concatenate(idx)


def main():
    t0 = time.time()
    log(f"N2 kNN(k={K_NN}) on 43-dist features — N_FOLDS={N_FOLDS}, "
        f"subsample={SUBSAMPLE_SIZE}, SMOKE={SMOKE}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    test_ids = test["id"].values

    if SMOKE:
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        sub_size = 8000
    else:
        sub_size = SUBSAMPLE_SIZE

    log("computing dist features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    feat_cols = [c for c in DIST_COLS if c in tr_d.columns]
    log(f"  feat_cols={len(feat_cols)}")

    X_tr_full = tr_d[feat_cols].to_numpy(dtype=np.float32)
    X_te = te_d[feat_cols].to_numpy(dtype=np.float32)
    y = train[TARGET].to_numpy().astype(np.int32)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    rng = np.random.default_rng(SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr_full, y), 1):
        t_fold = time.time()
        log(f"=== fold {fold}/{N_FOLDS} ===")
        # Stratified subsample of train fold
        sub_idx_local = stratified_subsample(
            X_tr_full[tr_idx], y[tr_idx],
            min(sub_size, len(tr_idx)), rng,
        )
        X_sub = X_tr_full[tr_idx][sub_idx_local]
        y_sub = y[tr_idx][sub_idx_local]
        log(f"  fit subsample = {len(X_sub):,}, classes = {np.bincount(y_sub).tolist()}")

        # Standardize (kNN is distance-based — must scale)
        scaler = StandardScaler().fit(X_sub)
        X_sub_s = scaler.transform(X_sub)
        X_va_s = scaler.transform(X_tr_full[va_idx])
        X_te_s = scaler.transform(X_te)

        knn = KNeighborsClassifier(n_neighbors=K_NN, n_jobs=-1, weights="distance")
        knn.fit(X_sub_s, y_sub)
        log(f"  fit done at {time.time()-t_fold:.1f}s; predicting val")
        oof[va_idx] = knn.predict_proba(X_va_s).astype(np.float32)
        log(f"  val done at {time.time()-t_fold:.1f}s; predicting test")
        test_pred += knn.predict_proba(X_te_s).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  wall={time.time()-t_fold:.1f}s")
        np.save(ART / f"oof_n2_knn_fold{fold}.npy", oof)
        np.save(ART / f"test_n2_knn_fold{fold}.npy", test_pred)

    overall = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"OOF argmax = {overall:.5f}  tuned = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / "oof_n2_knn.npy", oof)
    np.save(ART / "test_n2_knn.npy", test_pred)

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_features=len(feat_cols),
        k_nn=K_NN, subsample_size=sub_size,
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax_bal_acc=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "n2_knn_results.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote oof_n2_knn.npy + test + JSON  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
