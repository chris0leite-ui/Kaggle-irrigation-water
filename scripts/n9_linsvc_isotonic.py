"""#9 Calibrated LinearSVC + isotonic on dist features.

Linear methods completely untested on this problem. LinearSVC scales to
504k rows; one-vs-rest with sklearn CalibratedClassifierCV(method='isotonic',
cv=3) for proper probabilistic outputs.

Use 43-feature dist set (already in add_distance_features).
5-fold StratifiedKFold(seed=42) aligned with all other OOFs.

Goal: standalone OOF tuned >= 0.97 → blend-gate vs LB-best 4-stack.
If passes: becomes new OTHER for override mechanism.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, add_distance_features  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    log("Building dist features")
    train_dist = add_distance_features(train.drop(columns=[TARGET]))
    test_dist = add_distance_features(test)

    feat_cols = [c for c in train_dist.columns
                 if pd.api.types.is_numeric_dtype(train_dist[c])
                 and train_dist[c].dtype != bool
                 and not (train_dist[c].dtype.name == 'category')]
    # Drop binary and bool/category cols safely
    feat_cols = [c for c in feat_cols if train_dist[c].dtype.kind in "fiub"]
    log(f"  {len(feat_cols)} numeric features")
    Xtr = train_dist[feat_cols].to_numpy().astype(np.float32)
    Xte = test_dist[feat_cols].to_numpy().astype(np.float32)
    log(f"  Xtr shape: {Xtr.shape}, Xte shape: {Xte.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_pred = np.zeros((len(test_ids), 3), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(Xtr, y)):
        t0 = time.time()
        log(f"Fold {fold+1}/5: train {len(tr)}, val {len(va)}")
        scaler = StandardScaler().fit(Xtr[tr])
        Xtr_s = scaler.transform(Xtr[tr]).astype(np.float32)
        Xva_s = scaler.transform(Xtr[va]).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)

        # CalibratedClassifierCV with cv=3 + LinearSVC base
        # multi_class='ovr' is implicit for binary, here use OneVsRest via decision_function_shape
        # LinearSVC with sklearn 1.x supports multi_class via 'ovr' (default) or 'crammer_singer'
        base = LinearSVC(C=0.5, class_weight=None, max_iter=2000,
                         dual="auto", random_state=SEED + fold)
        clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3, n_jobs=4)
        clf.fit(Xtr_s, y[tr])
        log(f"  fit done in {time.time()-t0:.1f}s")

        oof[va] = clf.predict_proba(Xva_s).astype(np.float32)
        test_pred += clf.predict_proba(Xte_s).astype(np.float32) / 5.0

        # Per-fold sanity
        argm = oof[va].argmax(1)
        bal = balanced_accuracy_score(y[va], argm)
        log(f"  fold {fold+1} val bal_acc (argmax): {bal:.5f}  time={time.time()-t0:.1f}s")

    # Save raw OOF + test
    np.save(ART / "oof_n9_linsvc_isotonic.npy", oof)
    np.save(ART / "test_n9_linsvc_isotonic.npy", test_pred)

    # Tune log-bias
    bias, tuned = tune_log_bias(oof, y, prior)
    pred_tuned = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1)
    bal_argmax = balanced_accuracy_score(y, oof.argmax(1))
    pcr_tuned = per_class_recall(y, pred_tuned)
    log(f"\n=== n9 LinearSVC standalone ===")
    log(f"  OOF argmax: {bal_argmax:.5f}")
    log(f"  OOF tuned:  {tuned:.5f}  bias={bias.round(3).tolist()}")
    log(f"  PCR=[L={pcr_tuned[0]:.4f} M={pcr_tuned[1]:.4f} H={pcr_tuned[2]:.4f}]")

    # Compare to anchor v1
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_oof = v1_oof / np.clip(v1_oof.sum(1, keepdims=True), 1e-9, None)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    v1_arg = (np.log(np.clip(v1_oof, 1e-9, 1.0)) + v1_bias).argmax(1)

    # Jaccard vs v1
    diff = (pred_tuned != v1_arg).sum()
    log(f"  Jaccard error vs v1: differs on {diff} OOF rows ({diff/len(y)*100:.2f}%)")

    # Build submission
    test_log = np.log(np.clip(test_pred, 1e-9, 1.0)) + bias
    test_arg = test_log.argmax(1)
    path = SUB / "submission_n9_linsvc_isotonic.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_arg]}).to_csv(path, index=False)
    log(f"  Saved: {path}")

    # If standalone passes 0.97 threshold, attempt as new OTHER for override
    test_v1_pred = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")[TARGET].map(CLS2IDX).to_numpy()
    test_diff = (test_arg != test_v1_pred).sum()
    log(f"  Test diff vs v1 CSV: {test_diff}")

    summary = {
        "OOF_argmax": float(bal_argmax),
        "OOF_tuned": float(tuned),
        "bias": bias.tolist(),
        "PCR_L": float(pcr_tuned[0]),
        "PCR_M": float(pcr_tuned[1]),
        "PCR_H": float(pcr_tuned[2]),
        "diff_oof_vs_v1": int(diff),
        "diff_test_vs_v1": int(test_diff),
        "n_features": len(feat_cols),
        "submission": str(path),
    }
    with open(ART / "n9_linsvc_isotonic_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved summary")


if __name__ == "__main__":
    main()
