"""B: cuML meta-stacker on Kaggle GPU.

Loads pre-built meta-feature matrix (X_tr, X_te, y, fold_idx) from the
irrigation-cuml-meta-input dataset, trains 3 meta-stackers (cuML LR with
L2, cuML RF, cuML KNN) on the same 5-fold StratifiedKFold(seed=42) split
that produced LB-best 0.98094.

Mechanism: tests whether GPU-accelerated alternative L2 architectures find
a meta with a passing 4-gate-filter profile that the depth-4 XGB / sklearn
LR / sklearn MLP couldn't (all NULL or LB-regressed). Different numerical
implementations + different inductive biases:
  - cuML LR with L2: GPU SVD/QR solver, exact L-BFGS (vs sklearn's similar)
  - cuML RandomForest: bagging on a dense bank — UNTESTED on this bank
  - cuML KNeighborsClassifier: lazy learner, no parametric fit

Outputs:
  oof_cuml_lr.npy + test_cuml_lr.npy
  oof_cuml_rf.npy + test_cuml_rf.npy
  oof_cuml_knn.npy + test_cuml_knn.npy
  cuml_meta_results.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

print(f"[boot] cwd={os.getcwd()}", flush=True)


def _gpu_arch():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception as e:
        print(f"[boot] nvidia-smi err: {e}", flush=True)
        return []


print(f"[boot] gpu compute_cap = {_gpu_arch()}", flush=True)

import numpy as np  # noqa: E402

# Locate dataset npz.
INPUT_DIR = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True, parents=True)


def _find_npz():
    for p in INPUT_DIR.rglob("cuml_meta_input.npz"):
        return p
    raise FileNotFoundError("cuml_meta_input.npz not found in /kaggle/input")


npz_path = _find_npz()
print(f"[boot] loading {npz_path}", flush=True)
data = np.load(npz_path, allow_pickle=True)
X_tr = data["X_tr"].astype(np.float32)
X_te = data["X_te"].astype(np.float32)
y = data["y"].astype(np.int32)
fold_idx = data["fold_idx"].astype(np.int32)
feature_names = list(data["feature_names"])
print(f"[boot] X_tr={X_tr.shape}  X_te={X_te.shape}  features={len(feature_names)}", flush=True)
print(f"[boot] fold counts: {np.bincount(fold_idx)}", flush=True)

# SMOKE: 1 fold, k_neighbors=20, n_estimators=50
IS_SMOKE = True  # SMOKE override for this push
N_FOLDS = 5 if not IS_SMOKE else 1
print(f"[boot] IS_SMOKE={IS_SMOKE}  N_FOLDS={N_FOLDS}", flush=True)

# cuML imports — Kaggle GPU images come with cuML pre-installed.
print("[boot] importing cuML", flush=True)
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    print(f"[boot] cuML version: {cuml.__version__}", flush=True)
except Exception as e:
    print(f"[boot] cuML import FAILED: {e}", flush=True)
    raise

# Standardize features (cuML LR + KNN benefit; RF is invariant).
print("[boot] standardising X_tr / X_te", flush=True)
mu = X_tr.mean(0).astype(np.float32)
sd = X_tr.std(0).astype(np.float32)
sd = np.where(sd < 1e-6, 1.0, sd)
X_tr_s = ((X_tr - mu) / sd).astype(np.float32)
X_te_s = ((X_te - mu) / sd).astype(np.float32)

# Per-fold OOF + test storage.
oof_lr = np.zeros((len(y), 3), dtype=np.float32)
oof_rf = np.zeros((len(y), 3), dtype=np.float32)
oof_knn = np.zeros((len(y), 3), dtype=np.float32)
test_lr = np.zeros((len(X_te), 3), dtype=np.float32)
test_rf = np.zeros((len(X_te), 3), dtype=np.float32)
test_knn = np.zeros((len(X_te), 3), dtype=np.float32)

results = {"per_fold": [], "models": ["lr", "rf", "knn"]}

t_global = time.time()
for fold in range(1, N_FOLDS + 1):
    mask_tr = fold_idx != fold
    mask_va = fold_idx == fold
    Xa, Xb = X_tr_s[mask_tr], X_tr_s[mask_va]
    ya, yb = y[mask_tr], y[mask_va]
    print(f"\n=== fold {fold}/{N_FOLDS}  tr={len(Xa):,}  va={len(Xb):,} ===", flush=True)

    # --- cuML LR (L2, multi-class) ---
    t0 = time.time()
    lr = cuLR(C=0.1, penalty="l2", solver="qn", max_iter=400)
    lr.fit(Xa, ya)
    p_va = np.asarray(lr.predict_proba(Xb)).astype(np.float32)
    p_te = np.asarray(lr.predict_proba(X_te_s)).astype(np.float32)
    oof_lr[mask_va] = p_va
    test_lr += p_te / N_FOLDS
    print(f"  LR done in {time.time()-t0:.1f}s  argmax_acc={(p_va.argmax(1)==yb).mean():.4f}", flush=True)

    # --- cuML RF (n_est=300, max_depth=12) ---
    t0 = time.time()
    rf_n_est = 50 if IS_SMOKE else 300
    rf_max_depth = 8 if IS_SMOKE else 12
    rf = cuRF(n_estimators=rf_n_est, max_depth=rf_max_depth,
              n_streams=1, random_state=42)
    rf.fit(Xa, ya)
    p_va = np.asarray(rf.predict_proba(Xb)).astype(np.float32)
    p_te = np.asarray(rf.predict_proba(X_te_s)).astype(np.float32)
    oof_rf[mask_va] = p_va
    test_rf += p_te / N_FOLDS
    print(f"  RF done in {time.time()-t0:.1f}s  argmax_acc={(p_va.argmax(1)==yb).mean():.4f}", flush=True)

    # --- cuML KNN (k=50) ---
    t0 = time.time()
    k = 20 if IS_SMOKE else 50
    knn = cuKNN(n_neighbors=k)
    knn.fit(Xa, ya)
    p_va = np.asarray(knn.predict_proba(Xb)).astype(np.float32)
    p_te = np.asarray(knn.predict_proba(X_te_s)).astype(np.float32)
    oof_knn[mask_va] = p_va
    test_knn += p_te / N_FOLDS
    print(f"  KNN k={k} done in {time.time()-t0:.1f}s  argmax_acc={(p_va.argmax(1)==yb).mean():.4f}", flush=True)

    results["per_fold"].append({
        "fold": fold,
        "n_train": int(mask_tr.sum()), "n_val": int(mask_va.sum()),
    })
    # Save per-fold checkpoints (rehydrate-resilient even on Kaggle).
    np.save(WORK_DIR / f"oof_cuml_lr_fold{fold}.npy", p_va)
    np.save(WORK_DIR / f"oof_cuml_rf_fold{fold}.npy", oof_rf[mask_va])
    np.save(WORK_DIR / f"oof_cuml_knn_fold{fold}.npy", oof_knn[mask_va])

print(f"\n=== total wall {time.time()-t_global:.1f}s ===", flush=True)

# Standalone OOF accuracy at argmax (not yet log-bias tuned).
for nm, oof in [("lr", oof_lr), ("rf", oof_rf), ("knn", oof_knn)]:
    if not IS_SMOKE or fold == 1:
        # Compute on rows with predictions populated (smoke: only fold-1 rows)
        mask = oof.sum(1) > 1e-3
        acc = (oof[mask].argmax(1) == y[mask]).mean()
        print(f"  {nm}: argmax_acc on populated rows = {acc:.5f}", flush=True)

# Save outputs.
for nm, oof, te in [
    ("lr", oof_lr, test_lr), ("rf", oof_rf, test_rf), ("knn", oof_knn, test_knn),
]:
    np.save(WORK_DIR / f"oof_cuml_{nm}.npy", oof)
    np.save(WORK_DIR / f"test_cuml_{nm}.npy", te)
    print(f"  wrote oof_cuml_{nm}.npy + test_cuml_{nm}.npy", flush=True)

with open(WORK_DIR / "cuml_meta_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[done]", flush=True)
