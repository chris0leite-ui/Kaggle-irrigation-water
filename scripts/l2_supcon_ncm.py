"""L2: SupCon embedding + Mahalanobis NCM with macro-recall-Bayes-optimal
decision rule.

Mechanism-novel attack on the LB 0.98134 ceiling. Reuses p3's SupCon
embedding (proven on this problem) but swaps the decision layer:
  - p3 used label-propagation in the embedding space (Jaccard 0.65,
    blend null because magnitude trap)
  - L2 uses a closed-form Bayes-optimal NCM under uniform prior — no
    post-hoc bias retune (the load-bearing structural property).

Decision rule:
  argmax_k log P(z | y=k) - log(1/3)
  ≡ argmax_k log p(z | y=k)  (uniform prior cancels)

Because we use likelihoods rather than softmax-CE outputs, there is no
bias-retune leak channel.

Fold protocol: 5-fold StratifiedKFold(seed=42) for v1-bank alignment.
For each fold:
  1. Train SupCon embedding on train_tr only
  2. Embed train_va + test
  3. Fit MahalanobisNCM on (embed(train_tr), y[tr_idx])
  4. predict_proba_macro_recall on embed(train_va) → OOF for fold
For test predictions: train embedding on full train, fit NCM on full,
average over n_seeds embedding seeds.

Env:
  EPOCHS=30
  BATCH_SIZE=4096
  EMBED_DIM=32
  DEVICE=cpu|cuda (default auto)
  SMOKE=1 (20k subsample + 1 fold + 3 epochs)
  N_FOLDS=5 (override for SMOKE)
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from l2_ncm_helpers import MahalanobisNCM  # noqa: E402
# Reuse p3's SupCon Embedder unchanged.
from p3_embed_propagate import Embedder, _select_device, load_and_engineer, _build_feat_matrix  # noqa: E402

SEED = 42
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}

SMOKE = os.environ.get("SMOKE") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "3" if SMOKE else "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
EMBED_DIM = int(os.environ.get("EMBED_DIM", 32))
N_FOLDS = int(os.environ.get("N_FOLDS", "1" if SMOKE else "5"))
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fit_predict_ncm(emb_tr: np.ndarray, y_tr: np.ndarray,
                    emb_eval: np.ndarray) -> np.ndarray:
    """Fit MahalanobisNCM and return Bayes-macro-recall posterior."""
    ncm = MahalanobisNCM(n_classes=3).fit(emb_tr, y_tr)
    return ncm.predict_proba_macro_recall(emb_eval)


def main() -> None:
    log(f"config: SMOKE={SMOKE} EPOCHS={EPOCHS} BS={BATCH_SIZE} "
        f"EMBED_DIM={EMBED_DIM} N_FOLDS={N_FOLDS}")
    device = _select_device()
    log(f"device: {device}")

    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy()
    X_tr_all, X_te, feat_cols = _build_feat_matrix(train, test, info)
    log(f"feat matrix: train={X_tr_all.shape}  test={X_te.shape}")

    # Standardize once (full-train fit; test-scaling is leak-free since
    # test rows have no labels).
    scaler = StandardScaler().fit(X_tr_all)
    X_tr_all = scaler.transform(X_tr_all).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    fold_metrics = []

    splits = list(skf.split(X_tr_all, y))[:N_FOLDS]
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        log(f"=== fold {fold}/{N_FOLDS}: tr={len(tr_idx)} va={len(va_idx)} ===")
        emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
        emb.fit(X_tr_all[tr_idx], y[tr_idx], EPOCHS, BATCH_SIZE)
        emb_tr = emb.transform(X_tr_all[tr_idx])
        emb_va = emb.transform(X_tr_all[va_idx])
        log("  fitting MahalanobisNCM + predicting val")
        probs_va = fit_predict_ncm(emb_tr, y[tr_idx], emb_va)
        oof[va_idx] = probs_va
        bal = balanced_accuracy_score(y[va_idx], probs_va.argmax(1))
        fold_metrics.append(float(bal))
        log(f"  fold {fold} argmax bal_acc = {bal:.5f}")

    # Final test prediction: train embedding on FULL train, fit NCM on full.
    log("=== final test prediction: embed on full train ===")
    emb = Embedder(X_tr_all.shape[1], embed_dim=EMBED_DIM, device=device)
    emb.fit(X_tr_all, y, EPOCHS, BATCH_SIZE)
    emb_tr_full = emb.transform(X_tr_all)
    emb_te = emb.transform(X_te)
    log("  fitting MahalanobisNCM on full train + predicting test")
    test_probs = fit_predict_ncm(emb_tr_full, y, emb_te)

    # Diagnostics on OOF.
    results = dict(smoke=SMOKE, n_folds=N_FOLDS, epochs=EPOCHS,
                   embed_dim=EMBED_DIM, fold_argmax_bal=fold_metrics)
    if N_FOLDS == 5:
        argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
        prior = np.bincount(y, minlength=3) / len(y)
        # Diagnostic only: NCM does not consume tuned bias for its decision.
        # Reporting tuned for direct comparison vs other OOFs in the bank.
        bias, tuned = tune_log_bias(oof, y, prior)
        log(f"OOF argmax={argmax_bal:.5f}  tuned (diagnostic)={tuned:.5f}  "
            f"bias_diag={[round(b,3) for b in bias]}")
        results.update(
            argmax_bal_acc=float(argmax_bal),
            tuned_bal_acc_diagnostic=float(tuned),
            log_bias_diagnostic=[float(b) for b in bias],
        )

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_l2_supcon_ncm{suffix}.npy", oof)
    np.save(ART / f"test_l2_supcon_ncm{suffix}.npy", test_probs)
    with open(ART / f"l2_supcon_ncm_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
