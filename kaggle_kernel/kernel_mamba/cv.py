"""5-fold CV loop + log-bias tuning + per-fold checkpointed save.

Aligned with every other OOF on main via StratifiedKFold(seed=42).
Per-fold checkpoint pattern (lesson from SMOTE container-rehydrate
incident): persist OOF/test/JSON immediately after each fold so any
process death leaves recoverable progress.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import CATS, NUMS, TARGET, SEED, IDX2CLS
from model import fit_one_fold


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
                  eps: float = 1e-9):
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)

    def score(b):
        pred = (log_oof + b).argmax(1)
        per = np.zeros(3)
        for k in range(3):
            per[k] = ((pred == k) & (y == k)).sum() / max(cc[k], 1)
        return float(per.mean())

    best = score(bias)
    g_def = np.linspace(-3.0, 3.0, 61)
    g_high = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = g_high if k == 2 else g_def
            base = bias.copy()
            cands = []
            for g in grid:
                base[k] = bias[k] + g
                cands.append(score(base))
            j = int(np.argmax(cands))
            if cands[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = cands[j]
                improved = True
        if not improved:
            break
    return bias, best


def run_cv(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame,
           n_folds: int, max_folds: int, n_epochs: int, batch_size: int,
           lr: float, weight_decay: float, d_model: int, n_layers: int,
           d_state: int, d_conv: int, expand: int, dropout: float,
           fold1_kill_s: int, total_kill_s: int, out_dir: Path,
           suffix: str):
    y = train[TARGET].to_numpy().astype(np.int64)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_ba = []
    folds_done = 0
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(train[CATS + NUMS], y), 1,
    ):
        print(f"=== fold {fold}/{n_folds} ===", flush=True)
        fold_tr = pd.concat([train.iloc[tr_idx], orig], axis=0,
                            ignore_index=True)
        X_tr = fold_tr[CATS + NUMS]
        y_tr = fold_tr[TARGET].to_numpy().astype(np.int64)
        X_va = train.iloc[va_idx][CATS + NUMS].copy()
        X_te = test[CATS + NUMS].copy()
        p_va, p_te = fit_one_fold(
            X_tr, y_tr, X_va, X_te,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
        )
        oof[va_idx] = p_va.astype(np.float32)
        test_pred += p_te.astype(np.float32) / n_folds
        ba = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_ba.append(float(ba))
        folds_done = fold
        # Per-fold checkpoint — survive any process death
        np.save(out_dir / f"oof_mamba{suffix}.npy", oof)
        np.save(out_dir / f"test_mamba{suffix}.npy", test_pred)
        el = time.time() - t0
        print(f"  fold {fold} argmax bal_acc = {ba:.5f} "
              f"(elapsed {el/60:.1f}m)", flush=True)
        if fold == 1 and el > fold1_kill_s:
            print(f"!! FOLD-1 WALL-TIME KILL {el/60:.1f}m > "
                  f"{fold1_kill_s/60:.0f}m", flush=True)
            break
        if el > total_kill_s:
            print(f"!! TOTAL WALL-TIME KILL {el/60:.1f}m > "
                  f"{total_kill_s/60:.0f}m", flush=True)
            break
        if max_folds is not None and fold >= max_folds:
            print(f"!! MAX_FOLDS reached ({max_folds})", flush=True)
            break
    if folds_done and folds_done < n_folds:
        test_pred *= n_folds / folds_done
        np.save(out_dir / f"test_mamba{suffix}.npy", test_pred)
    return oof, test_pred, fold_ba, folds_done, y


def save_outputs(out_dir: Path, oof, test_pred, test_ids, fold_ba,
                 folds_done, n_folds, y, suffix: str = ""):
    prior = np.bincount(y, minlength=3) / len(y)
    nz = oof.sum(axis=1) > 0
    if nz.sum() > 0:
        bias, tuned = tune_log_bias(oof[nz], y[nz], prior)
    else:
        bias, tuned = -np.log(prior), 0.0
    overall = (balanced_accuracy_score(y[nz], oof[nz].argmax(1))
               if nz.sum() > 0 else 0.0)
    print(f"tuned OOF bal_acc = {tuned:.5f} "
          f"bias={bias.round(4).tolist()} "
          f"({folds_done}/{n_folds} folds, "
          f"covered={int(nz.sum()):,}/{len(oof):,})", flush=True)
    np.save(out_dir / f"oof_mamba{suffix}.npy", oof)
    np.save(out_dir / f"test_mamba{suffix}.npy", test_pred)
    pred = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({
        "id": test_ids, TARGET: [IDX2CLS[i] for i in pred],
    })
    sub.to_csv(out_dir / f"submission_mamba{suffix}_tuned.csv", index=False)
    (out_dir / f"mamba{suffix}_results.json").write_text(json.dumps({
        "overall_argmax_bal_acc": float(overall),
        "tuned_log_bias_bal_acc": float(tuned),
        "log_bias": bias.tolist(),
        "fold_ba": fold_ba,
        "n_folds": n_folds,
        "folds_completed": folds_done,
        "seed": SEED,
    }, indent=2))
    print(f"wrote oof_mamba{suffix}.npy / test_mamba{suffix}.npy / "
          f"submission_mamba{suffix}_tuned.csv / "
          f"mamba{suffix}_results.json", flush=True)
