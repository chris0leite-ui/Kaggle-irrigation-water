"""5-fold CV loop + log-bias tuning + output save.

Aligned with every other OOF on main via StratifiedKFold(seed=42).
Fold-1 + total wall-time kill switches mirror the RealMLP kernel.
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
from model import fit_one_fold, make_dataset


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
           device, n_folds: int, n_epochs: int, fold1_kill_s: int,
           total_kill_s: int):
    y = train[TARGET].to_numpy().astype(np.int64)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_ba = []
    folds_done = 0
    t0 = time.time()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(train[CATS + NUMS], y), 1,
    ):
        print(f"=== fold {fold}/{n_folds} ===", flush=True)
        fold_tr = pd.concat([train.iloc[tr_idx], orig], axis=0,
                            ignore_index=True)
        fold_va = train.iloc[va_idx].copy()
        tr_ds = make_dataset(fold_tr, CATS, NUMS, target=TARGET)
        va_ds = make_dataset(fold_va, CATS, NUMS, target=TARGET)
        te_ds = make_dataset(test.assign(**{TARGET: 0}), CATS, NUMS,
                             target=TARGET)
        p_va, p_te = fit_one_fold(tr_ds, va_ds, te_ds, device,
                                  n_epochs=n_epochs)
        oof[va_idx] = p_va.astype(np.float32)
        test_pred += p_te.astype(np.float32) / n_folds
        ba = balanced_accuracy_score(y[va_idx], p_va.argmax(1))
        fold_ba.append(float(ba))
        folds_done = fold
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
    if folds_done and folds_done < n_folds:
        test_pred *= n_folds / folds_done
    return oof, test_pred, fold_ba, folds_done, y


def save_outputs(out_dir: Path, oof, test_pred, test_ids, fold_ba,
                 folds_done, n_folds, y):
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
          f"({folds_done}/{n_folds} folds)", flush=True)
    np.save(out_dir / "oof_trompt.npy", oof)
    np.save(out_dir / "test_trompt.npy", test_pred)
    pred = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({
        "id": test_ids, TARGET: [IDX2CLS[i] for i in pred],
    })
    sub.to_csv(out_dir / "submission_trompt_tuned.csv", index=False)
    (out_dir / "trompt_results.json").write_text(json.dumps({
        "overall_argmax_bal_acc": float(overall),
        "tuned_log_bias_bal_acc": float(tuned),
        "log_bias": bias.tolist(),
        "fold_ba": fold_ba,
        "n_folds": n_folds,
        "folds_completed": folds_done,
        "seed": SEED,
    }, indent=2))
    print("wrote oof_trompt.npy / test_trompt.npy / "
          "submission_trompt_tuned.csv / trompt_results.json", flush=True)
