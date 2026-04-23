"""Optuna HP search on fold-0 hold-out with reduced epochs.

Important: fold-0 here is `StratifiedKFold(seed=42)` fold index 0, which
keeps the val-rows aligned with every other OOF on disk. The refit uses
the full 5 folds with the winning HPs.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from model import DigitFTTransformer
from train import make_loader, train_one_fold


SEED = 42


def sample_hp(trial):
    d_token = trial.suggest_categorical("d_token", [96, 128, 192])
    n_heads = trial.suggest_categorical("n_heads", [4, 8])
    if d_token % n_heads != 0:
        raise __import__("optuna").TrialPruned()
    return {
        "d_token": d_token,
        "n_blocks": trial.suggest_int("n_blocks", 3, 5),
        "n_heads": n_heads,
        "attn_drop": trial.suggest_float("attn_drop", 0.05, 0.30),
        "ffn_drop": trial.suggest_float("ffn_drop", 0.05, 0.35),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [1024, 2048]),
    }


def build_objective(data: dict, device, trial_epochs: int, log_fn: Callable):
    y = data["y"]
    prior = data["prior"]
    log_prior = torch.from_numpy(np.log(prior).astype(np.float32)).to(device)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(iter(skf.split(np.zeros(len(y)), y)))

    x_num_tr = data["x_num_tr"][tr_idx]
    x_num_va = data["x_num_tr"][va_idx]
    x_dig_tr = data["x_dig_tr"][tr_idx] if data["x_dig_tr"] is not None else None
    x_dig_va = data["x_dig_tr"][va_idx] if data["x_dig_tr"] is not None else None
    x_cat_tr = data["x_cat_tr"][tr_idx] if data["x_cat_tr"] is not None else None
    x_cat_va = data["x_cat_tr"][va_idx] if data["x_cat_tr"] is not None else None
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    n_num = x_num_tr.shape[1]
    digit_cards = data["digit_cards"]
    cat_cards = data["cat_cards"]

    tens_va = (
        torch.from_numpy(x_num_va).float(),
        torch.from_numpy(x_dig_va).long() if x_dig_va is not None else None,
        torch.from_numpy(x_cat_va).long() if x_cat_va is not None else None,
    )

    def objective(trial):
        hp = sample_hp(trial)
        torch.manual_seed(SEED)
        model = DigitFTTransformer(
            n_num=n_num, digit_cards=digit_cards, cat_cards=cat_cards,
            d_token=hp["d_token"], n_blocks=hp["n_blocks"],
            n_heads=hp["n_heads"], attn_drop=hp["attn_drop"],
            ffn_drop=hp["ffn_drop"],
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                                weight_decay=hp["wd"])
        loader = make_loader(x_num_tr, x_dig_tr, x_cat_tr, y_tr,
                             batch=hp["batch"], shuffle=True)
        _, best = train_one_fold(
            model, opt, {"train": loader}, tens_va, y_va, log_prior,
            epochs=trial_epochs, device=device, base_lr=hp["lr"],
            log_fn=log_fn, log_prefix=f"  [t{trial.number}] ",
        )
        return float(best)

    return objective
