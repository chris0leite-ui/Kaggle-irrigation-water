"""Fold-1 promise gate. Pure function; depends only on numpy + sklearn."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


GATE_ARGMAX_FLOOR = 0.97500
GATE_HIGH_FLOOR = 0.965
GATE_HIGH_LIFT = 0.005   # +0.5pp High over baseline = lever working
GATE_ERROR_CEIL = 1.05
RECIPE_FOLD1_HIGH_RECALL = 0.977


def evaluate(oof_fold, y_fold, recipe_fold1_errs):
    """Returns ("PROCEED" | "ABORT", metrics_dict)."""
    pred = oof_fold.argmax(1)
    argmax_bal = balanced_accuracy_score(y_fold, pred)
    cm = confusion_matrix(y_fold, pred, labels=[0, 1, 2])
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    errs = (pred != y_fold).sum()
    err_ratio = errs / max(recipe_fold1_errs, 1)

    pass_argmax = argmax_bal >= GATE_ARGMAX_FLOOR
    pass_high_floor = recalls[2] >= GATE_HIGH_FLOOR
    pass_high_lift = recalls[2] >= (RECIPE_FOLD1_HIGH_RECALL + GATE_HIGH_LIFT)
    pass_errs = err_ratio <= GATE_ERROR_CEIL

    n_pass = int(pass_argmax) + int(pass_high_floor) + int(pass_errs)
    decision = "PROCEED" if (n_pass >= 2 or pass_high_lift) else "ABORT"

    return decision, dict(
        argmax_bal=float(argmax_bal),
        recall_low=float(recalls[0]),
        recall_med=float(recalls[1]),
        recall_high=float(recalls[2]),
        errors=int(errs),
        err_ratio=float(err_ratio),
        pass_argmax=bool(pass_argmax),
        pass_high_floor=bool(pass_high_floor),
        pass_high_lift=bool(pass_high_lift),
        pass_errs=bool(pass_errs),
        decision=decision,
    )
