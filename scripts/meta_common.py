"""Shared utilities for Option 3 (disagreement meta-stack) + Option 1 (router).

Loads:
    - y (train labels)
    - dgp_score + signed distances (from common.add_distance_features)
    - LB-best 3-way teacher (recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40 log-blend)
    - Candidate OOF bank

Teacher weights and recipe bias match the LB-0.98005 submission on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CLS2IDX, N_FOLDS, SEED, add_distance_features, fast_bal_acc, tune_log_bias,
)

ART = Path("scripts/artifacts")
TEACHER_W = dict(recipe_full_te=0.25, recipe_pseudolabel=0.35,
                 recipe_pseudolabel_seed7labeler=0.40)
CANDIDATES = [
    "xgb_nonrule",
    "xgb_corn",
    "recipe_full_te_gby",
    "recipe_full_te_fexboth",
    "recipe_catboost",
    "recipe_allpairs",
    "recipe_full_te_dae",
]
EPS = 1e-9


def recipe_bias() -> np.ndarray:
    res = json.loads((ART / "recipe_full_te_results.json").read_text())
    return np.asarray(res["log_bias"], dtype=np.float64)


def log_blend(probs_list, weights) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(probs_list[0], dtype=np.float64)
    for wi, p in zip(w, probs_list):
        logits += wi * np.log(np.clip(p, EPS, 1.0))
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def build_teacher() -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_teacher, test_teacher) as LB-best 3-way log-blend."""
    names = list(TEACHER_W.keys())
    weights = [TEACHER_W[n] for n in names]
    oofs = [np.load(ART / f"oof_{n}.npy") for n in names]
    tests = [np.load(ART / f"test_{n}.npy") for n in names]
    return log_blend(oofs, weights), log_blend(tests, weights)


def load_candidates(names=CANDIDATES):
    oofs, tests = {}, {}
    for n in names:
        oofs[n] = np.load(ART / f"oof_{n}.npy")
        tests[n] = np.load(ART / f"test_{n}.npy")
    return oofs, tests


def load_y_and_features():
    """Returns y (630k,), dgp_score (630k,), signed_dists (630k, 4) for train,
    plus (test_dgp_score, test_signed_dists) for test."""
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int64)

    tr_dist = add_distance_features(tr)
    te_dist = add_distance_features(te)

    dist_cols = ["sm_dist", "rf_dist", "tc_dist", "ws_dist"]
    return (y,
            tr_dist["dgp_score"].to_numpy().astype(np.int16),
            tr_dist[dist_cols].to_numpy().astype(np.float32),
            te_dist["dgp_score"].to_numpy().astype(np.int16),
            te_dist[dist_cols].to_numpy().astype(np.float32))


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1.0)
    return (-(q * np.log(q)).sum(axis=1)).astype(np.float32)


def argmax_int8(p: np.ndarray) -> np.ndarray:
    return p.argmax(axis=1).astype(np.int8)


def teacher_report(oof_t: np.ndarray, y: np.ndarray, bias: np.ndarray) -> dict:
    """Report teacher's OOF stats at fixed bias."""
    prior = np.bincount(y, minlength=3) / len(y)
    log_t = np.log(np.clip(oof_t, EPS, 1.0))
    pred_fixed = (log_t + bias).argmax(axis=1)
    fixed_ba = fast_bal_acc(y, pred_fixed)
    tuned_bias, tuned_ba = tune_log_bias(oof_t, y, prior, high_grid_wide=True)
    return dict(
        fixed_bias_ba=float(fixed_ba),
        tuned_bias=tuned_bias.tolist(),
        tuned_ba=float(tuned_ba),
        errs_fixed=int((pred_fixed != y).sum()),
    )


def get_folds(y: np.ndarray):
    """Yield (fold_idx, tr_idx, va_idx) for pinned StratifiedKFold(seed=42)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))
