"""Shared helpers for the Tier-1b cross-pollinate + ensemble meta-stacker work.

All loaded OOFs are pinned to the same StratifiedKFold(n_splits=5, seed=42)
split that produced the LB-best 0.98094 submission. The recipe bias used as
the fixed-bias anchor across every blend gate is [1.4324, 1.4689, 3.4008].
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Components that must be excluded from the meta-stacker pool. Reasons logged
# inline; same rules used to validate the LB 0.98094 submission.
EXCLUDE = {
    # LB regressors as DIRECT blend legs (still safe as meta-stacker inputs).
    "recipe_pseudolabel_stage2",
    # Sparse / binary / shape-mismatched carriers.
    "xgb_spec_678", "xgb_spec_36",
    # Prior meta outputs (would cause circular leakage if reused).
    "xgb_metastack", "xgb_metastack_v2", "xgb_metastack_bag3",
    "xgb_nonrule_bag3",
    # Different fold split or partial-fold artefacts (leakage risk).
    "trompt_probe", "b2_groupkfold_region",
    # Already-derived / meta / blend-output artefacts.
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "c0_greedy", "disagree_meta", "selective_router",
    "per_bin_blend", "hedge_avg_lb_bests",
    "step1_greedy_lbbest", "hybrid_binhigh", "meta_v3", "eb_cell",
    # Auto-rejected by shape check anyway, but listed for clarity.
    "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
    "spec_lm_v3_score3", "xgb_bin_medium", "xgb_bin_high", "binhigh",
    "p_flip", "pflip", "missed_high", "flip_correction",
    "xgb_ovo_lowmed", "xgb_ovo_lowmed_nonrule",
    "xgb_ovo_medhigh", "xgb_ovo_medhigh_nonrule",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    """Per-class isotonic regression. Matches the LB-validated calibration."""
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def build_lbbest_stack(y):
    """Reconstructs the LB-best 0.98008 3-stack used as the meta anchor.

    Steps: log-blend(recipe, pseudo_s1, pseudo_s7) at (0.25, 0.35, 0.40), then
    + RealMLP at α=0.20, then + xgb_nonrule_iso at α=0.075. This IS the
    LB-best 3-stack at OOF 0.98061; the meta-stacker built on top of it
    landed LB 0.98094.
    """
    def L(name):
        return (normed(np.load(ART / f"oof_{name}.npy")),
                normed(np.load(ART / f"test_{name}.npy")))
    r, s1, s7 = L("recipe_full_te"), L("recipe_pseudolabel"), L("recipe_pseudolabel_seed7labeler")
    rm, nr = L("realmlp"), L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr[0], nr[1], y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r[0], s1[0], s7[0]], w3)
    lb3_t = log_blend([r[1], s1[1], s7[1]], w3)
    s1_o = log_blend([lb3_o, rm[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rm[1]], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_o], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nr_t], np.array([0.925, 0.075]))
    return s2_o, s2_t


def load_pool(extra_exclude: set[str] | None = None):
    """Load every (oof, test) pair from scripts/artifacts/ matching the
    StratifiedKFold(seed=42) 3-class shape, minus EXCLUDE and any extras.
    """
    drop = EXCLUDE | (extra_exclude or set())
    pool: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in drop:
            continue
        test_p = ART / f"test_{name}.npy"
        if not test_p.exists():
            continue
        try:
            o = np.load(oof_p).astype(np.float32)
            t = np.load(test_p).astype(np.float32)
        except Exception:
            continue
        if o.ndim != 2 or o.shape[1] != 3:
            continue
        if o.shape[0] != 630_000:  # filter per-fold checkpoint artefacts
            continue
        if (o.sum(1) < 1e-3).any():  # detect partial-fold artefacts
            continue
        pool[name] = (normed(o), normed(t))
    return pool


def load_y():
    train = pd.read_csv(DATA / "train.csv")
    return train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)


def bal_at_bias(p, y, bias=BIAS):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))
