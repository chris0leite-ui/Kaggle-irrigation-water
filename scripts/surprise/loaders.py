"""Load anchors, helpers, OOFs, and y for the surprise-options sweep.

Anchor v1 RF natural (LB 0.98129) is reproduced exactly from the OOF
oof_sklearn_rf_meta_natural_v1_lb98129.npy at v1 tuned bias
[0.4324, 0.8689, 3.2008]; the test argmax matches submissions/
submission_sklearn_rf_meta_natural_standalone.csv (md5 44ad26).

The 4 OTHERS are the LB-validated set used for the LB 0.98134 override
(reproduced exactly: 120 row-diffs match): {raw, tier1b, lb3, 3way}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def load_y() -> np.ndarray:
    df = pd.read_csv(DATA / "train.csv")
    return df["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)


def load_test_argmax(name: str) -> np.ndarray:
    df = pd.read_csv(SUB / name)
    return df["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)


def load_test_ids() -> np.ndarray:
    df = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone.csv",
                     usecols=["id"])
    return df["id"].to_numpy()


def normed(p: np.ndarray) -> np.ndarray:
    return p / p.sum(axis=1, keepdims=True)


def load_oof_test(stem: str):
    oof = normed(np.load(ART / f"oof_{stem}.npy").astype(np.float64))
    test = normed(np.load(ART / f"test_{stem}.npy").astype(np.float64))
    return oof, test


def reconstruct_3way():
    """3way = log_blend(recipe, pseudo_s1, pseudo_s7) at (0.25, 0.35, 0.40)."""
    r_o, r_t = load_oof_test("recipe_full_te")
    s1_o, s1_t = load_oof_test("recipe_pseudolabel")
    s7_o, s7_t = load_oof_test("recipe_pseudolabel_seed7labeler")
    w = np.array([0.25, 0.35, 0.40])
    return log_blend([r_o, s1_o, s7_o], w), log_blend([r_t, s1_t, s7_t], w)


def reconstruct_lb3():
    """LB-best 3-stack (LB 0.98008): lb3_recipe_pseudo + RealMLP α=0.20 +
    xgb_nonrule_iso α=0.075. Reproduces oof 0.98061 / submission_lb3_realmlp_nonruleiso.csv.
    """
    from sklearn.isotonic import IsotonicRegression
    y = load_y()
    r_o, r_t = load_oof_test("recipe_full_te")
    s1_o, s1_t = load_oof_test("recipe_pseudolabel")
    s7_o, s7_t = load_oof_test("recipe_pseudolabel_seed7labeler")
    rm_o, rm_t = load_oof_test("realmlp")
    nr_o, nr_t = load_oof_test("xgb_nonrule")
    # iso-cal nonrule per-class on full OOF (matches build_lbbest_stack)
    nr_o_iso = np.zeros_like(nr_o)
    nr_t_iso = np.zeros_like(nr_t)
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(nr_o[:, k], (y == k).astype(np.float32))
        nr_o_iso[:, k] = ir.predict(nr_o[:, k])
        nr_t_iso[:, k] = ir.predict(nr_t[:, k])
    nr_o_iso = normed(nr_o_iso)
    nr_t_iso = normed(nr_t_iso)
    lb3_o = log_blend([r_o, s1_o, s7_o], np.array([0.25, 0.35, 0.40]))
    lb3_t = log_blend([r_t, s1_t, s7_t], np.array([0.25, 0.35, 0.40]))
    s1_oof = log_blend([lb3_o, rm_o], np.array([0.8, 0.2]))
    s1_t2 = log_blend([lb3_t, rm_t], np.array([0.8, 0.2]))
    s2_oof = log_blend([s1_oof, nr_o_iso], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t2, nr_t_iso], np.array([0.925, 0.075]))
    return s2_oof, s2_t


def oof_argmax_at_bias(oof: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(oof, 1e-12, 1.0)) + bias).argmax(1).astype(np.int8)


def all_helpers(y: np.ndarray) -> dict:
    """Return {name: (oof, test, oof_argmax_at_tuned_bias, test_argmax_csv)} for the
    4 OTHERS plus T4 (rawashishsin pseudo)."""
    out = {}
    # rawashishsin: bias from OOF tuning
    o, t = load_oof_test("rawashishsin_2600")
    b, _ = tune_log_bias(o, y, np.bincount(y, minlength=3) / len(y))
    out["raw"] = (o, t, oof_argmax_at_bias(o, b), load_test_argmax("submission_rawashishsin_2600_standalone.csv"))
    # tier1b 4-stack
    o, t = load_oof_test("tier1b_greedy_meta")
    b, _ = tune_log_bias(o, y, np.bincount(y, minlength=3) / len(y))
    out["tier1b"] = (o, t, oof_argmax_at_bias(o, b), load_test_argmax("submission_tier1b_greedy_meta.csv"))
    # lb3 reconstructed
    o, t = reconstruct_lb3()
    b, _ = tune_log_bias(o, y, np.bincount(y, minlength=3) / len(y))
    out["lb3"] = (o, t, oof_argmax_at_bias(o, b), load_test_argmax("submission_lb3_realmlp_nonruleiso.csv"))
    # 3way reconstructed
    o, t = reconstruct_3way()
    b, _ = tune_log_bias(o, y, np.bincount(y, minlength=3) / len(y))
    out["3way"] = (o, t, oof_argmax_at_bias(o, b), load_test_argmax("submission_3way_recipe025_s1035_s7040.csv"))
    # T4 rawashishsin pseudo
    o, t = load_oof_test("rawashishsin_pseudo")
    b, _ = tune_log_bias(o, y, np.bincount(y, minlength=3) / len(y))
    # T4 has no published submission CSV; use OOF/test argmax directly
    out["t4"] = (o, t, oof_argmax_at_bias(o, b), oof_argmax_at_bias(t, b))
    return out


def load_v1_anchor():
    """Returns (oof_argmax_at_tuned_bias, test_argmax_from_csv, oof_array, tuned_bias)."""
    o, t = load_oof_test("sklearn_rf_meta_natural_v1_lb98129")
    bias_v1 = np.array([0.4324, 0.8689, 3.2008])  # from CLAUDE.md (LB 0.98129 entry)
    return oof_argmax_at_bias(o, bias_v1), \
        load_test_argmax("submission_sklearn_rf_meta_natural_standalone.csv"), \
        o, bias_v1


def load_winner_anchor():
    """0.98140 winner: only have its test-side argmax (it's a test override of v1).
    For OOF analog we apply the same k=2 unanimous {raw, tier1b} mask to v1's OOF."""
    return load_test_argmax("submission_2other_raw_tier1b_k2.csv")
