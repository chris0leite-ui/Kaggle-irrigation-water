"""Build LB-ready submissions for R2 + R5 candidates.

3 candidates ranked from most-aggressive to most-conservative:

1. R2-FULL-ISO α=0.45 (depth=2 heavy-reg, full-OOF iso):
   OOF 0.98124, +0.00039 vs PRIMARY 0.98084. Best raw OOF. Highest
   risk of iso-leak inflation.

2. R2+R5 PER-FOLD-ISO α=0.45 (depth=2 + leak-safe iso):
   OOF 0.98113, +0.00029 vs PRIMARY. Honest signal — same depth=2
   advantage but with leak removed. Best LB-transfer probability.

3. R2+R5 PER-FOLD-ISO α=0.25 (conservative dilution):
   OOF 0.98098, +0.00014 vs PRIMARY. Smallest test-set delta vs
   PRIMARY; safest fallback if (1) and (2) regress.

All use the LB-validated REPLACE architecture: 3-stack + meta_heavy
at α (replacing meta_v1_iso). Per-class recall guardrail PASS for
all three at OOF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, log, normed,
)


def per_fold_iso(oof, test, y, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    oo = np.zeros_like(oof, dtype=np.float32)
    for tr, va in skf.split(oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
            ir.fit(oof[tr, c], (y[tr] == c).astype(np.float32))
            oo[va, c] = ir.predict(oof[va, c])
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def predict_with_bias(p):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def emit(label, lb3_t, meta_t_iso, alpha, ids, y, lb3_o, meta_o_iso,
         primary_bal, primary_pcr_anchor):
    """Build OOF + test blend, verify guard, write CSV."""
    blend_o = log_blend([lb3_o, meta_o_iso], np.array([1-alpha, alpha]))
    blend_t = log_blend([lb3_t, meta_t_iso], np.array([1-alpha, alpha]))
    pred_oof = predict_with_bias(blend_o)
    pred_tst = predict_with_bias(blend_t)
    bal = balanced_accuracy_score(y, pred_oof)
    pcr = per_class_recall(y, pred_oof)
    delta = bal - primary_bal
    guard = all(pcr[c] >= primary_pcr_anchor[c] - 5e-4 for c in range(3))
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids, "Irrigation_Need": [cls_map[i] for i in pred_tst]})
    p = SUB / f"submission_{label}.csv"
    sub.to_csv(p, index=False)
    # diff vs primary
    primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    diff = (sub["Irrigation_Need"] != primary["Irrigation_Need"]).sum()
    log(f"  {label:50s} OOF={bal:.5f}  Δ={delta:+.5f}  guard={'PASS' if guard else 'FAIL'}  "
        f"PCR=L{pcr[0]:.4f}/M{pcr[1]:.4f}/H{pcr[2]:.4f}  diff_from_primary={diff}")
    log(f"     pred dist: {dict(sub['Irrigation_Need'].value_counts())}")


def main():
    y = load_y()
    log("loading anchor")
    lb3_o, lb3_t = build_lbbest_stack(y)
    m_v1_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    m_v1_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv1_iso_o, mv1_iso_t = iso_cal(m_v1_o, m_v1_t, y)
    primary_o = log_blend([lb3_o, mv1_iso_o], np.array([0.7, 0.3]))
    primary_pred = predict_with_bias(primary_o)
    primary_bal = balanced_accuracy_score(y, primary_pred)
    primary_pcr = per_class_recall(y, primary_pred)
    log(f"  PRIMARY OOF = {primary_bal:.5f} PCR=L{primary_pcr[0]:.4f}/M{primary_pcr[1]:.4f}/H{primary_pcr[2]:.4f}")

    log("loading meta_heavy raw")
    m_h_o = normed(np.load(ART / "oof_xgb_metastack_heavy.npy").astype(np.float32))
    m_h_t = normed(np.load(ART / "test_xgb_metastack_heavy.npy").astype(np.float32))

    log("computing full-OOF iso + per-fold iso for meta_heavy")
    mh_full_o, mh_full_t = iso_cal(m_h_o, m_h_t, y)
    mh_pf_o, mh_pf_t = per_fold_iso(m_h_o, m_h_t, y)

    ids = pd.read_csv("data/test.csv")["id"].values
    log("\n=== R2 + R5 LB-ready candidates ===")
    emit("r2_heavy_fulliso_a045", lb3_t, mh_full_t, 0.45, ids, y,
         lb3_o, mh_full_o, primary_bal, primary_pcr)
    emit("r2r5_heavy_perfoldiso_a045", lb3_t, mh_pf_t, 0.45, ids, y,
         lb3_o, mh_pf_o, primary_bal, primary_pcr)
    emit("r2r5_heavy_perfoldiso_a025_safe", lb3_t, mh_pf_t, 0.25, ids, y,
         lb3_o, mh_pf_o, primary_bal, primary_pcr)
    log("\nready for LB probe.")


if __name__ == "__main__":
    main()
