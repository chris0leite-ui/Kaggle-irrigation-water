"""Blend-gate analysis for the TabPFN-10k 1-fold probe.

Pulls oof_tabpfn_10k.npy + test_tabpfn_10k.npy from kernel output,
restricts to fold-0 val rows (~126k), reconstructs the LB-best primary
fold-0 slice, and reports:
  - standalone fold-0 tuned bal_acc + per-class recall + errs
  - Jaccard vs LB-best primary fold-0
  - blend gate at the same fixed bias [1.4324, 1.4689, 3.4008] used by
    every prior LB-validated stack add

Decision rule:
  PASS (proceed to full 5-fold push):
    - fold-0 standalone tuned bal_acc ≥ 0.974
    - fold-0 Jaccard < 0.78 vs LB-best primary
    - fold-0 errs ≤ 1.05 × LB-best fold-0 errs (the +5% magnitude rule
      established by the 13 prior NN nulls + RealMLP n_ens=1 success)
  FAIL (close lever, document, lock LB 0.98094):
    - any of the above violated.

This is a SIGNAL CHECK only, not an LB-probe candidate. A 1-fold
standalone OOF is not deployable; full 5-fold is required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, build_lbbest_stack, iso_cal, load_y, normed,
)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return cm.diagonal() / np.maximum(cm.sum(axis=1), 1)


def err_jaccard(y, pa, pb):
    ea = pa != y
    eb = pb != y
    inter = (ea & eb).sum()
    union = (ea | eb).sum()
    return float(inter) / max(int(union), 1)


def main():
    if not (ART / "oof_tabpfn_10k.npy").exists():
        log("ERROR: scripts/artifacts/oof_tabpfn_10k.npy not found.")
        log("Pull from Kaggle kernel output first:")
        log("  kaggle kernels output chrisleitescha/irrigation-tabpfn-10k -p scripts/artifacts/")
        sys.exit(1)

    y = load_y()
    oof_pfn = np.load(ART / "oof_tabpfn_10k.npy")
    test_pfn = np.load(ART / "test_tabpfn_10k.npy")
    log(f"loaded TabPFN-10k OOF shape={oof_pfn.shape}, test shape={test_pfn.shape}")

    # Identify fold-0 val rows from sentinel pattern (rows that sum > 0).
    nonzero = oof_pfn.sum(axis=1) > 1e-3
    va_idx = np.where(nonzero)[0]
    log(f"fold-0 val rows: {len(va_idx):,}  (expected ~126k for fold 0)")
    if len(va_idx) < 100_000:
        log("WARNING: fewer val rows than expected — kernel may have errored.")

    y_va = y[va_idx]
    P_pfn_va = normed(oof_pfn[va_idx])

    # Reconstruct LB-best primary on fold-0 slice.
    log("reconstructing LB-best primary 4-stack (LB 0.98094)...")
    s2_o, _ = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    iso_o, _ = iso_cal(meta_o, np.load(ART / "test_xgb_metastack.npy"), y)
    P_lb_full = log_blend([s2_o, iso_o], np.array([0.70, 0.30]))
    P_lb_va = P_lb_full[va_idx]

    # Compute per-class recall, errs, balanced accuracy.
    pred_pfn = (np.log(np.clip(P_pfn_va, 1e-12, 1)) + BIAS).argmax(1)
    pred_lb = (np.log(np.clip(P_lb_va, 1e-12, 1)) + BIAS).argmax(1)

    bal_pfn = balanced_accuracy_score(y_va, pred_pfn)
    bal_lb = balanced_accuracy_score(y_va, pred_lb)
    rec_pfn = per_class_recall(y_va, pred_pfn)
    rec_lb = per_class_recall(y_va, pred_lb)
    errs_pfn = int((pred_pfn != y_va).sum())
    errs_lb = int((pred_lb != y_va).sum())
    jacc = err_jaccard(y_va, pred_pfn, pred_lb)

    log("=" * 80)
    log(f"FOLD-0 standalone (TabPFN @ recipe bias):")
    log(f"  bal_acc = {bal_pfn:.5f}  errs = {errs_pfn}  "
        f"PCR=[L={rec_pfn[0]:.4f}, M={rec_pfn[1]:.4f}, H={rec_pfn[2]:.4f}]")
    log(f"FOLD-0 LB-best primary (anchor):")
    log(f"  bal_acc = {bal_lb:.5f}  errs = {errs_lb}  "
        f"PCR=[L={rec_lb[0]:.4f}, M={rec_lb[1]:.4f}, H={rec_lb[2]:.4f}]")
    log(f"Jaccard(TabPFN errs, LB-best errs) = {jacc:.4f}")
    log(f"Δ errs vs LB-best: {errs_pfn - errs_lb:+d}  "
        f"({(errs_pfn / errs_lb - 1) * 100:+.1f}%)")

    # Tuned bias on fold-0 alone (informational; any standalone tune is
    # selection-overfit prone since one fold = strong calibration shift).
    log("tuning log-bias on fold-0 (informational; not for blending)")
    from common import tune_log_bias
    prior = np.bincount(y_va, minlength=3) / len(y_va)
    bias_tuned, bal_tuned = tune_log_bias(P_pfn_va, y_va, prior)
    rec_tuned = per_class_recall(
        y_va,
        (np.log(np.clip(P_pfn_va, 1e-12, 1)) + bias_tuned).argmax(1)
    )
    log(f"  fold-0 tuned bal = {bal_tuned:.5f}  bias={bias_tuned.round(3).tolist()}  "
        f"PCR=[L={rec_tuned[0]:.4f}, M={rec_tuned[1]:.4f}, H={rec_tuned[2]:.4f}]")

    # Decision gate.
    log("=" * 80)
    gate_bal = bal_tuned >= 0.974
    gate_jacc = jacc < 0.78
    gate_errs = errs_pfn <= 1.05 * errs_lb
    overall = gate_bal and gate_jacc and gate_errs
    log(f"DECISION GATE for full 5-fold push:")
    log(f"  bal_tuned ≥ 0.974   :  {gate_bal}  ({bal_tuned:.5f})")
    log(f"  Jaccard < 0.78      :  {gate_jacc}  ({jacc:.4f})")
    log(f"  errs ≤ 1.05 × anchor:  {gate_errs}  ({errs_pfn} vs cap {1.05 * errs_lb:.0f})")
    log(f"  OVERALL             :  {'PASS' if overall else 'FAIL'}")

    out = {
        "fold_run": 0,
        "n_val_rows": int(len(va_idx)),
        "tabpfn_at_recipe_bias": {
            "bal_acc": float(bal_pfn), "errs": errs_pfn,
            "rec_low": float(rec_pfn[0]), "rec_med": float(rec_pfn[1]),
            "rec_high": float(rec_pfn[2]),
        },
        "lbbest_at_recipe_bias": {
            "bal_acc": float(bal_lb), "errs": errs_lb,
            "rec_low": float(rec_lb[0]), "rec_med": float(rec_lb[1]),
            "rec_high": float(rec_lb[2]),
        },
        "jaccard_vs_lbbest": float(jacc),
        "tabpfn_tuned": {
            "bal_acc": float(bal_tuned),
            "bias": [float(b) for b in bias_tuned],
            "rec_low": float(rec_tuned[0]), "rec_med": float(rec_tuned[1]),
            "rec_high": float(rec_tuned[2]),
        },
        "gate": {
            "bal_tuned_ge_974": bool(gate_bal),
            "jaccard_lt_078": bool(gate_jacc),
            "errs_le_105pct_anchor": bool(gate_errs),
            "overall": "PASS" if overall else "FAIL",
        },
    }
    out_path = ART / "blend_tabpfn_10k_fold0_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"results saved to {out_path}")


if __name__ == "__main__":
    main()
