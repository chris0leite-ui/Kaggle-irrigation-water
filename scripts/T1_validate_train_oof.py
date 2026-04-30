"""T1 — TRAIN OOF precision validation for the 4-axis override.

We can NOT compute the LLM filter on TRAIN OOF (no labels there) — but
we CAN compute the bank-only precision of {bank_argmax=M & 4b_oof=H}.
The LLM acts as a confirmation filter on top, so the LLM-confirmed
precision should be at least the bank-only precision.

Compute analog of 4b on TRAIN OOF (matching T6_directional_compose.py),
then measure precision in the H->M direction for two filter variants:
  V1: {bank_mean.argmax = M  &  fb_oof = H}                   — 3-axis floor
  V2: V1 + (4b confidence < 0.85 proxy via bank_max < 0.85)  — borderline-only

Apply the T6-documented asymmetry caveat (TRAIN OOF inflates ~15-20pp
because v1 is IN the bank). The break-even is 92% on test-side.

Outputs:
  scripts/artifacts/T1_validate_train_oof_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402
from T6_diversity_helpers import load_y_train, macro_recall, normed, tune_log_bias_simple  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")


def main():
    print("=== T1 validate TRAIN OOF precision ===\n")
    y = load_y_train()
    print(f"y shape: {y.shape}, dist: L={(y==0).sum()} M={(y==1).sum()} H={(y==2).sum()}")

    # Build 4b OOF analog (same as T6_directional_compose.py)
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))

    bv1, _ = tune_log_bias_simple(v1_oof, y)
    bra, _ = tune_log_bias_simple(raw_oof, y)
    bt1, _ = tune_log_bias_simple(t1_oof, y)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1).astype(np.int8)

    una = a_ra == a_t1
    fb_oof = a_v1.copy()
    om = una & (a_v1 != a_ra)
    fb_oof[om] = a_ra[om]
    base_macro = macro_recall(y, fb_oof)
    print(f"4b OOF analog macro: {base_macro:.6f}")

    # 14-bank mean probs on TRAIN OOF
    bank = load_bank("oof")  # (14, n_train, 3)
    bank_mean = bank_mean_probs(bank)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)
    bank_max_prob = bank_mean.max(axis=1)

    # Filter V1: bank says M, 4b says H
    v1_mask = (bank_argmax == 1) & (fb_oof == 2)
    n_v1 = int(v1_mask.sum())
    print(f"\nV1 filter (bank_argmax=M & fb_oof=H): {n_v1} TRAIN OOF rows")

    if n_v1 > 0:
        true = y[v1_mask]
        p_M = float((true == 1).mean())
        p_H = float((true == 2).mean())
        p_L = float((true == 0).mean())
        print(f"  P(true=M): {p_M:.4f}  ({int((true == 1).sum())}/{n_v1})")
        print(f"  P(true=H): {p_H:.4f}  ({int((true == 2).sum())}/{n_v1})")
        print(f"  P(true=L): {p_L:.4f}  ({int((true == 0).sum())}/{n_v1})")
        print(f"  break-even: 0.92")
        # If we override H->M on V1, precision = P(true=M)
        macro_v1 = macro_recall(y, np.where(v1_mask, 1, fb_oof).astype(np.int8))
        print(f"  TRAIN OOF macro after V1 H->M override: {macro_v1:.6f} (delta {macro_v1 - base_macro:+.6f})")

    # Filter V2: V1 + bank_max_prob < 0.85 (borderline only, mimics our test-side selection)
    v2_mask = v1_mask & (bank_max_prob < 0.85)
    n_v2 = int(v2_mask.sum())
    print(f"\nV2 filter (V1 + bank_max<0.85): {n_v2} TRAIN OOF rows")

    if n_v2 > 0:
        true = y[v2_mask]
        p_M2 = float((true == 1).mean())
        p_H2 = float((true == 2).mean())
        p_L2 = float((true == 0).mean())
        print(f"  P(true=M): {p_M2:.4f}  ({int((true == 1).sum())}/{n_v2})")
        print(f"  P(true=H): {p_H2:.4f}  ({int((true == 2).sum())}/{n_v2})")
        print(f"  P(true=L): {p_L2:.4f}  ({int((true == 0).sum())}/{n_v2})")
        macro_v2 = macro_recall(y, np.where(v2_mask, 1, fb_oof).astype(np.int8))
        print(f"  TRAIN OOF macro after V2 H->M override: {macro_v2:.6f} (delta {macro_v2 - base_macro:+.6f})")

    # Apply the T6 asymmetry haircut: train_precision - ~15pp ≈ test_precision
    print("\n=== Test-side projection (T6 asymmetry haircut: -15-20pp) ===")
    if n_v1 > 0:
        proj_v1 = p_M - 0.175
        print(f"  V1: TRAIN OOF P(M)={p_M:.4f} -> projected test P(M)~{proj_v1:.4f}  break-even=0.92")
    if n_v2 > 0:
        proj_v2 = p_M2 - 0.175
        print(f"  V2: TRAIN OOF P(M)={p_M2:.4f} -> projected test P(M)~{proj_v2:.4f}  break-even=0.92")

    # T1 fire details from compose step
    fire_csv = ART / "T1_fire_details.csv"
    fire_n = 0
    if fire_csv.exists():
        fire = pd.read_csv(fire_csv)
        fire_n = len(fire)
        print(f"\nT1 candidate (test-side, 4-axis fire): {fire_n} rows H->M")

    out = {
        "n_train_oof_v1": n_v1,
        "v1_p_true_M": float(p_M) if n_v1 else None,
        "v1_p_true_H": float(p_H) if n_v1 else None,
        "v1_p_true_L": float(p_L) if n_v1 else None,
        "v1_macro_after_override": float(macro_v1) if n_v1 else None,
        "v1_macro_delta": float(macro_v1 - base_macro) if n_v1 else None,
        "n_train_oof_v2": n_v2,
        "v2_p_true_M": float(p_M2) if n_v2 else None,
        "v2_macro_after_override": float(macro_v2) if n_v2 else None,
        "v2_macro_delta": float(macro_v2 - base_macro) if n_v2 else None,
        "base_macro": float(base_macro),
        "v1_projected_test_precision_haircut": float(p_M - 0.175) if n_v1 else None,
        "v2_projected_test_precision_haircut": float(p_M2 - 0.175) if n_v2 else None,
        "n_t1_test_fire_4axis": fire_n,
        "break_even_precision": 0.92,
    }
    (ART / "T1_validate_train_oof_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
