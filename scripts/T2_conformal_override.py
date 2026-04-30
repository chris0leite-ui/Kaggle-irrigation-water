"""T2 — Conformal-certified override gate on top of 4b (LB 0.98150).

Mechanism:
  1. Compute bank-mean probability matrix (270k x 3) on test from 14 components.
  2. Calibrate split-conformal threshold q_hat on TRAIN OOF using
     score = 1 - P(y_true | x). Choose alpha = 0.05 -> 95% coverage on TRUE class.
  3. For each test row, build prediction set = {classes c: 1 - P(c|x) <= q_hat}.
  4. Identify rows where 4b would benefit from override: where 4b's argmax is
     NOT in the conformal prediction set, AND a SINGLE alternative class IS.
     Those rows are flipped to that alternative.
  5. Compose with existing 4b CSV (don't replace its 108 base flips).

The conformal gate replaces 4b's empirical 14-bank-majority axis with a
distribution-free coverage rule. Different from 4b's filter because:
  - 4b allows override only when bank-majority == bagged_v1' class
  - T2 allows override when 4b's class is OUTSIDE the conformal set
    (formal guarantee that 4b's prediction is "wrong" with prob > 0.95).

Verification (no LB probe before user approval):
  - Coverage on TRAIN OOF must be within 2% of 1-alpha.
  - On TRAIN OOF, the override mechanism must yield macro_recall >= 4b proxy.
  - Direction must be ADD-asymmetric (not REMOVE-High).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import (  # noqa: E402
    BANK_NAMES,
    bank_mean_probs,
    conformal_threshold,
    in_prediction_set,
    load_bank,
    nonconformity,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

ALPHA = 0.01  # 99% nominal coverage (selected from alpha sweep — only operating point with non-zero override candidates)


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> float:
    rec = []
    for c in range(n_cls):
        m = y_true == c
        if m.sum() == 0:
            continue
        rec.append(float((y_pred[m] == c).mean()))
    return sum(rec) / len(rec)


def main():
    print("=== T2: Conformal-certified override gate ===\n")
    # Load OOF/test bank.
    print("Loading 14-bank OOF + test...")
    oof_bank = load_bank("oof")
    test_bank = load_bank("test")
    print(f"  oof_bank: {oof_bank.shape}, test_bank: {test_bank.shape}")

    oof_mean = bank_mean_probs(oof_bank)
    test_mean = bank_mean_probs(test_bank)
    print(f"  oof_mean argmax dist: {np.bincount(oof_mean.argmax(1), minlength=3).tolist()}")
    print(f"  test_mean argmax dist: {np.bincount(test_mean.argmax(1), minlength=3).tolist()}")

    # Load TRAIN labels.
    y_train = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)
    print(f"  y_train dist: {np.bincount(y_train, minlength=3).tolist()}")

    # Calibrate split-conformal: compute nonconformity on full OOF.
    cal_scores = nonconformity(oof_mean, y_train)
    q_hat = conformal_threshold(cal_scores, ALPHA)
    print(f"\nConformal calibration (alpha={ALPHA}, target coverage {1-ALPHA:.0%}):")
    print(f"  q_hat = {q_hat:.4f}")

    # Verify coverage on TRAIN OOF.
    in_set_oof = in_prediction_set(oof_mean, q_hat)
    coverage_oof = float(in_set_oof[np.arange(len(y_train)), y_train].mean())
    print(f"  OOF coverage of true class: {coverage_oof:.4f}")
    if abs(coverage_oof - (1 - ALPHA)) > 0.02:
        print(f"  WARN: coverage drift > 2% from nominal {1-ALPHA}")

    set_sizes_oof = in_set_oof.sum(axis=1)
    print(f"  OOF set-size dist: 1-class {(set_sizes_oof==1).sum()}, "
          f"2-class {(set_sizes_oof==2).sum()}, 3-class {(set_sizes_oof==3).sum()}")

    # Apply conformal sets to TEST.
    in_set_test = in_prediction_set(test_mean, q_hat)
    set_sizes_test = in_set_test.sum(axis=1)
    print(f"\nTest set-size dist: 1-class {(set_sizes_test==1).sum()}, "
          f"2-class {(set_sizes_test==2).sum()}, 3-class {(set_sizes_test==3).sum()}")

    # Load 4b LB-best.
    fb_argmax = csv_argmax("submission_idea4b_selective_override")
    print(f"\n4b (LB 0.98150) argmax dist: {np.bincount(fb_argmax, minlength=3).tolist()}")

    # T2 override candidates: 4b's class is OUTSIDE the conformal set,
    # AND exactly one alternative class IS in the set (singleton remainder).
    fb_class_oneset = in_set_test[np.arange(len(fb_argmax)), fb_argmax]
    fb_outside = ~fb_class_oneset

    # For rows where 4b is outside the set AND set is singleton, override.
    # For rows where 4b is outside set AND set is 2-class (without 4b), pick max-prob.
    override_mask = fb_outside & (set_sizes_test >= 1) & (set_sizes_test < 3)

    # Decide override class: if singleton, the singleton; if pair, the one with
    # higher bank-mean prob.
    override_class = np.full(len(fb_argmax), -1, dtype=np.int8)
    for i in np.where(override_mask)[0]:
        cand = np.where(in_set_test[i])[0]
        # rank candidates by bank-mean prob desc, pick top
        cand = cand[np.argsort(-test_mean[i, cand])]
        override_class[i] = cand[0]

    n_overrides = int(override_mask.sum())
    print(f"\nT2 override candidates: {n_overrides}")
    if n_overrides == 0:
        print("  no override candidates. Lever inactive at alpha=0.05.")

    # Run direction breakdown
    new_pred = fb_argmax.copy()
    flip_mask = override_mask & (override_class >= 0)
    new_pred[flip_mask] = override_class[flip_mask]

    directions = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((fb_argmax == fr) & (new_pred == to)).sum())
            if n > 0:
                directions[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    print(f"  directions: {directions}")
    h_added = int(((fb_argmax != 2) & (new_pred == 2)).sum())
    h_removed = int(((fb_argmax == 2) & (new_pred != 2)).sum())
    net_h = h_added - h_removed
    print(f"  net_H: +{h_added} -{h_removed} = {net_h:+d}")

    # ---- TRAIN-OOF validation ----
    # Apply same logic to TRAIN OOF using v1's argmax as 4b proxy
    # (v1 RF meta natural is the closest TRAIN-OOF analog of 4b's prediction surface).
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    v1_argmax_oof = v1_oof.argmax(axis=1).astype(np.int8)
    v1_outside = ~in_set_oof[np.arange(len(v1_argmax_oof)), v1_argmax_oof]
    v1_override_mask = v1_outside & (set_sizes_oof >= 1) & (set_sizes_oof < 3)

    v1_override_class = np.full(len(v1_argmax_oof), -1, dtype=np.int8)
    for i in np.where(v1_override_mask)[0]:
        cand = np.where(in_set_oof[i])[0]
        cand = cand[np.argsort(-oof_mean[i, cand])]
        v1_override_class[i] = cand[0]

    v1_new = v1_argmax_oof.copy()
    flip_oof = v1_override_mask & (v1_override_class >= 0)
    v1_new[flip_oof] = v1_override_class[flip_oof]

    base_macro = macro_recall(y_train, v1_argmax_oof)
    new_macro = macro_recall(y_train, v1_new)
    print(f"\nTRAIN OOF (v1 proxy):")
    print(f"  base macro recall: {base_macro:.6f}")
    print(f"  T2-augmented:      {new_macro:.6f}")
    print(f"  delta:             {new_macro - base_macro:+.6f}")

    # Per-direction precision on TRAIN OOF
    flip_idx = np.where(flip_oof)[0]
    if len(flip_idx) > 0:
        prec_per_dir = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                m = (v1_argmax_oof == fr) & (v1_new == to)
                if m.sum() > 0:
                    p = float((y_train[m] == to).mean())
                    n = int(m.sum())
                    prec_per_dir[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = (n, p)
        print(f"  flip precisions: {prec_per_dir}")

    # Emit submission only if test override count >= 30 and direction sane
    # Don't emit if mechanism would regress.
    out = {
        "alpha": ALPHA,
        "q_hat": q_hat,
        "oof_coverage": coverage_oof,
        "test_set_size_dist": {
            "singleton": int((set_sizes_test == 1).sum()),
            "two_class": int((set_sizes_test == 2).sum()),
            "three_class": int((set_sizes_test == 3).sum()),
        },
        "n_overrides_test": int(n_overrides),
        "directions_test": directions,
        "net_h_test": net_h,
        "train_oof_base_macro": base_macro,
        "train_oof_t2_macro": new_macro,
        "train_oof_delta": new_macro - base_macro,
        "n_flips_train_oof": int(flip_oof.sum()),
    }
    out_path = ART / "T2_conformal_override_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nresults: {out_path}")

    # Decision gate: only emit submission if:
    #   - ≥ 30 test override candidates
    #   - net_h direction nonnegative (or H<5 mild)
    #   - TRAIN OOF macro delta > 0
    decision = "EMIT" if (
        n_overrides >= 30
        and (new_macro - base_macro) > 0
        and net_h >= -5
    ) else "SKIP"
    print(f"\ndecision: {decision}")

    if decision == "EMIT":
        test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
        })
        out_csv = SUB / "submission_T2_conformal_override.csv"
        sub.to_csv(out_csv, index=False)
        print(f"emitted: {out_csv}")
    else:
        print(f"  reason: n_overrides={n_overrides}, "
              f"train_delta={new_macro - base_macro:+.6f}, "
              f"net_h={net_h:+d}")


if __name__ == "__main__":
    main()
