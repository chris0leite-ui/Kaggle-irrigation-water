"""B — DGP-rule-anchored override precision audit on TRAIN OOF.

Mechanism (orthogonal to all 41 prior saturations):
  Use the closed-form DGP rule (~98.4% global accuracy) as the override
  AUTHORITY for rows where 4b is uncertain. Prior overrides used 14-bank
  majority or LLM as authority; this uses the rule itself.

Fire condition (per row):
  - rule_pred != 4b_pred                                        (disagreement)
  - rule_score in extreme-precision bin                         (rule-conf proxy)
  - bank_mean_max_prob < tau                                    (4b-conf proxy)
  - bank_argmax == rule_pred                                    (bank confirms rule)
  -> override 4b_pred -> rule_pred

This script only AUDITS precision on TRAIN OOF for various filters; it
does NOT emit a submission. We need to clear the 92% break-even on
test (after applying the T6-documented 15-20pp TRAIN-OOF -> test
haircut, since v1 is in the bank).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402
from T6_diversity_helpers import (  # noqa: E402
    load_y_train,
    macro_recall,
    normed,
    tune_log_bias_simple,
)
from dgp_formula import dgp_score  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")


def build_4b_oof_analog(y: np.ndarray) -> np.ndarray:
    """Replicate T1/T6 reconstruction of 4b OOF argmax."""
    v1 = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1 = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    bv1, _ = tune_log_bias_simple(v1, y)
    bra, _ = tune_log_bias_simple(raw, y)
    bt1, _ = tune_log_bias_simple(t1, y)
    a_v1 = (np.log(np.clip(v1, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1, 1e-9, None)) + bt1).argmax(1).astype(np.int8)
    una = a_ra == a_t1
    fb = a_v1.copy()
    om = una & (a_v1 != a_ra)
    fb[om] = a_ra[om]
    return fb


def main() -> None:
    print("=== B: DGP-rule-anchored override precision audit ===\n")
    train = pd.read_csv(DATA / "train.csv")
    y = load_y_train()
    print(f"TRAIN rows: {len(y)}, dist L={(y==0).sum()} M={(y==1).sum()} H={(y==2).sum()}")

    # Score under the closed-form DGP rule
    score = dgp_score(train).astype(np.int16)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    rule_acc = float((rule_pred == y).mean())
    print(f"\nGlobal rule accuracy on TRAIN: {rule_acc:.6f}")
    print(f"Rule score distribution:")
    for s in range(int(score.min()), int(score.max()) + 1):
        m = score == s
        if m.sum() == 0:
            continue
        sub_acc = (rule_pred[m] == y[m]).mean() if m.sum() else 0.0
        print(f"  score={s}: n={int(m.sum()):>7d}  rule-acc={sub_acc:.5f}")

    # 4b OOF analog
    fb_oof = build_4b_oof_analog(y)
    print(f"\n4b OOF analog macro: {macro_recall(y, fb_oof):.6f}")

    # 14-bank mean probs on OOF
    bank = load_bank("oof")
    bm = bank_mean_probs(bank)
    bank_argmax = bm.argmax(axis=1).astype(np.int8)
    bank_max = bm.max(axis=1)

    print(f"\nBank-argmax dist: L={(bank_argmax==0).sum()} M={(bank_argmax==1).sum()} H={(bank_argmax==2).sum()}")
    print(f"Bank max-prob: median={np.median(bank_max):.4f} q05={np.quantile(bank_max,0.05):.4f}")

    # ------- Candidate override fire conditions -------
    diff = rule_pred != fb_oof
    bank_confirms = bank_argmax == rule_pred

    print("\n--- Filter precision audit ---")
    print("Reading: 'precision' = P(y == rule_pred | filter)")
    print("Break-even ~92%; apply T6 ~15-20pp TRAIN-OOF haircut for test projection.\n")

    results = {"global_rule_acc": rule_acc, "filters": []}

    # By rule-score band, restricted to 'rule != 4b' subset
    score_bands = [
        ("low_extreme", lambda s: s <= 1),
        ("low_safe", lambda s: s <= 2),
        ("low_band", lambda s: s == 3),
        ("mid_band", lambda s: (s >= 4) & (s <= 6)),
        ("high_band", lambda s: s == 7),
        ("high_safe", lambda s: s >= 8),
        ("high_extreme", lambda s: s >= 9),
    ]
    for name, fn in score_bands:
        mask = diff & fn(score)
        n = int(mask.sum())
        if n == 0:
            continue
        prec = float((rule_pred[mask] == y[mask]).mean())
        # Also conditional on bank confirming rule
        m2 = mask & bank_confirms
        n2 = int(m2.sum())
        prec2 = float((rule_pred[m2] == y[m2]).mean()) if n2 else 0.0
        # Direction breakdown
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                mm = mask & (fb_oof == fr) & (rule_pred == to)
                if mm.sum() == 0:
                    continue
                dirs[f"{fr}->{to}"] = {
                    "n": int(mm.sum()),
                    "prec": float((rule_pred[mm] == y[mm]).mean()),
                }
        print(f"[{name}] rule!=4b  n={n:>7d}  prec={prec:.4f}   "
              f"bank-confirms n={n2:>6d}  prec={prec2:.4f}")
        for k, v in dirs.items():
            print(f"   dir {k}  n={v['n']:>6d}  prec={v['prec']:.4f}")
        results["filters"].append({
            "band": name,
            "n_diff": n, "prec_diff": prec,
            "n_bank_confirms": n2, "prec_bank_confirms": prec2,
            "directions": dirs,
        })

    # Sweep tau on bank_max for rule-extreme bands
    print("\n--- Bank-max threshold sweep (rule-extreme + bank-confirms only) ---")
    tau_grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    extreme = (score <= 1) | (score >= 9)
    base_mask = diff & bank_confirms & extreme
    print(f"Extreme + diff + bank-confirms total: {int(base_mask.sum())}")
    sweep = []
    for tau in tau_grid:
        m = base_mask & (bank_max < tau)
        n = int(m.sum())
        if n == 0:
            continue
        prec = float((rule_pred[m] == y[m]).mean())
        # Macro delta if applied
        new_pred = fb_oof.copy()
        new_pred[m] = rule_pred[m]
        new_macro = macro_recall(y, new_pred)
        delta = new_macro - macro_recall(y, fb_oof)
        print(f"  tau<{tau:.2f}  n={n:>5d}  prec={prec:.4f}  oof-delta={delta:+.6f}")
        sweep.append({"tau": tau, "n": n, "prec": prec, "oof_delta": delta})
    results["bank_max_sweep_extreme"] = sweep

    # Also sweep on the safe (s<=2 or s>=8) band
    print("\n--- Bank-max sweep (safe band s<=2 or s>=8 + bank-confirms) ---")
    safe = (score <= 2) | (score >= 8)
    base2 = diff & bank_confirms & safe
    print(f"Safe + diff + bank-confirms total: {int(base2.sum())}")
    sweep2 = []
    for tau in tau_grid:
        m = base2 & (bank_max < tau)
        n = int(m.sum())
        if n == 0:
            continue
        prec = float((rule_pred[m] == y[m]).mean())
        new_pred = fb_oof.copy()
        new_pred[m] = rule_pred[m]
        new_macro = macro_recall(y, new_pred)
        delta = new_macro - macro_recall(y, fb_oof)
        print(f"  tau<{tau:.2f}  n={n:>5d}  prec={prec:.4f}  oof-delta={delta:+.6f}")
        sweep2.append({"tau": tau, "n": n, "prec": prec, "oof_delta": delta})
    results["bank_max_sweep_safe"] = sweep2

    out = ART / "B_rule_anchor_precision_audit_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
