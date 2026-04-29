"""3-way agreement gating: v1 (LB 0.98129) × a1lgbm (LB 0.98097) × primary (LB 0.98094).

Mechanism:
  For each row:
    if v1.argmax == a1lgbm.argmax: use v1.argmax (no disagreement, ~99.91% of rows)
    else (243 test rows / N OOF rows where v1 ≠ a1lgbm):
      if primary.argmax == v1.argmax: use v1.argmax (primary endorses v1)
      elif primary.argmax == a1lgbm.argmax: use a1lgbm.argmax (primary endorses a1lgbm)
      else: use v1.argmax (LB-best fallback)

OOF diagnostic FIRST: on training OOF rows where v1 ≠ a1lgbm, what's the
true-label distribution and what does each model get right?
  - If primary's tiebreaker correctness on those rows > 50%, gating > pure v1
  - If primary's tiebreaker correctness on those rows < 50%, gating < pure v1
  - The ratio of tiebreaker accuracy determines expected LB delta

Hypothesis: primary uses ~88 Jaccard-shared error rows with v1 but is
INDEPENDENTLY trained (recipe family vs natural-cal family). On the
243 disagreement rows, primary's argmax is an independent third opinion.

CLAUDE.md PCR analysis:
  v1     PCR=[L=0.9946, M=0.9694, H=0.9779]
  a1lgbm PCR=[L=0.9949, M=0.9672, H=0.9803]   ← +0.0024 H, -0.0022 M
  prim   PCR=[L=0.9951, M=0.9702, H=0.9775]   ← matches v1 on H, more M

Disagreement rows are where a1lgbm pushed H but v1 stayed at M (or vice versa).
Primary is structurally CLOSER to v1 on H but has more M precision.
Predicted: primary will side with v1 on most disagreements, gating ~= pure v1.

Output: standalone candidate CSV + diagnostic JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from tier1b_helpers import build_lbbest_stack, iso_cal  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    print("=== Loading components ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # v1 RF natural (LB 0.98129)
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32)

    # a1lgbm RF natural (LB 0.98097)
    a1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_a1lgbm.npy").astype(np.float32)
    a1_test = np.load(ART / "test_sklearn_rf_meta_natural_a1lgbm.npy").astype(np.float32)

    # LB-best 4-stack primary (LB 0.98094)
    lb3_oof, lb3_test = build_lbbest_stack(y)
    meta_o = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    meta_t = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    eps = 1e-9
    primary_oof = np.exp(0.7 * safelog(lb3_oof) + 0.3 * safelog(meta_o_iso))
    primary_oof /= primary_oof.sum(1, keepdims=True)
    primary_test = np.exp(0.7 * safelog(lb3_test) + 0.3 * safelog(meta_t_iso))
    primary_test /= primary_test.sum(1, keepdims=True)

    # Tune log-bias for each
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    a1_bias, a1_tuned = tune_log_bias(a1_oof, y, prior)
    pr_bias, pr_tuned = tune_log_bias(primary_oof, y, prior)

    print(f"v1 RF natural   : OOF tuned {v1_tuned:.5f}  bias {v1_bias.round(3).tolist()}")
    print(f"a1lgbm RF natural: OOF tuned {a1_tuned:.5f}  bias {a1_bias.round(3).tolist()}")
    print(f"LB-best primary : OOF tuned {pr_tuned:.5f}  bias {pr_bias.round(3).tolist()}")

    # Argmax under tuned bias
    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    a1_pred_oof = (safelog(a1_oof) + a1_bias).argmax(1)
    pr_pred_oof = (safelog(primary_oof) + pr_bias).argmax(1)
    v1_pred_test = (safelog(v1_test) + v1_bias).argmax(1)
    a1_pred_test = (safelog(a1_test) + a1_bias).argmax(1)
    pr_pred_test = (safelog(primary_test) + pr_bias).argmax(1)

    # === OOF diagnostic on disagreement rows ===
    disagree_oof = v1_pred_oof != a1_pred_oof
    n_dis_oof = disagree_oof.sum()
    print(f"\n=== OOF disagreement rows: {n_dis_oof}/{len(y)} ({100*n_dis_oof/len(y):.3f}%) ===")
    print("Per-model accuracy on disagreement rows:")
    print(f"  v1    correct: {(v1_pred_oof[disagree_oof] == y[disagree_oof]).sum():>5} / {n_dis_oof}  ({100*(v1_pred_oof[disagree_oof] == y[disagree_oof]).mean():.2f}%)")
    print(f"  a1lgbm correct: {(a1_pred_oof[disagree_oof] == y[disagree_oof]).sum():>5} / {n_dis_oof}  ({100*(a1_pred_oof[disagree_oof] == y[disagree_oof]).mean():.2f}%)")
    print(f"  prim  correct: {(pr_pred_oof[disagree_oof] == y[disagree_oof]).sum():>5} / {n_dis_oof}  ({100*(pr_pred_oof[disagree_oof] == y[disagree_oof]).mean():.2f}%)")

    # Tiebreaker behavior
    pr_with_v1 = (pr_pred_oof[disagree_oof] == v1_pred_oof[disagree_oof])
    pr_with_a1 = (pr_pred_oof[disagree_oof] == a1_pred_oof[disagree_oof])
    pr_neither = ~(pr_with_v1 | pr_with_a1)
    print(f"\nPrimary's tiebreaker behavior on {n_dis_oof} disagreement rows:")
    print(f"  primary endorses v1   : {pr_with_v1.sum():>5} ({100*pr_with_v1.mean():.2f}%)")
    print(f"  primary endorses a1lgbm: {pr_with_a1.sum():>5} ({100*pr_with_a1.mean():.2f}%)")
    print(f"  primary disagrees both: {pr_neither.sum():>5} ({100*pr_neither.mean():.2f}%)")

    # === Apply gate to OOF and measure macro-recall ===
    gated_pred_oof = v1_pred_oof.copy()
    # On disagreement rows: if primary endorses a1lgbm, switch to a1lgbm
    switch_mask_oof = disagree_oof & (pr_pred_oof == a1_pred_oof)
    gated_pred_oof[switch_mask_oof] = a1_pred_oof[switch_mask_oof]
    n_switched_oof = switch_mask_oof.sum()
    print(f"\nGate switches v1→a1lgbm on {n_switched_oof}/{n_dis_oof} OOF rows")

    # Diagnose those switches
    if n_switched_oof > 0:
        switched_correct_a1 = (a1_pred_oof[switch_mask_oof] == y[switch_mask_oof]).sum()
        switched_correct_v1 = (v1_pred_oof[switch_mask_oof] == y[switch_mask_oof]).sum()
        print(f"  Among switched: a1lgbm correct {switched_correct_a1}, v1 correct {switched_correct_v1}, net flip win = {switched_correct_a1 - switched_correct_v1}")

    # OOF macro-recall comparison
    v1_oof_bal = balanced_accuracy_score(y, v1_pred_oof)
    a1_oof_bal = balanced_accuracy_score(y, a1_pred_oof)
    pr_oof_bal = balanced_accuracy_score(y, pr_pred_oof)
    gated_oof_bal = balanced_accuracy_score(y, gated_pred_oof)
    print(f"\nOOF tuned bal_acc (each at own tuned bias):")
    print(f"  v1                {v1_oof_bal:.5f}  (LB 0.98129)")
    print(f"  a1lgbm            {a1_oof_bal:.5f}  (LB 0.98097)")
    print(f"  primary           {pr_oof_bal:.5f}  (LB 0.98094)")
    print(f"  3-way gated       {gated_oof_bal:.5f}  Δ vs v1 = {gated_oof_bal - v1_oof_bal:+.5f}")

    # PCR for gated
    v1_pcr = per_class_recall(y, v1_pred_oof)
    a1_pcr = per_class_recall(y, a1_pred_oof)
    g_pcr = per_class_recall(y, gated_pred_oof)
    print(f"\nPer-class recall delta (gated - v1):")
    print(f"  Low  : {g_pcr[0] - v1_pcr[0]:+.5f}  (a1lgbm: {a1_pcr[0] - v1_pcr[0]:+.5f})")
    print(f"  Med  : {g_pcr[1] - v1_pcr[1]:+.5f}  (a1lgbm: {a1_pcr[1] - v1_pcr[1]:+.5f})")
    print(f"  High : {g_pcr[2] - v1_pcr[2]:+.5f}  (a1lgbm: {a1_pcr[2] - v1_pcr[2]:+.5f})")

    # === Apply gate to TEST and emit candidate ===
    gated_pred_test = v1_pred_test.copy()
    switch_mask_test = (v1_pred_test != a1_pred_test) & (pr_pred_test == a1_pred_test)
    gated_pred_test[switch_mask_test] = a1_pred_test[switch_mask_test]
    n_disagree_test = (v1_pred_test != a1_pred_test).sum()
    n_switched_test = switch_mask_test.sum()
    print(f"\nTest disagreement rows: {n_disagree_test}/{len(test_ids)}")
    print(f"Gate switches on test:  {n_switched_test}")
    print(f"Test class shift vs v1: net_H = "
          f"{((gated_pred_test == 2).sum() - (v1_pred_test == 2).sum()):+d}")

    # Compare gated_pred_test to v1 standalone
    diff_vs_v1 = (gated_pred_test != v1_pred_test).sum()
    print(f"\nGated candidate vs v1 (LB 0.98129 standalone): {diff_vs_v1} test rows differ")

    sub_path = SUB / "submission_3way_gate_v1_a1_primary.csv"
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in gated_pred_test]})
    sub.to_csv(sub_path, index=False)
    print(f"\nWrote {sub_path}")

    # Save diagnostic JSON
    summary = dict(
        v1_oof_tuned=float(v1_tuned),
        a1lgbm_oof_tuned=float(a1_tuned),
        primary_oof_tuned=float(pr_tuned),
        gated_oof_tuned=float(gated_oof_bal),
        gated_minus_v1=float(gated_oof_bal - v1_oof_bal),
        n_oof_disagreements=int(n_dis_oof),
        n_oof_switches=int(n_switched_oof),
        switched_a1_correct=int((a1_pred_oof[switch_mask_oof] == y[switch_mask_oof]).sum()) if n_switched_oof > 0 else 0,
        switched_v1_correct=int((v1_pred_oof[switch_mask_oof] == y[switch_mask_oof]).sum()) if n_switched_oof > 0 else 0,
        n_test_disagreements=int(n_disagree_test),
        n_test_switches=int(n_switched_test),
        test_diff_vs_v1=int(diff_vs_v1),
        v1_pcr=v1_pcr.tolist(),
        a1lgbm_pcr=a1_pcr.tolist(),
        gated_pcr=g_pcr.tolist(),
        sub_path=str(sub_path),
    )
    out_p = ART / "3way_gate_v1_a1_primary_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
