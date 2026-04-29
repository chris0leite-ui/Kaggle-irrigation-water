"""LB-consensus override analysis: where do strong LB-validated submissions agree
on a class different from v1 LB-best? Hard-vote override on those rows.

Inputs (LB-confirmed):
  v1 LB-best (LB 0.98129) — submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv
  rawashishsin v3 (LB 0.98109)
  tier1b_greedy_meta (LB 0.98094)
  recipe_full_te_catboost (LB 0.97935)

For each test row:
  - If v1 says class A and ALL helpers say class B (B ≠ A), override v1 → B
  - If 3/3 helpers agree class B and B ≠ v1's prediction A, override
  - Otherwise: keep v1

Diagnostic on TRAIN OOF: do these consensus-override rules actually IMPROVE
balanced accuracy on OOF? If yes (and macro-recall positive), worth LB probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)
    prior = np.bincount(y, minlength=3) / len(y)

    # Load OOF + test for each LB-validated candidate
    pool = {}
    for name, oof_p, lb in [
        ("v1", "oof_sklearn_rf_meta_natural_v1_lb98129.npy", 0.98129),
        ("raw", "oof_rawashishsin_2600.npy", 0.98109),
        ("tier1b", "oof_xgb_metastack.npy", 0.98094),  # NOTE: tier1b uses meta_iso through 4-stack
        ("cb", "oof_recipe_full_te_catboost.npy", 0.97935),
    ]:
        oof_path = ART / oof_p
        test_path = ART / oof_p.replace("oof_", "test_")
        if not oof_path.exists() or not test_path.exists():
            print(f"  SKIP {name}: missing")
            continue
        oof = _normed(np.load(oof_path).astype(np.float32))
        tst = _normed(np.load(test_path).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        pool[name] = (oof, tst, bias, tuned, lb)
        print(f"  {name} (LB {lb:.5f}): OOF tuned={tuned:.5f}  bias={bias.round(4).tolist()}")

    # OOF argmax for each (at its own tuned bias)
    oof_pred = {}
    test_pred = {}
    for name, (oof, tst, bias, _, _) in pool.items():
        oof_pred[name] = (safelog(oof) + bias).argmax(1)
        test_pred[name] = (safelog(tst) + bias).argmax(1)

    v1_pred = oof_pred["v1"]
    v1_test = test_pred["v1"]
    print(f"\nv1 OOF tuned bal_acc: {balanced_accuracy_score(y, v1_pred):.5f}")
    pcr_v1 = per_class_recall(y, v1_pred)
    print(f"v1 PCR=[L={pcr_v1[0]:.4f} M={pcr_v1[1]:.4f} H={pcr_v1[2]:.4f}]")

    # === RULE A: ALL helpers agree on a different class than v1 ===
    print("\n=== Rule A: ALL helpers agree (and disagree with v1) ===")
    helpers = [n for n in pool if n != "v1"]
    if len(helpers) >= 2:
        helper_preds = np.stack([oof_pred[h] for h in helpers], axis=1)
        helper_test = np.stack([test_pred[h] for h in helpers], axis=1)

        # All helpers agree
        all_agree_oof = (helper_preds == helper_preds[:, :1]).all(axis=1)
        all_agree_test = (helper_test == helper_test[:, :1]).all(axis=1)
        # AND consensus differs from v1
        consensus_oof = helper_preds[:, 0]
        consensus_test = helper_test[:, 0]
        switch_oof = all_agree_oof & (consensus_oof != v1_pred)
        switch_test = all_agree_test & (consensus_test != v1_test)

        n_oof_sw = switch_oof.sum()
        n_test_sw = switch_test.sum()
        print(f"  OOF switches: {n_oof_sw} / {n_tr}  ({n_oof_sw/n_tr*100:.3f}%)")
        print(f"  Test switches: {n_test_sw} / {n_te}  ({n_test_sw/n_te*100:.3f}%)")

        if n_oof_sw > 0:
            new_pred = v1_pred.copy()
            new_pred[switch_oof] = consensus_oof[switch_oof]
            bal = balanced_accuracy_score(y, new_pred)
            n_correct_sw = (consensus_oof[switch_oof] == y[switch_oof]).sum()
            n_v1_was_correct = (v1_pred[switch_oof] == y[switch_oof]).sum()
            prec = n_correct_sw / max(1, n_oof_sw)
            pcr = per_class_recall(y, new_pred)
            print(f"    OOF bal_acc after override: {bal:.5f}  (v1 was {balanced_accuracy_score(y, v1_pred):.5f})")
            print(f"    Δ = {bal - balanced_accuracy_score(y, v1_pred):+.5f}")
            print(f"    consensus correct on {n_correct_sw}/{n_oof_sw}; v1 was correct on {n_v1_was_correct}")
            print(f"    PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

            # Build test-side override
            new_test = v1_test.copy()
            new_test[switch_test] = consensus_test[switch_test]

            # Class breakdown of switches
            print(f"    test switches: v1→consensus class shift counts:")
            for v in [0, 1, 2]:
                for c in [0, 1, 2]:
                    if v != c:
                        n_vc = ((v1_test == v) & (consensus_test == c) & switch_test).sum()
                        if n_vc > 0:
                            print(f"      v1={IDX2CLS[v]:>6}, consensus={IDX2CLS[c]:>6}: {n_vc} rows")

            # Save submission if OOF lift
            if bal > balanced_accuracy_score(y, v1_pred):
                sub_path = SUB / "submission_consensus_override_all_helpers.csv"
                sub = pd.DataFrame({"id": test_ids,
                                    TARGET: [IDX2CLS[i] for i in new_test]})
                sub.to_csv(sub_path, index=False)
                print(f"    EMIT: {sub_path}")

    # === RULE B: Strict-2/3 majority vote (all 4 candidates) ===
    print("\n=== Rule B: Strict 2/3 majority across all 4 (v1 + 3 helpers) ===")
    if len(pool) >= 3:
        all_oof = np.stack([oof_pred[n] for n in pool], axis=1)
        all_test = np.stack([test_pred[n] for n in pool], axis=1)
        # Plurality vote with v1 as tiebreaker
        from scipy.stats import mode
        major_oof, _ = mode(all_oof, axis=1, keepdims=False)
        major_test, _ = mode(all_test, axis=1, keepdims=False)
        # Where majority differs from v1
        diff_oof = (major_oof != v1_pred).sum()
        diff_test = (major_test != v1_test).sum()
        print(f"  OOF rows where majority ≠ v1: {diff_oof} ({diff_oof/n_tr*100:.3f}%)")
        print(f"  Test rows where majority ≠ v1: {diff_test} ({diff_test/n_te*100:.3f}%)")
        bal = balanced_accuracy_score(y, major_oof)
        print(f"  Majority OOF bal_acc: {bal:.5f}")
        print(f"  Δ vs v1: {bal - balanced_accuracy_score(y, v1_pred):+.5f}")
        pcr = per_class_recall(y, major_oof)
        print(f"  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")


if __name__ == "__main__":
    main()
