"""#12 Direction-stratified override on top of LB-best 0.98140 winner.

Mechanism context:
  LB-best 0.98140 winner = layered (v1 anchor → 0.98134 → 0.98140) with
  286 test rows differing from v1's argmax. Per-direction breakdown:
    L→M:   8   (winner overrode v1's Low to Medium)
    M→L:  95
    M→H: 134
    H→M:  49

Per-direction OOF precision (single-stage k=2 unanimous analog, computed
on OOF using v1 anchor + raw/tier1b OTHERS):
    L→M: 0.25  (BE 0.39, MARGIN −0.14) → REVERT
    M→L: 0.57  (BE 0.61, MARGIN −0.03) → REVERT
    M→H: 0.18  (BE 0.08, MARGIN +0.10) → KEEP
    H→M: 0.96  (BE 0.92, MARGIN +0.04) → KEEP

Strategy:
  Build candidates by REVERTING weak-direction overrides from winner CSV
  back to v1's prediction. Two variants:
    REVERT_WEAK   - revert L→M + M→L (margin ≤ 0)
    REVERT_M_TO_L - revert only M→L (the highest-volume weak direction)

Per CLAUDE.md submission rule: emit candidates only; user approves probe.
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

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLS)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def _normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def break_even_precision(prior, anchor_class, override_class):
    return prior[override_class] / (prior[anchor_class] + prior[override_class])


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # ===== Load anchor + 2 OTHERS for OOF-side direction precision =====
    components = {
        "v1":  ("oof_sklearn_rf_meta_natural_v1_lb98129.npy",
                "test_sklearn_rf_meta_natural_v1_lb98129.npy", 0.98129),
        "raw": ("oof_rawashishsin_2600.npy",
                "test_rawashishsin_2600.npy", 0.98109),
        "t1b": ("oof_tier1b_greedy_meta.npy",
                "test_tier1b_greedy_meta.npy", 0.98094),
    }
    pool = {}
    for name, (oof_p, test_p, lb) in components.items():
        oof = _normed(np.load(ART / oof_p).astype(np.float32))
        tst = _normed(np.load(ART / test_p).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        pool[name] = dict(
            oof=oof, test=tst, bias=bias, tuned=tuned, lb=lb,
            oof_argmax=(np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1),
            test_argmax=(np.log(np.clip(tst, 1e-9, 1.0)) + bias).argmax(1),
        )
        print(f"  {name} (LB {lb:.5f}): tuned OOF {tuned:.5f}")

    # ===== Compute per-direction OOF precision for k=2 unanimous override =====
    v1_oof = pool["v1"]["oof_argmax"]
    raw_oof = pool["raw"]["oof_argmax"]
    t1b_oof = pool["t1b"]["oof_argmax"]
    consensus_oof = raw_oof
    unanimous_oof = (raw_oof == t1b_oof) & (consensus_oof != v1_oof)

    print(f"\nOOF k=2 unanimous candidates: {unanimous_oof.sum()}")
    print(f"\n=== Per-direction OOF precision (single-stage k=2 unanimous) ===")
    print(f"{'A':<7}{'C':<7}{'n':>6}{'prec':>8}{'BE':>8}{'margin':>9}{'verdict':>10}")
    direction_stats = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            mask = unanimous_oof & (v1_oof == a) & (consensus_oof == c)
            n = mask.sum()
            if n == 0: continue
            n_correct = (y[mask] == c).sum()
            prec = n_correct / n
            be = break_even_precision(prior, a, c)
            margin = prec - be
            verdict = "KEEP" if margin > 0 else "REVERT"
            direction_stats[(a, c)] = dict(
                n=int(n), prec=float(prec), be=float(be),
                margin=float(margin), verdict=verdict)
            print(f"{IDX2CLS[a]:<7}{IDX2CLS[c]:<7}{n:>6}{prec:>8.4f}"
                  f"{be:>8.4f}{margin:>+9.4f}{verdict:>10}")

    revert_dirs = {(a, c) for (a, c), s in direction_stats.items() if s["margin"] <= 0}
    keep_dirs = {(a, c) for (a, c), s in direction_stats.items() if s["margin"] > 0}
    print(f"\nREVERT directions (margin ≤ 0): "
          f"{[(IDX2CLS[a], IDX2CLS[c]) for a, c in revert_dirs]}")
    print(f"KEEP directions (margin > 0):    "
          f"{[(IDX2CLS[a], IDX2CLS[c]) for a, c in keep_dirs]}")

    # ===== Load LB-best winner submission as test-side anchor =====
    winner_csv = SUB / "submission_2other_raw_tier1b_k2.csv"
    winner = pd.read_csv(winner_csv)
    winner_pred = winner[TARGET].map(CLS2IDX).to_numpy()
    v1_test = pool["v1"]["test_argmax"]

    # Verify v1's submission alignment
    v1_csv = SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv"
    v1_csv_pred = pd.read_csv(v1_csv)[TARGET].map(CLS2IDX).to_numpy()
    if (v1_csv_pred != v1_test).any():
        print(f"\nNOTE: v1 OOF argmax differs from v1 CSV by "
              f"{(v1_csv_pred != v1_test).sum()} rows; using v1 CSV as test anchor.")
        v1_test = v1_csv_pred  # winner was built on v1 CSV, not OOF argmax

    # Test-side: where winner differs from v1 (these are the overrides)
    diff_mask = winner_pred != v1_test
    print(f"\nWinner vs v1 test diffs: {diff_mask.sum()}")
    print("Direction breakdown of winner overrides on test:")
    test_dir_counts = {}
    for a in range(3):
        for c in range(3):
            if a == c: continue
            cnt = ((v1_test == a) & (winner_pred == c) & diff_mask).sum()
            if cnt:
                test_dir_counts[(a, c)] = int(cnt)
                print(f"  {IDX2CLS[a]:<7}→{IDX2CLS[c]:<7}: {cnt}")

    # ===== Build candidates =====
    print("\n=== Building candidates ===")

    def emit(label, revert_set):
        new_pred = winner_pred.copy()
        n_reverted = 0
        for (a, c) in revert_set:
            m = (v1_test == a) & (winner_pred == c) & diff_mask
            new_pred[m] = a  # revert: winner had c, restore v1's a
            n_reverted += int(m.sum())
        diff_winner = int((new_pred != winner_pred).sum())
        path = SUB / f"submission_n12_dirstrat_{label}.csv"
        pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in new_pred],
        }).to_csv(path, index=False)
        return path, n_reverted, diff_winner, new_pred

    p_w, n_w, d_w, _ = emit("REVERT_WEAK",
                            revert_set=revert_dirs)
    p_ml, n_ml, d_ml, _ = emit("REVERT_ML_ONLY",
                               revert_set={(1, 0)} & revert_dirs)
    p_lm, n_lm, d_lm, _ = emit("REVERT_LM_ONLY",
                               revert_set={(0, 1)} & revert_dirs)

    # Sanity: REVERT all -> should equal v1
    p_all, n_all, d_all, _ = emit("REVERT_ALL",
                                  revert_set={(a, c) for a in range(3)
                                              for c in range(3) if a != c})
    print(f"\n  REVERT_WEAK    -> {p_w.name}  reverted={n_w}  diff vs winner={d_w}")
    print(f"  REVERT_ML_ONLY -> {p_ml.name}  reverted={n_ml}  diff vs winner={d_ml}")
    print(f"  REVERT_LM_ONLY -> {p_lm.name}  reverted={n_lm}  diff vs winner={d_lm}")
    print(f"  REVERT_ALL     -> {p_all.name}  (sanity, =v1)  reverted={n_all}  diff={d_all}")

    # ===== Estimated LB delta from reverting weak directions =====
    # Per direction: each revert moves 1 prediction from C→A
    # If true label was A: gain on rec_A (winner was wrong)
    # If true label was C: loss on rec_C (winner was right)
    # Use direction precision from OOF as estimate of P(true=C | override)
    print("\n=== Estimated LB delta from REVERT_WEAK (using OOF precision proxy) ===")
    n_class = np.bincount(y, minlength=3)
    macro_delta = 0.0
    for (a, c) in revert_dirs:
        if (a, c) not in test_dir_counts: continue
        k = test_dir_counts[(a, c)]
        prec = direction_stats[(a, c)]["prec"]
        # On test: assume same precision applies
        # n_correct_reverted = (1-prec) * k → rows where revert is right (gain rec_A)
        # n_wrong_reverted = prec * k → rows where revert is wrong (lose rec_C)
        gain_a = (1 - prec) * k / (3 * n_class[a])
        loss_c = prec * k / (3 * n_class[c])
        net = gain_a - loss_c
        macro_delta += net
        print(f"  {IDX2CLS[a]:<7}→{IDX2CLS[c]:<7}: k={k}, prec={prec:.3f}, "
              f"net={net:+.6e}")
    print(f"  TOTAL ESTIMATED Δ macro-recall: {macro_delta:+.6f}")
    print(f"  Projected LB: 0.98140 + {macro_delta:+.6f} = {0.98140 + macro_delta:.5f}")

    # Save summary
    summary = {
        "anchor": "submission_2other_raw_tier1b_k2.csv (LB 0.98140)",
        "v1_test_anchor": "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv",
        "winner_v1_test_diff": int(diff_mask.sum()),
        "test_direction_counts": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": v
                                  for (a, c), v in test_dir_counts.items()},
        "oof_direction_stats": {f"{IDX2CLS[a]}->{IDX2CLS[c]}": s
                                for (a, c), s in direction_stats.items()},
        "revert_dirs": [f"{IDX2CLS[a]}->{IDX2CLS[c]}" for a, c in revert_dirs],
        "keep_dirs": [f"{IDX2CLS[a]}->{IDX2CLS[c]}" for a, c in keep_dirs],
        "n_reverted_per_variant": {
            "REVERT_WEAK": n_w, "REVERT_ML_ONLY": n_ml,
            "REVERT_LM_ONLY": n_lm, "REVERT_ALL": n_all,
        },
        "estimated_macro_delta_REVERT_WEAK": float(macro_delta),
        "projected_LB_REVERT_WEAK": 0.98140 + float(macro_delta),
    }
    json_path = ART / "n12_direction_strat_override_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
