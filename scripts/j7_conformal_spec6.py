"""J7 — conformal-gated overrides on score=6 boundary.

The 2026-04-24 spec6_mh_v2 specialist had OOF AUC 0.938 but its
hard-threshold deploy peaked at +0.00009 OOF (prec 28% on 25 overrides).
Raw-prob threshold sweep is OOF-overfit-prone (one degree of freedom
selected to maximise OOF delta).

J7 replaces the raw-threshold gate with split-conformal calibration:
threshold derived from a coverage guarantee on the True-High class,
not from sweeping OOF deltas. Alpha is pre-specified (no selection bias).

Per-fold leak-safe Mondrian split-conformal:
  for fold k:
    cal = rows in fold f!=k AND override domain
    for each alpha:
      q = quantile(P_high[cal AND y=High], alpha)   # finite-sample-adjusted
      override fold-k rows in override-domain where P_high >= q

Test side: cal = full OOF override-domain rows; apply per-alpha threshold.

Compare resulting (overrides, precision, OOF macro-recall delta) vs
raw-threshold v2 baseline (peak +0.00009).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    add_distance_features, fast_bal_acc, log_blend, tune_log_bias, CLS2IDX,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)
SEED = 42
N_FOLDS = 5
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40
ALPHAS = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01]


def per_class_recall(y, pred, K=3):
    cc = np.bincount(y, minlength=K)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(K)], dtype=np.int64)
    return hit / np.maximum(cc, 1)


def conformal_threshold(p_high_cal_pos: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold for High coverage 1-alpha.

    Non-conformity score for True-High examples is s = 1 - P_high.
    Finite-sample quantile q = ceil((1-alpha)(n+1))/n -th order stat of s.
    Override rule: predict High if s_test <= q  ⟺  P_high >= 1 - q.
    Return the P_high cutoff (1 - q).
    """
    n = len(p_high_cal_pos)
    if n == 0:
        return 1.0  # never override
    s_cal = 1.0 - p_high_cal_pos
    rank = int(np.ceil((1.0 - alpha) * (n + 1)))
    rank = min(max(rank, 1), n)  # clamp to [1, n]
    q = np.partition(s_cal, rank - 1)[rank - 1]  # rank-th smallest
    return float(1.0 - q)


def main() -> None:
    print("=== J7 conformal-gated overrides on score=6 ===")

    # Teacher reconstruction (LB-best 3-way).
    print("loading teacher OOF + test")
    oofs = [np.load(ART / f"oof_{n}.npy") for n in
            ("recipe_full_te", "recipe_pseudolabel",
             "recipe_pseudolabel_seed7labeler")]
    tests = [np.load(ART / f"test_{n}.npy") for n in
             ("recipe_full_te", "recipe_pseudolabel",
              "recipe_pseudolabel_seed7labeler")]
    w = np.array([W_RECIPE, W_S1, W_S7])
    oof_t = log_blend(oofs, w)
    test_t = log_blend(tests, w)

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    bias, base_bal = tune_log_bias(oof_t, y, prior)
    base_rec = per_class_recall(
        y, (np.log(np.clip(oof_t, 1e-9, 1.0)) + bias).argmax(1))
    print(f"teacher OOF bal_acc={base_bal:.5f}  bias={bias.round(3).tolist()}")
    print(f"teacher per-class recall L/M/H = "
          f"{base_rec[0]:.4f} / {base_rec[1]:.4f} / {base_rec[2]:.4f}")

    eps = 1e-9
    teacher_pred_oof = (np.log(np.clip(oof_t, eps, 1.0)) + bias).argmax(1)
    teacher_pred_test = (np.log(np.clip(test_t, eps, 1.0)) + bias).argmax(1)

    tr_d = add_distance_features(tr)
    te_d = add_distance_features(te)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    mask_ovr_oof = (s_tr == 6) & (teacher_pred_oof == CLS2IDX["Medium"])
    mask_ovr_test = (s_te == 6) & (teacher_pred_test == CLS2IDX["Medium"])
    n_high_in_space = int((y[mask_ovr_oof] == CLS2IDX["High"]).sum())
    print(f"override space (OOF)  : {mask_ovr_oof.sum():,} "
          f"(truly-High: {n_high_in_space})")
    print(f"override space (test) : {mask_ovr_test.sum():,}")
    cc = np.bincount(y, minlength=3)
    break_even = cc[2] / (cc[1] + cc[2])
    print(f"break-even precision under macro-recall = {break_even:.4f}")

    # Spec6 v2 probs
    ph_oof = np.load(ART / "oof_spec6_mh_v2.npy")
    ph_test = np.load(ART / "test_spec6_mh_v2.npy")

    # Reproduce 5-fold split (must match spec6_mh_v2.py)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.full(len(y), -1, dtype=np.int32)
    for fid, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[va_idx] = fid
    assert (fold_id >= 0).all()

    # Per-fold leak-safe Mondrian conformal.
    print("\n=== per-alpha results (leak-safe per-fold Mondrian conformal) ===")
    print(f"{'alpha':>6} {'n_ovr':>7} {'ok':>5} {'bad':>5} {'prec':>7} "
          f"{'dRecH':>9} {'dRecM':>9} {'bal':>9} {'delta':>9}")
    rows = []
    for alpha in ALPHAS:
        flip = np.zeros(len(y), dtype=bool)
        for k in range(N_FOLDS):
            cal_mask = (fold_id != k) & mask_ovr_oof & (y == CLS2IDX["High"])
            theta_k = conformal_threshold(ph_oof[cal_mask], alpha)
            test_mask_k = (fold_id == k) & mask_ovr_oof
            flip |= test_mask_k & (ph_oof >= theta_k)
        n = int(flip.sum())
        ok = int((y[flip] == CLS2IDX["High"]).sum())
        bad = n - ok
        prec = ok / max(n, 1)
        new = teacher_pred_oof.copy()
        new[flip] = CLS2IDX["High"]
        bal = fast_bal_acc(y, new)
        rec = per_class_recall(y, new)
        delta = bal - base_bal
        d_rec = rec - base_rec
        rows.append(dict(alpha=alpha, n=n, correct=ok, wrong=bad,
                         precision=prec, d_rec_H=float(d_rec[2]),
                         d_rec_M=float(d_rec[1]), bal=float(bal),
                         delta=float(delta)))
        print(f"{alpha:>6.2f} {n:>7} {ok:>5} {bad:>5} {prec:>7.1%} "
              f"{d_rec[2]:>+9.5f} {d_rec[1]:>+9.5f} "
              f"{bal:>9.5f} {delta:>+9.5f}")

    # Identify best (positive-delta + per-class guardrail)
    GUARD = -5e-4
    safe = [r for r in rows
            if r["delta"] > 0 and r["d_rec_H"] >= GUARD
            and r["d_rec_M"] >= GUARD]
    print("\n=== best operating points ===")
    if safe:
        best_safe = max(safe, key=lambda r: r["delta"])
        print(f"safe best:  alpha={best_safe['alpha']}  delta="
              f"{best_safe['delta']:+.5f}  prec={best_safe['precision']:.1%}  "
              f"n={best_safe['n']}  dRecH={best_safe['d_rec_H']:+.5f}")
    else:
        best_safe = None
        print("safe best:  none (no alpha with positive delta + guardrail)")
    best_any = max(rows, key=lambda r: r["delta"]) if rows else None
    if best_any:
        print(f"any best:   alpha={best_any['alpha']}  delta="
              f"{best_any['delta']:+.5f}  prec={best_any['precision']:.1%}  "
              f"n={best_any['n']}  dRecH={best_any['d_rec_H']:+.5f}")

    # Emit submission only if a safe positive operating point exists
    sub_path = None
    if best_safe is not None and best_safe["delta"] > 5e-5:
        alpha = best_safe["alpha"]
        # Test-side calibration: use ALL OOF override-domain rows as cal
        cal_mask = mask_ovr_oof & (y == CLS2IDX["High"])
        theta = conformal_threshold(ph_oof[cal_mask], alpha)
        flip_test = mask_ovr_test & (ph_test >= theta)
        new_test = teacher_pred_test.copy()
        new_test[flip_test] = CLS2IDX["High"]
        cls_idx = {v: k for k, v in CLS2IDX.items()}
        sub = pd.DataFrame({
            "id": te["id"].to_numpy(),
            "Irrigation_Need": [cls_idx[i] for i in new_test],
        })
        tag = f"a{int(alpha*100):02d}"
        sub_path = SUB / f"submission_j7_conformal_spec6_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"\ntest-side: theta={theta:.4f}  overrides="
              f"{int(flip_test.sum()):,}")
        print(f"wrote {sub_path}")
        print(f"test dist: {dict(sub['Irrigation_Need'].value_counts())}")
    else:
        print("\nNo safe positive operating point — no submission emitted.")

    out = dict(
        teacher_bal_acc=float(base_bal),
        teacher_per_class_recall=base_rec.tolist(),
        break_even_precision=float(break_even),
        override_space_oof=int(mask_ovr_oof.sum()),
        override_space_test=int(mask_ovr_test.sum()),
        high_in_override_space=n_high_in_space,
        alpha_results=rows,
        best_safe=best_safe,
        best_any=best_any,
        submission=str(sub_path) if sub_path else None,
    )
    with open(ART / "j7_conformal_spec6_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote scripts/artifacts/j7_conformal_spec6_results.json")


if __name__ == "__main__":
    main()
