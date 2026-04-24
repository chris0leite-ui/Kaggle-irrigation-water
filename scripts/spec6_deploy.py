"""Hard-override deploy of spec6 binary specialist on top of LB-best 3-way.

Mechanism: on score=6 rows where teacher_argmax == Medium AND
P_spec_high > theta, override to High.

Under macro-recall, the break-even precision is
    21009 / 239074 = 8.8 %   (High_count / Medium_count).
Every correct override adds 1/21009 High recall; every wrong override
subtracts 1/239074 Medium recall. Argmax-invariant for Low.

We sweep theta on OOF, compute:
  - rows overridden
  - correct (truly-High) / wrong (truly-Medium) counts
  - per-class recall deltas
  - overall bal_acc delta vs teacher

Also tests the "conditional on teacher confidence" variant:
  override only if P_spec > theta AND teacher's P(High) > theta_tH.
  Tighter precision at the cost of recall on the detector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    add_distance_features, fast_bal_acc, log_blend, tune_log_bias, CLS2IDX,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)

# LB-best 3-way teacher components and their log-blend weights.
# From CLAUDE.md 2026-04-24:
#   3-way: recipe 0.25 + pseudo_s1 0.35 + pseudo_s7 0.40  (LB 0.98005)
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40


def build_teacher() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (oof_teacher_raw, test_teacher_raw, teacher_bias)."""
    oofs = [
        np.load(ART / "oof_recipe_full_te.npy"),
        np.load(ART / "oof_recipe_pseudolabel.npy"),
        np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"),
    ]
    tests = [
        np.load(ART / "test_recipe_full_te.npy"),
        np.load(ART / "test_recipe_pseudolabel.npy"),
        np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"),
    ]
    w = np.array([W_RECIPE, W_S1, W_S7], dtype=np.float64)
    oof_t = log_blend(oofs, w)
    test_t = log_blend(tests, w)

    # Tune log-bias on teacher OOF (should reproduce ~0.98029)
    tr = pd.read_csv("data/train.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof_t, y, prior)
    print(f"teacher tuned bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")
    return oof_t, test_t, bias


def per_class_recall(y: np.ndarray, pred: np.ndarray, K: int = 3) -> np.ndarray:
    cc = np.bincount(y, minlength=K)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(K)], dtype=np.int64)
    return hit / np.maximum(cc, 1)


def main() -> None:
    print("=== building LB-best 3-way teacher ===")
    oof_t, test_t, bias = build_teacher()

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    tr_dist = add_distance_features(tr)
    te_dist = add_distance_features(te)
    s_tr = tr_dist["dgp_score"].to_numpy()
    s_te = te_dist["dgp_score"].to_numpy()

    # Teacher argmax at fixed bias
    eps = 1e-9
    log_oof = np.log(np.clip(oof_t, eps, 1.0))
    log_test = np.log(np.clip(test_t, eps, 1.0))
    teacher_pred_oof = (log_oof + bias).argmax(1)
    teacher_pred_test = (log_test + bias).argmax(1)

    base_bal = fast_bal_acc(y, teacher_pred_oof)
    base_recall = per_class_recall(y, teacher_pred_oof)
    print(f"\nteacher OOF bal_acc = {base_bal:.5f}")
    print(f"teacher per-class recall L/M/H = "
          f"{base_recall[0]:.4f} / {base_recall[1]:.4f} / {base_recall[2]:.4f}")

    # Specialist probs
    ph_oof = np.load(ART / "oof_spec6_mh.npy")  # (630k,) float32
    ph_test = np.load(ART / "test_spec6_mh.npy")  # (270k,)
    assert ph_oof.shape == (len(tr),), ph_oof.shape
    assert ph_test.shape == (len(te),), ph_test.shape

    mask_tr_s6 = s_tr == 6
    mask_te_s6 = s_te == 6
    # "Override space": teacher-predicted-Medium rows on score=6
    mask_override_oof = mask_tr_s6 & (teacher_pred_oof == CLS2IDX["Medium"])
    print(f"\noverride space (OOF)  : score=6 ∩ teacher_pred=Medium = "
          f"{mask_override_oof.sum():,}")
    n_high_in_space = int((y[mask_override_oof] == CLS2IDX["High"]).sum())
    print(f"  truly-High in space : {n_high_in_space:,}  "
          f"(max possible +correct)")
    print(f"  truly-Med  in space : {mask_override_oof.sum() - n_high_in_space:,}")

    # Theta sweep
    print("\n=== theta sweep (override Medium -> High on score=6) ===")
    print(f"{'theta':>6} {'n_ovr':>7} {'correct':>7} {'wrong':>7} "
          f"{'prec':>6} {'d_recL':>8} {'d_recM':>8} {'d_recH':>8} "
          f"{'bal_acc':>8} {'delta':>8}")

    results = []
    thetas = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
              0.80, 0.85, 0.90, 0.95]
    for th in thetas:
        flip_mask = mask_override_oof & (ph_oof > th)
        n_ovr = int(flip_mask.sum())
        correct = int((y[flip_mask] == CLS2IDX["High"]).sum())
        wrong = n_ovr - correct
        prec = correct / max(n_ovr, 1)

        new_pred = teacher_pred_oof.copy()
        new_pred[flip_mask] = CLS2IDX["High"]
        new_bal = fast_bal_acc(y, new_pred)
        new_rec = per_class_recall(y, new_pred)

        d_rec = new_rec - base_recall
        delta = new_bal - base_bal
        results.append(dict(
            theta=th, n_ovr=n_ovr, correct=correct, wrong=wrong,
            precision=prec, d_rec_L=float(d_rec[0]),
            d_rec_M=float(d_rec[1]), d_rec_H=float(d_rec[2]),
            bal_acc=float(new_bal), delta=float(delta),
        ))
        print(f"{th:>6.2f} {n_ovr:>7} {correct:>7} {wrong:>7} "
              f"{prec:>5.1%} {d_rec[0]:>+8.5f} {d_rec[1]:>+8.5f} "
              f"{d_rec[2]:>+8.5f} {new_bal:>8.5f} {delta:>+8.5f}")

    # Also diagnostic: variant that always fires when spec>theta, even if
    # teacher says High already (to isolate "teacher-correct" interference).
    print("\n=== alternative: fire on ANY score=6 row, teacher_pred not checked ===")
    for th in (0.50, 0.70, 0.90):
        flip_mask = mask_tr_s6 & (ph_oof > th) & (teacher_pred_oof != CLS2IDX["High"])
        n_ovr = int(flip_mask.sum())
        correct = int((y[flip_mask] == CLS2IDX["High"]).sum())
        wrong = n_ovr - correct
        prec = correct / max(n_ovr, 1)
        new_pred = teacher_pred_oof.copy()
        new_pred[flip_mask] = CLS2IDX["High"]
        new_bal = fast_bal_acc(y, new_pred)
        delta = new_bal - base_bal
        print(f"  th={th:.2f}  n_ovr={n_ovr:>6,}  correct={correct:>5,}  "
              f"prec={prec:>5.1%}  delta={delta:+.5f}")

    # Emit: find best theta (max delta with n_ovr>0)
    best = max((r for r in results if r["n_ovr"] > 0),
               key=lambda r: r["delta"], default=None)
    if best is None or best["delta"] <= 0:
        print("\nNo positive-delta operating point found. No submission emitted.")
    else:
        print(f"\nbest: theta={best['theta']}  delta={best['delta']:+.5f}  "
              f"prec={best['precision']:.2%}")
        # Build test-side submission
        flip_test = mask_te_s6 & (ph_test > best["theta"]) & (teacher_pred_test == CLS2IDX["Medium"])
        n_test_ovr = int(flip_test.sum())
        new_test_pred = teacher_pred_test.copy()
        new_test_pred[flip_test] = CLS2IDX["High"]
        cls_idx = {v: k for k, v in CLS2IDX.items()}
        sub = pd.DataFrame({
            "id": te["id"].to_numpy(),
            "Irrigation_Need": [cls_idx[i] for i in new_test_pred],
        })
        tag = f"th{int(best['theta']*100):02d}"
        sub_path = SUB / f"submission_spec6_override_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"test override rows: {n_test_ovr:,} at theta={best['theta']}")
        print(f"wrote {sub_path}")
        print(f"test dist: {dict(sub['Irrigation_Need'].value_counts())}")

    summary = dict(
        teacher_bal_acc=float(base_bal),
        teacher_per_class_recall=base_recall.tolist(),
        teacher_bias=bias.tolist(),
        w_recipe=W_RECIPE, w_s1=W_S1, w_s7=W_S7,
        n_score6_train=int(mask_tr_s6.sum()),
        n_score6_test=int(mask_te_s6.sum()),
        override_space_size=int(mask_override_oof.sum()),
        high_in_override_space=n_high_in_space,
        theta_sweep=results,
    )
    out = ART / "spec6_deploy_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
