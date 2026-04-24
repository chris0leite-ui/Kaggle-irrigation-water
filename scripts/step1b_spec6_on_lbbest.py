"""Step 1b: Deploy spec6_mh_v2 as a hard override on the NEW LB-best 3-stack.

spec6_mh_v2 was trained with teacher meta-features from the OLD 3-way teacher
(OOF 0.98029 / LB 0.98005). The NEW LB-best is the 3-stack (OOF 0.98061 /
LB 0.98008). Question: does the spec6 override mechanism transfer across
stacks?

Mechanism: for rows with score=6 AND new_argmax == Medium, flip to High if
P_spec6_v2(row) > theta. Under macro-recall, break-even precision = H/M
counts in the override space.

This is cheap (no retraining) — uses existing spec6_mh_v2 OOF/test probs
and just swaps which teacher's Medium argmax defines the override space.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    add_distance_features, fast_bal_acc, log_blend, CLS2IDX,
)
from sklearn.isotonic import IsotonicRegression

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def per_class_recall(y, pred):
    cc = np.bincount(y, minlength=3)
    hit = np.array([((pred == k) & (y == k)).sum() for k in range(3)],
                   dtype=np.int64)
    return hit / np.maximum(cc, 1)


def build_lbbest_stack(y):
    recipe_oof = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    recipe_test = _normed(np.load(ART / "test_recipe_full_te.npy"))
    ps1_oof = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    ps1_test = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    ps7_oof = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    ps7_test = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm_oof = _normed(np.load(ART / "oof_realmlp.npy"))
    rm_test = _normed(np.load(ART / "test_realmlp.npy"))
    nr_oof = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_test = _normed(np.load(ART / "test_xgb_nonrule.npy"))

    nr_iso_oof, nr_iso_test = iso_cal(nr_oof, nr_test, y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_oof = log_blend([recipe_oof, ps1_oof, ps7_oof], w3)
    lb3_test = log_blend([recipe_test, ps1_test, ps7_test], w3)
    s1_oof = log_blend([lb3_oof, rm_oof], np.array([0.8, 0.2]))
    s1_test = log_blend([lb3_test, rm_test], np.array([0.8, 0.2]))
    s2_oof = log_blend([s1_oof, nr_iso_oof], np.array([0.925, 0.075]))
    s2_test = log_blend([s1_test, nr_iso_test], np.array([0.925, 0.075]))

    pred_oof = (np.log(np.clip(s2_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_test = (np.log(np.clip(s2_test, 1e-12, 1)) + BIAS).argmax(1)
    return s2_oof, s2_test, pred_oof, pred_test


def theta_sweep(name, ph, mask_ovr, y, base_pred, base_bal, base_rec):
    rows = []
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"\n=== {name} theta sweep ===")
    print(f"{'theta':>6} {'n_ovr':>6} {'ok':>5} {'bad':>5} {'prec':>6} "
          f"{'dRecH':>8} {'dRecM':>8} {'bal':>8} {'delta':>8}")
    for th in grid:
        fm = mask_ovr & (ph > th)
        n = int(fm.sum())
        ok = int((y[fm] == CLS2IDX["High"]).sum())
        bad = n - ok
        prec = ok / max(n, 1)
        new = base_pred.copy()
        new[fm] = CLS2IDX["High"]
        b = fast_bal_acc(y, new)
        r = per_class_recall(y, new)
        dr = r - base_rec
        d = b - base_bal
        rows.append(dict(theta=th, n=n, correct=ok, wrong=bad, precision=prec,
                         d_rec_H=float(dr[2]), d_rec_M=float(dr[1]),
                         bal=float(b), delta=float(d)))
        print(f"{th:>6.2f} {n:>6} {ok:>5} {bad:>5} {prec:>6.1%} "
              f"{dr[2]:>+8.5f} {dr[1]:>+8.5f} {b:>8.5f} {d:>+8.5f}")
    return rows


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    # NEW teacher = LB-best 3-stack
    _, _, pred_oof, pred_test = build_lbbest_stack(y)
    base_bal = fast_bal_acc(y, pred_oof)
    base_rec = per_class_recall(y, pred_oof)
    print(f"NEW teacher (LB-best stack) OOF  = {base_bal:.5f}")
    print(f"per-class recall: L={base_rec[0]:.4f} M={base_rec[1]:.4f} H={base_rec[2]:.4f}")

    # dgp_score
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    # override space (score=6 AND teacher predicts Medium)
    mask_ovr_oof = (s_tr == 6) & (pred_oof == CLS2IDX["Medium"])
    mask_ovr_test = (s_te == 6) & (pred_test == CLS2IDX["Medium"])
    n_high_in = int((y[mask_ovr_oof] == CLS2IDX["High"]).sum())
    print(f"override space OOF:  {int(mask_ovr_oof.sum()):,}  (truly-High: {n_high_in})")
    print(f"override space test: {int(mask_ovr_test.sum()):,}")

    cc = np.bincount(y, minlength=3)
    break_even_precision = cc[2] / (cc[1] + cc[2])
    print(f"break-even precision (macro-recall): {break_even_precision:.4f}")

    ph_v2 = np.load(ART / "oof_spec6_mh_v2.npy")
    ph_v2_test = np.load(ART / "test_spec6_mh_v2.npy")
    rows = theta_sweep("v2 on NEW teacher", ph_v2, mask_ovr_oof, y,
                        pred_oof, base_bal, base_rec)
    best = max(rows, key=lambda r: r["delta"])
    print(f"\n=== BEST v2 on NEW teacher ===")
    print(f"theta={best['theta']}  n={best['n']}  correct={best['correct']}  "
          f"prec={best['precision']:.2%}  delta={best['delta']:+.5f}")

    # Compare with the old-teacher result recorded in spec6_deploy_v2_results.json
    try:
        prev = json.loads((ART / "spec6_deploy_v2_results.json").read_text())
        prev_best = prev.get("overall_best")
        if prev_best:
            print("\n=== OLD teacher spec6 v2 (from spec6_deploy_v2_results.json) ===")
            print(f"variant={prev_best.get('variant')} theta={prev_best.get('theta')} "
                  f"n={prev_best.get('n')} prec={prev_best.get('precision'):.2%} "
                  f"delta={prev_best.get('delta'):+.5f}")
    except Exception:
        pass

    out = dict(
        new_teacher_bal=float(base_bal),
        new_teacher_per_class_recall=base_rec.tolist(),
        override_space_size=int(mask_ovr_oof.sum()),
        high_in_override=n_high_in,
        break_even_precision=float(break_even_precision),
        theta_sweep=rows,
        best=best,
    )
    (ART / "step1b_spec6_on_lbbest_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/artifacts/step1b_spec6_on_lbbest_results.json")
    print(f"elapsed: {time.time() - t0:.1f}s")

    # Emit submission if delta > 0
    if best["delta"] > 0:
        fm_test = mask_ovr_test & (ph_v2_test > best["theta"])
        new_test = pred_test.copy()
        new_test[fm_test] = CLS2IDX["High"]
        idx_to_cls = {v: k for k, v in CLS2IDX.items()}
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [idx_to_cls[i] for i in new_test]
        path = SUB / f"submission_step1b_spec6_v2_newt_th{int(best['theta']*100):02d}.csv"
        sub.to_csv(path, index=False)
        print(f"\ntest overrides: {int(fm_test.sum()):,}")
        print(f"wrote {path}")
    else:
        print("\nno positive delta; no submission emitted")


if __name__ == "__main__":
    main()
