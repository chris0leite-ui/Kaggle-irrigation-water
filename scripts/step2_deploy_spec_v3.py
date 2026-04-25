"""Step 2: deploy spec_mh_v3 specialists (score=5 and score=6) against the
NEW LB-best 3-stack (OOF 0.98061 / LB 0.98008). Sweeps per-score theta +
combined deploy. Emits submissions for any positive-delta config.

v3 trained with NEW teacher meta-features (unlike v2 which used the old
3-way teacher), so override rows should better align with the new stack's
Medium-argmax predictions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    add_distance_features, fast_bal_acc, log_blend, CLS2IDX,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])
IDX2CLS = {v: k for k, v in CLS2IDX.items()}


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


def build_lbbest_stack(y):
    r = (_normed(np.load(ART / "oof_recipe_full_te.npy")),
         _normed(np.load(ART / "test_recipe_full_te.npy")))
    s1 = (_normed(np.load(ART / "oof_recipe_pseudolabel.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel.npy")))
    s7 = (_normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")),
          _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")))
    rm = (_normed(np.load(ART / "oof_realmlp.npy")),
          _normed(np.load(ART / "test_realmlp.npy")))
    nr = (_normed(np.load(ART / "oof_xgb_nonrule.npy")),
          _normed(np.load(ART / "test_xgb_nonrule.npy")))
    nr_iso_o, nr_iso_t = iso_cal(nr[0], nr[1], y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r[0], s1[0], s7[0]], w3)
    lb3_t = log_blend([r[1], s1[1], s7[1]], w3)
    s1_o = log_blend([lb3_o, rm[0]], np.array([0.8, 0.2]))
    s1_t = log_blend([lb3_t, rm[1]], np.array([0.8, 0.2]))
    s2_o = log_blend([s1_o, nr_iso_o], np.array([0.925, 0.075]))
    s2_t = log_blend([s1_t, nr_iso_t], np.array([0.925, 0.075]))
    return s2_o, s2_t


def per_class_recall(y, pred):
    cc = np.bincount(y, minlength=3)
    hit = np.array([((pred == k) & (y == k)).sum() for k in range(3)],
                   dtype=np.int64)
    return hit / np.maximum(cc, 1)


def theta_sweep(name, ph_oof, mask_ovr, y, base_pred, base_bal, base_rec):
    grid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]
    print(f"\n=== {name} theta sweep ===")
    print(f"{'theta':>6} {'n_ovr':>6} {'ok':>5} {'bad':>5} {'prec':>6} "
          f"{'dRecH':>8} {'dRecM':>8} {'bal':>9} {'delta':>9}")
    rows = []
    for th in grid:
        fm = mask_ovr & (ph_oof > th)
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
        print(f"{th:>6.3f} {n:>6} {ok:>5} {bad:>5} {prec:>6.1%} "
              f"{dr[2]:>+8.5f} {dr[1]:>+8.5f} {b:>9.5f} {d:>+9.5f}")
    return rows


def joint_sweep(name, ph5_oof, ph6_oof, s_tr, y, base_pred, base_bal, base_rec):
    """Independent per-band theta tuning — picks best theta per band and
    applies both overrides jointly. Uses greedy optimization on OOF."""
    grid = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]
    base_mask_5 = (s_tr == 5) & (base_pred == CLS2IDX["Medium"])
    base_mask_6 = (s_tr == 6) & (base_pred == CLS2IDX["Medium"])
    print(f"\n=== {name} JOINT theta sweep (score=5 × score=6) ===")
    print(f"{'t5':>6} {'t6':>6} {'n5':>5} {'n6':>5} {'ok5':>4} {'ok6':>4} "
          f"{'bal':>9} {'delta':>9}")
    rows = []
    best = None
    for t5 in grid:
        for t6 in grid:
            fm5 = base_mask_5 & (ph5_oof > t5)
            fm6 = base_mask_6 & (ph6_oof > t6)
            fm = fm5 | fm6
            if fm.sum() == 0:
                continue
            ok5 = int((y[fm5] == CLS2IDX["High"]).sum())
            ok6 = int((y[fm6] == CLS2IDX["High"]).sum())
            new = base_pred.copy()
            new[fm] = CLS2IDX["High"]
            b = fast_bal_acc(y, new)
            d = b - base_bal
            row = dict(t5=t5, t6=t6, n5=int(fm5.sum()), n6=int(fm6.sum()),
                       ok5=ok5, ok6=ok6, bal=float(b), delta=float(d))
            rows.append(row)
            if best is None or d > best["delta"]:
                best = row
            if d > 0:
                print(f"{t5:>6.3f} {t6:>6.3f} {int(fm5.sum()):>5} "
                      f"{int(fm6.sum()):>5} {ok5:>4} {ok6:>4} {b:>9.5f} {d:>+9.5f}")
    return rows, best


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    oof_t, test_t = build_lbbest_stack(y)
    pred_oof = (np.log(np.clip(oof_t, 1e-12, 1)) + BIAS).argmax(1)
    pred_test = (np.log(np.clip(test_t, 1e-12, 1)) + BIAS).argmax(1)
    base_bal = fast_bal_acc(y, pred_oof)
    base_rec = per_class_recall(y, pred_oof)
    print(f"NEW teacher (LB-best 3-stack) OOF = {base_bal:.5f}")
    print(f"per-class recall: L={base_rec[0]:.4f} M={base_rec[1]:.4f} H={base_rec[2]:.4f}")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    ph5_oof = np.load(ART / "oof_spec_mh_v3_score5.npy")
    ph6_oof = np.load(ART / "oof_spec_mh_v3_score6.npy")
    ph5_test = np.load(ART / "test_spec_mh_v3_score5.npy")
    ph6_test = np.load(ART / "test_spec_mh_v3_score6.npy")

    # override spaces vs NEW teacher
    mask_ovr_5 = (s_tr == 5) & (pred_oof == CLS2IDX["Medium"])
    mask_ovr_6 = (s_tr == 6) & (pred_oof == CLS2IDX["Medium"])
    n_h_5 = int((y[mask_ovr_5] == CLS2IDX["High"]).sum())
    n_h_6 = int((y[mask_ovr_6] == CLS2IDX["High"]).sum())
    print(f"score=5 override space OOF: {int(mask_ovr_5.sum()):,}  truly-H: {n_h_5}")
    print(f"score=6 override space OOF: {int(mask_ovr_6.sum()):,}  truly-H: {n_h_6}")

    cc = np.bincount(y, minlength=3)
    break_even_prec = cc[2] / (cc[1] + cc[2])
    print(f"break-even precision: {break_even_prec:.4f}")

    r5 = theta_sweep("score=5", ph5_oof, mask_ovr_5, y, pred_oof, base_bal, base_rec)
    r6 = theta_sweep("score=6", ph6_oof, mask_ovr_6, y, pred_oof, base_bal, base_rec)

    best_5 = max(r5, key=lambda x: x["delta"])
    best_6 = max(r6, key=lambda x: x["delta"])
    print(f"\nbest score=5: θ={best_5['theta']}  n={best_5['n']}  "
          f"prec={best_5['precision']:.2%}  Δ={best_5['delta']:+.5f}")
    print(f"best score=6: θ={best_6['theta']}  n={best_6['n']}  "
          f"prec={best_6['precision']:.2%}  Δ={best_6['delta']:+.5f}")

    # JOINT sweep
    rj, best_j = joint_sweep("joint", ph5_oof, ph6_oof, s_tr, y,
                              pred_oof, base_bal, base_rec)
    print(f"\nbest joint: {best_j}")

    # Emit submissions for every positive config
    ART.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    emitted = []

    if best_5["delta"] > 0:
        mask_te_5 = (s_te == 5) & (pred_test == CLS2IDX["Medium"])
        fm_te = mask_te_5 & (ph5_test > best_5["theta"])
        nt = pred_test.copy(); nt[fm_te] = CLS2IDX["High"]
        sub = pd.DataFrame({"id": test["id"].values,
                            "Irrigation_Need": [IDX2CLS[i] for i in nt]})
        path = SUB / f"submission_step2_spec5_th{int(best_5['theta']*1000):03d}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  test overrides: {int(fm_te.sum()):,}")
        emitted.append(("spec5_only", str(path), best_5))

    if best_6["delta"] > 0:
        mask_te_6 = (s_te == 6) & (pred_test == CLS2IDX["Medium"])
        fm_te = mask_te_6 & (ph6_test > best_6["theta"])
        nt = pred_test.copy(); nt[fm_te] = CLS2IDX["High"]
        sub = pd.DataFrame({"id": test["id"].values,
                            "Irrigation_Need": [IDX2CLS[i] for i in nt]})
        path = SUB / f"submission_step2_spec6_th{int(best_6['theta']*1000):03d}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  test overrides: {int(fm_te.sum()):,}")
        emitted.append(("spec6_only", str(path), best_6))

    if best_j is not None and best_j["delta"] > 0:
        t5, t6 = best_j["t5"], best_j["t6"]
        mask_te_5 = (s_te == 5) & (pred_test == CLS2IDX["Medium"])
        mask_te_6 = (s_te == 6) & (pred_test == CLS2IDX["Medium"])
        fm_te = (mask_te_5 & (ph5_test > t5)) | (mask_te_6 & (ph6_test > t6))
        nt = pred_test.copy(); nt[fm_te] = CLS2IDX["High"]
        sub = pd.DataFrame({"id": test["id"].values,
                            "Irrigation_Need": [IDX2CLS[i] for i in nt]})
        path = SUB / f"submission_step2_joint_t5{int(t5*1000):03d}_t6{int(t6*1000):03d}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  test overrides: {int(fm_te.sum()):,}")
        emitted.append(("joint", str(path), best_j))

    out = dict(
        base_bal=float(base_bal),
        base_per_class=base_rec.tolist(),
        break_even_precision=float(break_even_prec),
        override_space_5=int(mask_ovr_5.sum()),
        override_space_6=int(mask_ovr_6.sum()),
        truly_high_5=n_h_5, truly_high_6=n_h_6,
        score5_sweep=r5, score6_sweep=r6,
        best_5=best_5, best_6=best_6, best_joint=best_j,
        emitted=[{"tag": t, "path": p, "config": c} for t, p, c in emitted],
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "step2_deploy_spec_v3_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/artifacts/step2_deploy_spec_v3_results.json")


if __name__ == "__main__":
    main()
