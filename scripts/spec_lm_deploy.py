"""Deploy spec_lm_v3 (score=3 L↔M specialist) as a hard override on the
NEW LB-best 3-stack. Flips stack_argmax=Low → Medium where P_spec > theta.

Break-even precision under macro-recall for L→M override:
  L_count=369917, M_count=239074 → break-even = M/(L+M) = 0.393

Sweeps theta from 0.05 to 0.99 (fine near 0.4) + top-N rank deploy.
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
BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
IDX2CLS = {v: k for k, v in CLS2IDX.items()}
SPEC_SCORE = int(__import__("os").environ.get("SPEC_SCORE", "3"))


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
    print(f"LB-best 3-stack OOF = {base_bal:.5f}")
    print(f"per-class recall: L={base_rec[0]:.4f} M={base_rec[1]:.4f} H={base_rec[2]:.4f}")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    ph_oof = np.load(ART / f"oof_spec_lm_v3_score{SPEC_SCORE}.npy")
    ph_test = np.load(ART / f"test_spec_lm_v3_score{SPEC_SCORE}.npy")

    mask_ovr_oof = (s_tr == SPEC_SCORE) & (pred_oof == CLS2IDX["Low"])
    mask_ovr_test = (s_te == SPEC_SCORE) & (pred_test == CLS2IDX["Low"])
    n_med_in_ovr = int((y[mask_ovr_oof] == CLS2IDX["Medium"]).sum())
    print(f"\noverride space OOF (score={SPEC_SCORE}, argmax=Low): "
          f"{int(mask_ovr_oof.sum()):,}  truly-Medium: {n_med_in_ovr}")
    print(f"override space test: {int(mask_ovr_test.sum()):,}")

    cc = np.bincount(y, minlength=3)
    break_even = cc[1] / (cc[0] + cc[1])
    print(f"break-even precision (L→M under macro-recall): {break_even:.4f}")

    # Theta sweep
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print(f"\n=== theta sweep ===")
    print(f"{'theta':>6} {'n_ovr':>6} {'ok':>5} {'bad':>5} {'prec':>6} "
          f"{'dRecL':>8} {'dRecM':>8} {'bal':>9} {'delta':>9}")
    rows = []
    for th in grid:
        fm = mask_ovr_oof & (ph_oof > th)
        n = int(fm.sum())
        ok = int((y[fm] == CLS2IDX["Medium"]).sum())
        bad = n - ok
        prec = ok / max(n, 1)
        new = pred_oof.copy()
        new[fm] = CLS2IDX["Medium"]
        b = fast_bal_acc(y, new)
        r = per_class_recall(y, new)
        dr = r - base_rec
        d = b - base_bal
        rows.append(dict(theta=th, n=n, correct=ok, wrong=bad, precision=prec,
                         d_rec_L=float(dr[0]), d_rec_M=float(dr[1]),
                         bal=float(b), delta=float(d)))
        print(f"{th:>6.3f} {n:>6} {ok:>5} {bad:>5} {prec:>6.1%} "
              f"{dr[0]:>+8.5f} {dr[1]:>+8.5f} {b:>9.5f} {d:>+9.5f}")
    best = max(rows, key=lambda x: x["delta"])

    # Top-N rank deploy (idealised precision)
    print(f"\n=== top-N rank deploy ===")
    print(f"{'N':>5} {'ok':>5} {'prec':>6} {'delta':>9}")
    top_rows = []
    in_space = np.where(mask_ovr_oof)[0]
    ranked = in_space[np.argsort(-ph_oof[in_space])]
    for N in (50, 100, 200, 500, 1000, 2000, 4000):
        flip = ranked[:N]
        ok = int((y[flip] == CLS2IDX["Medium"]).sum())
        prec = ok / N
        new = pred_oof.copy()
        new[flip] = CLS2IDX["Medium"]
        b = fast_bal_acc(y, new)
        top_rows.append(dict(N=N, ok=ok, prec=prec, delta=float(b - base_bal)))
        print(f"{N:>5} {ok:>5} {prec:>6.1%} {b-base_bal:>+9.5f}")

    # AUC + ph distribution diagnostic
    in_space_med = (y[mask_ovr_oof] == CLS2IDX["Medium"])
    ph_in_space = ph_oof[mask_ovr_oof]
    print(f"\nph in-override-space distribution (truly-M vs truly-L):")
    print(f"  truly-M (n={in_space_med.sum()}): "
          f"min={ph_in_space[in_space_med].min():.4f} "
          f"median={np.median(ph_in_space[in_space_med]):.4f} "
          f"max={ph_in_space[in_space_med].max():.4f}")
    print(f"  truly-L (n={(~in_space_med).sum()}): "
          f"min={ph_in_space[~in_space_med].min():.4f} "
          f"median={np.median(ph_in_space[~in_space_med]):.4f} "
          f"max={ph_in_space[~in_space_med].max():.4f}")

    # Emit submission if best delta > 0
    print(f"\n=== BEST θ={best['theta']}  n={best['n']}  "
          f"prec={best['precision']:.2%}  delta={best['delta']:+.5f} ===")

    if best["delta"] > 0:
        fm_te = mask_ovr_test & (ph_test > best["theta"])
        new_te = pred_test.copy()
        new_te[fm_te] = CLS2IDX["Medium"]
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [IDX2CLS[i] for i in new_te]
        tag = f"s{SPEC_SCORE}_th{int(best['theta']*1000):03d}"
        path = SUB / f"submission_spec_lm_v3_{tag}.csv"
        sub.to_csv(path, index=False)
        print(f"test overrides: {int(fm_te.sum()):,}")
        print(f"wrote {path}")
    else:
        print("no positive-delta operating point — no submission")

    out = dict(
        base_bal=float(base_bal),
        base_per_class=base_rec.tolist(),
        break_even_precision=float(break_even),
        override_space_oof=int(mask_ovr_oof.sum()),
        truly_medium_in_ovr=n_med_in_ovr,
        theta_sweep=rows,
        top_n_rank=top_rows,
        best=best,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / f"spec_lm_deploy_score{SPEC_SCORE}_results.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
