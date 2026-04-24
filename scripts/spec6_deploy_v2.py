"""Deploy v1+v2 spec6 specialists. Hard-override + variant comparison.

Extends scripts/spec6_deploy.py with:
  - v2 artefacts (teacher-meta-feature specialist, OOF AUC 0.938 vs v1 0.862)
  - wider theta grid (0.10 .. 0.99) to find the sharp precision cliff
  - top-N rank deploy (idealized precision ceiling), reports break-even
  - combined deploy: override if v1>t1 AND v2>t2 (precision-boost)
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
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40


def per_class_recall(y, pred, K=3):
    cc = np.bincount(y, minlength=K)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(K)], dtype=np.int64)
    return hit / np.maximum(cc, 1)


def build_teacher():
    oofs = [np.load(ART / f"oof_{n}.npy") for n in
            ("recipe_full_te", "recipe_pseudolabel", "recipe_pseudolabel_seed7labeler")]
    tests = [np.load(ART / f"test_{n}.npy") for n in
             ("recipe_full_te", "recipe_pseudolabel", "recipe_pseudolabel_seed7labeler")]
    w = np.array([W_RECIPE, W_S1, W_S7])
    return log_blend(oofs, w), log_blend(tests, w)


def theta_sweep(name, ph, mask_ovr, y, teacher_pred, base_bal, base_rec):
    rows = []
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    print(f"\n=== {name} theta sweep (override teacher_pred=Medium→High on score=6) ===")
    print(f"{'theta':>6} {'n_ovr':>6} {'ok':>5} {'bad':>5} {'prec':>6} "
          f"{'dRecH':>8} {'dRecM':>8} {'bal':>8} {'delta':>8}")
    for th in grid:
        fm = mask_ovr & (ph > th)
        n = int(fm.sum())
        ok = int((y[fm] == CLS2IDX["High"]).sum())
        bad = n - ok
        prec = ok / max(n, 1)
        new = teacher_pred.copy()
        new[fm] = CLS2IDX["High"]
        bal = fast_bal_acc(y, new)
        rec = per_class_recall(y, new)
        dr = rec - base_rec
        delta = bal - base_bal
        rows.append(dict(theta=th, n=n, correct=ok, wrong=bad,
                         precision=prec, d_rec_H=float(dr[2]),
                         d_rec_M=float(dr[1]), bal=float(bal),
                         delta=float(delta)))
        print(f"{th:>6.2f} {n:>6} {ok:>5} {bad:>5} {prec:>6.1%} "
              f"{dr[2]:>+8.5f} {dr[1]:>+8.5f} {bal:>8.5f} {delta:>+8.5f}")
    return rows


def top_n_sweep(name, ph, mask_ovr, y, teacher_pred, base_bal):
    print(f"\n=== {name} top-N rank deploy (idealised precision ceiling) ===")
    print(f"{'N':>5} {'ok':>5} {'prec':>6} {'delta':>8}")
    # Scores only within the override space
    in_space = np.where(mask_ovr)[0]
    ranked = in_space[np.argsort(-ph[in_space])]
    for N in (50, 100, 200, 331, 500, 1000):
        flip = ranked[:N]
        ok = int((y[flip] == CLS2IDX["High"]).sum())
        prec = ok / N
        new = teacher_pred.copy()
        new[flip] = CLS2IDX["High"]
        bal = fast_bal_acc(y, new)
        print(f"{N:>5} {ok:>5} {prec:>6.1%} {bal-base_bal:>+8.5f}")


def combined_sweep(name, ph1, ph2, mask_ovr, y, teacher_pred, base_bal,
                    base_rec):
    print(f"\n=== {name} combined deploy (v1 > t1 AND v2 > t2) ===")
    print(f"{'t1':>5} {'t2':>5} {'n':>5} {'ok':>5} {'prec':>6} {'delta':>8}")
    best = None
    for t1 in (0.10, 0.20, 0.30, 0.40, 0.50):
        for t2 in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
            fm = mask_ovr & (ph1 > t1) & (ph2 > t2)
            n = int(fm.sum())
            if n == 0:
                continue
            ok = int((y[fm] == CLS2IDX["High"]).sum())
            prec = ok / n
            new = teacher_pred.copy()
            new[fm] = CLS2IDX["High"]
            bal = fast_bal_acc(y, new)
            delta = bal - base_bal
            if best is None or delta > best["delta"]:
                best = dict(t1=t1, t2=t2, n=n, ok=ok, prec=prec, delta=delta)
            if delta > 0:
                print(f"{t1:>5.2f} {t2:>5.2f} {n:>5} {ok:>5} {prec:>6.1%} "
                      f"{delta:>+8.5f}")
    return best


def main():
    print("=== teacher reconstruction ===")
    oof_t, test_t = build_teacher()
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    bias, base_bal = tune_log_bias(oof_t, y, prior)
    print(f"teacher bal_acc={base_bal:.5f}  bias={bias.round(3).tolist()}")

    eps = 1e-9
    teacher_pred_oof = (np.log(np.clip(oof_t, eps, 1.0)) + bias).argmax(1)
    teacher_pred_test = (np.log(np.clip(test_t, eps, 1.0)) + bias).argmax(1)
    base_rec = per_class_recall(y, teacher_pred_oof)
    print(f"per-class recall L/M/H = "
          f"{base_rec[0]:.4f} / {base_rec[1]:.4f} / {base_rec[2]:.4f}")

    tr_d = add_distance_features(tr)
    te_d = add_distance_features(te)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    mask_ovr_oof = (s_tr == 6) & (teacher_pred_oof == CLS2IDX["Medium"])
    mask_ovr_test = (s_te == 6) & (teacher_pred_test == CLS2IDX["Medium"])
    n_high_in_space = int((y[mask_ovr_oof] == CLS2IDX["High"]).sum())
    print(f"\noverride space (OOF)  : {mask_ovr_oof.sum():,}  "
          f"(truly-High: {n_high_in_space})")
    print(f"override space (test) : {mask_ovr_test.sum():,}")
    # break-even precision
    cc = np.bincount(y, minlength=3)
    break_even = cc[2] / cc[1]
    print(f"break-even precision under macro-recall: "
          f"H_count / M_count = {cc[2]}/{cc[1]} = {break_even:.4f} "
          f"-> >={break_even/(1+break_even):.4f} precision")

    ph_v1 = np.load(ART / "oof_spec6_mh.npy")
    ph_v2 = np.load(ART / "oof_spec6_mh_v2.npy")
    ph_v1_test = np.load(ART / "test_spec6_mh.npy")
    ph_v2_test = np.load(ART / "test_spec6_mh_v2.npy")

    r_v1 = theta_sweep("v1 (dist+nonrule)", ph_v1, mask_ovr_oof, y,
                        teacher_pred_oof, base_bal, base_rec)
    r_v2 = theta_sweep("v2 (+teacher meta)", ph_v2, mask_ovr_oof, y,
                        teacher_pred_oof, base_bal, base_rec)

    top_n_sweep("v1", ph_v1, mask_ovr_oof, y, teacher_pred_oof, base_bal)
    top_n_sweep("v2", ph_v2, mask_ovr_oof, y, teacher_pred_oof, base_bal)

    combined_best = combined_sweep("v1 AND v2", ph_v1, ph_v2,
                                    mask_ovr_oof, y, teacher_pred_oof,
                                    base_bal, base_rec)
    print(f"\ncombined best: {combined_best}")

    # Best of any deploy
    all_results = [("v1", r_v1), ("v2", r_v2)]
    best = None
    best_tag = None
    for tag, rows in all_results:
        for r in rows:
            if best is None or r["delta"] > best["delta"]:
                best = r; best_tag = tag
    print(f"\n=== OVERALL BEST ===")
    print(f"  variant={best_tag}  theta={best['theta']}  "
          f"n_overr={best['n']}  correct={best['correct']}  "
          f"prec={best['precision']:.2%}  delta={best['delta']:+.5f}")

    # Emit submission using the best variant+theta (only if delta > 0)
    if best is not None and best["delta"] > 0:
        ph_test = ph_v1_test if best_tag == "v1" else ph_v2_test
        flip_test = mask_ovr_test & (ph_test > best["theta"])
        new_test = teacher_pred_test.copy()
        new_test[flip_test] = CLS2IDX["High"]
        cls_idx = {v: k for k, v in CLS2IDX.items()}
        sub = pd.DataFrame({
            "id": te["id"].to_numpy(),
            "Irrigation_Need": [cls_idx[i] for i in new_test],
        })
        tag = f"{best_tag}_th{int(best['theta']*100):02d}"
        sub_path = SUB / f"submission_spec6_override_{tag}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"\ntest overrides: {int(flip_test.sum()):,}")
        print(f"wrote {sub_path}")
        print(f"test dist: {dict(sub['Irrigation_Need'].value_counts())}")
    else:
        print("\nNo positive-delta operating point — no submission emitted.")

    # Save consolidated diagnostics
    out = dict(
        teacher_bal_acc=float(base_bal),
        teacher_per_class_recall=base_rec.tolist(),
        break_even_precision=float(break_even / (1 + break_even)),
        override_space_size=int(mask_ovr_oof.sum()),
        high_in_override_space=n_high_in_space,
        v1_theta_sweep=r_v1,
        v2_theta_sweep=r_v2,
        combined_best=combined_best,
        overall_best=dict(variant=best_tag, **best) if best else None,
    )
    with open(ART / "spec6_deploy_v2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote scripts/artifacts/spec6_deploy_v2_results.json")


if __name__ == "__main__":
    main()
