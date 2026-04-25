"""Tier 1b #4: combine meta-stacker blend + spec_lm_v3 override on LB-best.

Individual contributions on LB-best 3-stack (OOF 0.98061):
  - meta-stacker (xgb_metastack) α=0.40:   Δ=+0.00012
  - spec_lm_v3 score=3 override θ=0.35:    Δ=+0.00002

Combined: blend LB-best with meta-stacker, then apply spec_lm_v3 override
on the blend's Medium-argmax at score=3.

Also adds a fine-grained meta-stacker α grid (0.30, 0.325, 0.35, 0.375,
0.40, 0.425, 0.45) and searches the joint (α_meta, θ_spec_lm) grid.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    add_distance_features, fast_bal_acc, log_blend, CLS2IDX,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
IDX2CLS = {v: k for k, v in CLS2IDX.items()}
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


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    lb_o, lb_t = build_lbbest_stack(y)
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))
    spec_lm_o = np.load(ART / "oof_spec_lm_v3_score3.npy")
    spec_lm_t = np.load(ART / "test_spec_lm_v3_score3.npy")

    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    s_tr = tr_d["dgp_score"].to_numpy()
    s_te = te_d["dgp_score"].to_numpy()

    base = bal(lb_o, y)
    print(f"LB-best 3-stack OOF = {base:.5f}")

    # Fine meta α grid
    print("\n=== fine meta-stacker α grid ===")
    metas = []
    for a in [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475]:
        blend_o = log_blend([lb_o, meta_o], np.array([1 - a, a]))
        blend_t = log_blend([lb_t, meta_t], np.array([1 - a, a]))
        b = bal(blend_o, y)
        pred = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
        errs = int((pred != y).sum())
        print(f"α_meta={a:.3f}  OOF={b:.5f}  errs={errs}  Δ={b-base:+.5f}")
        metas.append(dict(a=a, oof=float(b), errs=errs,
                          blend_oof=blend_o, blend_test=blend_t))
    best_meta = max(metas, key=lambda r: r["oof"])
    print(f"best meta α: {best_meta['a']}  OOF={best_meta['oof']:.5f}")

    # Use meta-best blend as new base; apply spec_lm_v3 override on top
    print("\n=== spec_lm_v3 on meta-enhanced stack ===")
    base_mo = best_meta["blend_oof"]
    base_mt = best_meta["blend_test"]
    pred_mo = (np.log(np.clip(base_mo, 1e-12, 1)) + BIAS).argmax(1)
    pred_mt = (np.log(np.clip(base_mt, 1e-12, 1)) + BIAS).argmax(1)
    base_mbal = fast_bal_acc(y, pred_mo)
    base_mrec = per_class_recall(y, pred_mo)
    print(f"meta-enhanced OOF = {base_mbal:.5f}  Δ vs LB-best = {base_mbal-base:+.5f}")
    print(f"per-class: L={base_mrec[0]:.4f} M={base_mrec[1]:.4f} H={base_mrec[2]:.4f}")

    mask_ovr = (s_tr == 3) & (pred_mo == CLS2IDX["Low"])
    mask_ovr_te = (s_te == 3) & (pred_mt == CLS2IDX["Low"])
    print(f"score=3 L-argmax space: {int(mask_ovr.sum()):,}  "
          f"truly-M: {int((y[mask_ovr] == CLS2IDX['Medium']).sum())}")

    print(f"{'θ':>6} {'n':>6} {'ok':>5} {'prec':>6} {'bal':>9} {'Δ vs LB':>9} {'Δ vs meta':>10}")
    combo_rows = []
    for th in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        fm = mask_ovr & (spec_lm_o > th)
        n = int(fm.sum())
        ok = int((y[fm] == CLS2IDX["Medium"]).sum())
        prec = ok / max(n, 1)
        new = pred_mo.copy(); new[fm] = CLS2IDX["Medium"]
        b = fast_bal_acc(y, new)
        combo_rows.append(dict(theta=th, n=n, correct=ok, prec=prec,
                               bal=float(b),
                               delta_vs_lb=float(b - base),
                               delta_vs_meta=float(b - base_mbal)))
        print(f"{th:>6.3f} {n:>6} {ok:>5} {prec:>6.1%} {b:>9.5f} "
              f"{b-base:>+9.5f} {b-base_mbal:>+10.5f}")

    best_combo = max(combo_rows, key=lambda r: r["delta_vs_lb"])
    print(f"\nBEST COMBO: meta α={best_meta['a']}  spec θ={best_combo['theta']}")
    print(f"  OOF={best_combo['bal']:.5f}  Δ vs LB-best={best_combo['delta_vs_lb']:+.5f}")

    # Emit if Δ vs LB-best ≥ +0.0002
    if best_combo["delta_vs_lb"] >= 2e-4:
        th = best_combo["theta"]
        fm_te = mask_ovr_te & (spec_lm_t > th)
        new_te = pred_mt.copy(); new_te[fm_te] = CLS2IDX["Medium"]
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub["Irrigation_Need"] = [IDX2CLS[i] for i in new_te]
        am = int(best_meta["a"] * 1000)
        th_i = int(th * 1000)
        path = SUB / f"submission_tier1b_combo_m{am:03d}_th{th_i:03d}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  test overrides: {int(fm_te.sum()):,}")
    else:
        print(f"\nbest Δ={best_combo['delta_vs_lb']:+.5f} below +2e-4 gate; no submission")

    # Save a lite summary (drop the blend_oof arrays to keep the JSON small)
    summary_metas = [{k: v for k, v in m.items()
                      if k not in ("blend_oof", "blend_test")} for m in metas]
    out = dict(
        lb_best_oof=float(base),
        meta_alpha_grid=summary_metas,
        best_meta=dict(a=best_meta["a"], oof=float(best_meta["oof"])),
        meta_enhanced_oof=float(base_mbal),
        combo_sweep=combo_rows,
        best_combo=best_combo,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1b_combined_results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote scripts/artifacts/tier1b_combined_results.json")


if __name__ == "__main__":
    main()
