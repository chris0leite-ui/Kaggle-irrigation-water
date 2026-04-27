"""Experiment #3: 3-meta L3 weighted average (XGB-meta + MLP-meta + LR-meta-v2).

Pure blend math — no retraining. All three meta OOFs already on disk.
Iso-cals each, then sweeps over a 3-simplex of L3 weights, then log-blends
into the LB-best 3-stack at α-sweep.

Compares to B (XGB+MLP only) which scored LB 0.98091 (gap +0.00027).
A 3-meta L3 has a chance to compound positive transfer if all three
metas have positive (but bounded) LB carryover, AND the per-class
trade-off remains favorable.

Outputs:
  oof_three_meta_l3_<best>.npy / test_three_meta_l3_<best>.npy
  three_meta_l3_results.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
DATA = Path("data")
TARGET = "Irrigation_Need"
EPS = 1e-12


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("loading LB-best 3-stack + 3 metas")
    lb_oof, lb_test = build_lbbest_stack(y)
    xgb_meta_oof = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    xgb_meta_test = _normed(np.load(ART / "test_xgb_metastack.npy"))
    mlp_meta_oof = _normed(np.load(ART / "oof_mlp_metastack.npy"))
    mlp_meta_test = _normed(np.load(ART / "test_mlp_metastack.npy"))
    lr_meta_oof = _normed(np.load(ART / "oof_lr_metastack_v2.npy"))
    lr_meta_test = _normed(np.load(ART / "test_lr_metastack_v2.npy"))

    log("iso-calibrating each meta vs y")
    xgb_iso_oof, xgb_iso_test = iso_cal(xgb_meta_oof, xgb_meta_test, y)
    mlp_iso_oof, mlp_iso_test = iso_cal(mlp_meta_oof, mlp_meta_test, y)
    lr_iso_oof, lr_iso_test = iso_cal(lr_meta_oof, lr_meta_test, y)

    log(f"  xgb_iso @bias = {bal(xgb_iso_oof, y):.5f}")
    log(f"  mlp_iso @bias = {bal(mlp_iso_oof, y):.5f}")
    log(f"  lr_iso  @bias = {bal(lr_iso_oof, y):.5f}")

    # LB-best 4-stack reference (anchor).
    lb4_oof = log_blend([lb_oof, xgb_iso_oof], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb_test, xgb_iso_test], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_oof, y)
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")

    # B reference (XGB + MLP only, W_MLP=0.5 known LB-validated).
    b_l3_oof = 0.5 * xgb_iso_oof + 0.5 * mlp_iso_oof
    b_l3_oof = b_l3_oof / b_l3_oof.sum(1, keepdims=True)
    b_blend_oof = log_blend([lb_oof, b_l3_oof], np.array([0.5, 0.5]))
    b_bal = bal(b_blend_oof, y)
    log(f"  B reference (XGB+MLP L3, α=0.5)  = {b_bal:.5f}  (LB-validated 0.98091)")

    # Anchor PCR.
    pred_anchor = (np.log(np.clip(lb4_oof, EPS, 1)) + BIAS).argmax(1)
    pcr_anchor = np.array([(pred_anchor[y == k] == k).mean() for k in range(3)])

    # 3-simplex L3 weight grid (XGB / MLP / LR sum to 1, step 0.1).
    log(f"\n=== 3-meta L3 sweep (XGB×MLP×LR weights, log-blend into LB-best 3-stack) ===")
    log(f"{'w_xgb':>6} {'w_mlp':>6} {'w_lr':>6} {'best_α':>8} {'best OOF':>10} {'Δ vs 4st':>10}  PCR pass")
    rows = []
    alphas = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    weights_grid = []
    step = 0.1
    for wx in np.arange(0.0, 1.01, step):
        for wm in np.arange(0.0, 1.01 - wx, step):
            wl = 1.0 - wx - wm
            if wl < -1e-9 or wl > 1.0 + 1e-9:
                continue
            wl = max(wl, 0.0)
            weights_grid.append((round(wx, 2), round(wm, 2), round(wl, 2)))

    for wx, wm, wl in weights_grid:
        l3_oof = wx * xgb_iso_oof + wm * mlp_iso_oof + wl * lr_iso_oof
        l3_oof = l3_oof / l3_oof.sum(1, keepdims=True)
        # Sweep α; pick best-OOF that PCR-passes.
        best_a = None
        best_d = -1
        best_oof = -1
        best_pcr = None
        for a in alphas:
            blend = log_blend([lb_oof, l3_oof], np.array([1 - a, a]))
            b = bal(blend, y)
            d = b - lb4_bal
            pred = (np.log(np.clip(blend, EPS, 1)) + BIAS).argmax(1)
            pcr = np.array([(pred[y == k] == k).mean() for k in range(3)])
            pcr_delta = pcr - pcr_anchor
            pcr_pass = bool((pcr_delta >= -5e-4).all())
            if pcr_pass and d > best_d:
                best_a = a; best_d = d; best_oof = b; best_pcr = pcr_delta
        if best_a is None:
            continue
        rows.append({
            "w_xgb": wx, "w_mlp": wm, "w_lr": wl,
            "best_alpha": best_a, "best_oof": float(best_oof),
            "delta_4stack": float(best_d),
            "pcr_delta": best_pcr.tolist(),
        })

    # Sort by delta descending.
    rows.sort(key=lambda r: -r["delta_4stack"])
    log(f"\nTop-10 weight combinations (gate-passing only):")
    for r in rows[:10]:
        log(f"  XGB={r['w_xgb']:.2f}  MLP={r['w_mlp']:.2f}  LR={r['w_lr']:.2f}  "
            f"α={r['best_alpha']:.2f}  OOF={r['best_oof']:.5f}  "
            f"Δ={r['delta_4stack']:+.5f}  "
            f"PCRΔ=[{r['pcr_delta'][0]:+.4f},{r['pcr_delta'][1]:+.4f},{r['pcr_delta'][2]:+.4f}]")

    # Pick the winner.
    best = rows[0] if rows else None
    if best:
        gate_pass = bool(best["delta_4stack"] >= 2e-4)
        log(f"\nBEST: w_xgb={best['w_xgb']:.2f}  w_mlp={best['w_mlp']:.2f}  "
            f"w_lr={best['w_lr']:.2f}  α={best['best_alpha']:.2f}  "
            f"Δ={best['delta_4stack']:+.5f}")
        log(f"GATE: {'PASS' if gate_pass else 'FAIL'} (need Δ ≥ +2e-4)")

        # Save the best blend's OOF + test for downstream use.
        wx, wm, wl, a = best["w_xgb"], best["w_mlp"], best["w_lr"], best["best_alpha"]
        l3_oof = wx * xgb_iso_oof + wm * mlp_iso_oof + wl * lr_iso_oof
        l3_oof = l3_oof / l3_oof.sum(1, keepdims=True)
        l3_test = wx * xgb_iso_test + wm * mlp_iso_test + wl * lr_iso_test
        l3_test = l3_test / l3_test.sum(1, keepdims=True)
        blend_oof = log_blend([lb_oof, l3_oof], np.array([1 - a, a]))
        blend_test = log_blend([lb_test, l3_test], np.array([1 - a, a]))
        np.save(ART / "oof_three_meta_l3.npy", blend_oof.astype(np.float32))
        np.save(ART / "test_three_meta_l3.npy", blend_test.astype(np.float32))
        log(f"  saved oof/test_three_meta_l3.npy at best blend")
    else:
        gate_pass = False
        log(f"\nNo gate-passing combinations found.")

    out = dict(
        anchor_lb4=float(lb4_bal),
        b_reference=float(b_bal),
        meta_iso_standalone={
            "xgb": float(bal(xgb_iso_oof, y)),
            "mlp": float(bal(mlp_iso_oof, y)),
            "lr":  float(bal(lr_iso_oof, y)),
        },
        n_grid_evaluated=len(weights_grid),
        n_gate_passing=len(rows),
        top10=rows[:10],
        best=best,
        gate_pass=gate_pass,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "three_meta_l3_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote three_meta_l3_results.json")


if __name__ == "__main__":
    main()
