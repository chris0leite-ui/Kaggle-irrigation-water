"""Blend-gate for J1-v2 (recipe-feature base leaf-OTE meta-stacker).

Loads oof_leaf_ote_meta_v2.npy + test_leaf_ote_meta_v2.npy and runs
the standard fixed-bias α-sweep vs LB-best 4-stack with per-class
recall guardrail. Auto-emits submission iff Δ ≥ +2e-4 OOF AND per-
class recall ≥ 4-stack − 5e-4 AND Jaccard < 0.97.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    build_lbbest_stack, iso_cal, log_blend, normed,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
LB_BIAS = np.array([1.4324, 1.4689, 3.4008])


def fixed_bias_argmax(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(probs, 1e-9, 1)) + bias).argmax(1)


def err_jaccard(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    ea = (a != y); eb = (b != y)
    return float((ea & eb).sum() / max((ea | eb).sum(), 1))


def main() -> None:
    train = pd.read_csv("data/train.csv")
    train[TARGET] = train[TARGET].map(CLS_MAP).astype(np.int32)
    y = train[TARGET].to_numpy()
    print(f"loaded train y shape={y.shape}, prior={np.bincount(y) / len(y)}")

    oof_meta = np.load(ART / "oof_leaf_ote_meta_v2.npy").astype(np.float32)
    test_meta = np.load(ART / "test_leaf_ote_meta_v2.npy").astype(np.float32)
    print(f"loaded leaf-OTE-v2 meta: oof={oof_meta.shape} test={test_meta.shape}")

    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    ms_t = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    ms_o_iso, ms_t_iso = iso_cal(normed(ms_o), normed(ms_t), y)
    s4_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    s4_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))
    s4_pred = fixed_bias_argmax(s4_o, LB_BIAS)
    s4_argmax_bal = balanced_accuracy_score(y, s4_pred)
    s4_pcr = recall_score(y, s4_pred, average=None)
    s4_errs = int((s4_pred != y).sum())
    print(f"LB-best 4-stack: argmax_bal={s4_argmax_bal:.5f}  errs={s4_errs}  "
          f"recL={s4_pcr[0]:.4f} recM={s4_pcr[1]:.4f} recH={s4_pcr[2]:.4f}")

    prior = np.bincount(y) / len(y)
    own_bias, own_tuned = tune_log_bias(oof_meta, y, prior)
    fb_pred = fixed_bias_argmax(oof_meta, LB_BIAS)
    fb_argmax_bal = balanced_accuracy_score(y, fb_pred)
    fb_errs = int((fb_pred != y).sum())
    fb_pcr = recall_score(y, fb_pred, average=None)
    print(f"leaf-OTE-v2 standalone @ own bias: tuned={own_tuned:.5f}  bias={own_bias.tolist()}")
    print(f"leaf-OTE-v2 standalone @ LB bias:  argmax_bal={fb_argmax_bal:.5f}  errs={fb_errs}  "
          f"recL={fb_pcr[0]:.4f} recM={fb_pcr[1]:.4f} recH={fb_pcr[2]:.4f}")
    j_vs_4 = err_jaccard(fb_pred, s4_pred, y)
    print(f"Jaccard(leaf-OTE-v2, LB-best 4-stack) @ LB bias = {j_vs_4:.4f}")

    meta_iso_o, meta_iso_t = iso_cal(normed(oof_meta), normed(test_meta), y)
    print("\n=== α-sweep: LB-best 4-stack × leaf-OTE-v2-meta_iso (fixed bias) ===")
    print("alpha   tuned       Δ          errs   recL    recM    recH   Jaccard")
    rows = []
    for a in [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend_o = log_blend([s4_o, meta_iso_o], np.array([1 - a, a]))
        blend_t = log_blend([s4_t, meta_iso_t], np.array([1 - a, a]))
        pred = fixed_bias_argmax(blend_o, LB_BIAS)
        b = balanced_accuracy_score(y, pred)
        pcr = recall_score(y, pred, average=None)
        errs = int((pred != y).sum())
        jac = err_jaccard(pred, s4_pred, y)
        delta = b - s4_argmax_bal
        emit = (delta >= 2e-4 and jac < 0.97 and
                all(pcr[c] >= s4_pcr[c] - 5e-4 for c in range(3)))
        flag = " EMIT" if emit else ""
        print(f"{a:5.3f}  {b:.5f}  {delta:+.5f}  {errs:5d}  "
              f"{pcr[0]:.4f}  {pcr[1]:.4f}  {pcr[2]:.4f}  {jac:.4f}{flag}")
        rows.append(dict(alpha=a, tuned=float(b), delta=float(delta),
                         errs=errs, recL=float(pcr[0]), recM=float(pcr[1]),
                         recH=float(pcr[2]), jaccard=jac, emit=bool(emit),
                         blend_test=blend_t))

    best = max(rows, key=lambda r: r["delta"])
    if best["emit"]:
        test_csv = pd.DataFrame({"id": pd.read_csv("data/test.csv")["id"].values,
                                 "Irrigation_Need": [IDX2CLS[i] for i in
                                                     fixed_bias_argmax(best["blend_test"], LB_BIAS)]})
        out_path = SUB / f"submission_leaf_ote_v2_meta_a{int(best['alpha']*1000):03d}.csv"
        test_csv.to_csv(out_path, index=False)
        print(f"\nEMIT: best α={best['alpha']:.3f} Δ={best['delta']:+.5f} → {out_path}")
    else:
        print(f"\nNO EMIT: best α={best['alpha']:.3f} Δ={best['delta']:+.5f} "
              f"(below +2e-4 gate or guardrail fail)")

    rows_clean = [{k: v for k, v in r.items() if k != "blend_test"} for r in rows]
    summary = dict(
        s4_argmax_bal=float(s4_argmax_bal), s4_errs=s4_errs, s4_pcr=s4_pcr.tolist(),
        leaf_ote_v2_tuned_own_bias=float(own_tuned),
        leaf_ote_v2_argmax_lb_bias=float(fb_argmax_bal), leaf_ote_v2_errs=fb_errs,
        leaf_ote_v2_pcr=fb_pcr.tolist(), jaccard_vs_4stack=j_vs_4,
        sweep=rows_clean,
    )
    (ART / "leaf_ote_v2_blend_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
