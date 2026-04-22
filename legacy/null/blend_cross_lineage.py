"""Cross-lineage blend: our greedy winner × main branch's hybrid_lgbmxgb_blend.

Our greedy (OOF 0.97375, LB 0.97296) and main's hybrid_lgbmxgb_blend
(OOF 0.97362) come from DIFFERENT ensemble lineages:
  - Ours:  log-blend of hybrid_v3 + routed_v3 + spec_678 (3-way specialist)
  - Main's: 0.75*hybrid_v3 + 0.25*(LGBM-dist*0.45 + XGB-dist*0.55)
           (hybrid + 2-way model-family blend)

Blending TWO independent blends is compound diversity. Expected lift
small-to-none on OOF (both are already at the ~0.97362-0.97375 level)
but could transfer better on LB if the error patterns differ.

Also adds main's `oof_xgb_vanilla_dist` (XGB-dist, no routing, 0.97304)
and `oof_lgbm_te_orig` (LGBM-dist + TE, 0.97270) to the candidate pool
in case they contribute as 3rd / 4th legs.

Gates submission on beating the current LB best (OOF 0.97375) by
>= 3e-4 on OOF — otherwise not worth an LB probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ID, TARGET = "id", "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def fast_bal_acc(y, pred, cc):
    m = pred == y
    hit = np.array([m[y == k].sum() for k in range(3)])
    return float((hit / np.maximum(cc, 1)).mean())


def tune_log_bias(oof, y, prior, cc):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = fast_bal_acc(y, (log_oof + bias).argmax(axis=1), cc)
    gd = np.linspace(-3.0, 3.0, 61)
    gh = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = gh if k == 2 else gd
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(axis=1), cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def per_class_recall(y, pred, cc):
    m = pred == y
    return {CLASSES[k]: float(m[y == k].sum() / cc[k]) for k in range(3)}


def log_blend(oofs, weights):
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, 1e-9, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def evaluate(oof, y, prior, cc, label):
    bias, tuned = tune_log_bias(oof, y, prior, cc)
    pred = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    pcr = per_class_recall(y, pred, cc)
    print(f"  {label:50s}  bal={tuned:.5f}  rec_H={pcr['High']:.4f}  "
          f"bias={np.round(bias, 3).tolist()}")
    return bias, tuned, pcr


def main():
    tr = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    cc = np.bincount(y, minlength=3)

    ours = np.load(ART / "oof_blend_greedy_final.npy")
    ours_te = np.load(ART / "test_blend_greedy_final.npy")
    main_ = np.load(ART / "oof_hybrid_lgbmxgb_blend.npy")
    main_te = np.load(ART / "test_hybrid_lgbmxgb_blend.npy")
    hv3 = np.load(ART / "oof_xgb_hybrid_v3.npy")
    hv3_te = np.load(ART / "test_xgb_hybrid_v3.npy")

    print("\n--- standalones ---")
    evaluate(ours, y, prior, cc, "our greedy (OOF 0.97375, LB 0.97296)")
    evaluate(main_, y, prior, cc, "main's hybrid_lgbmxgb_blend (OOF 0.97362)")
    evaluate(hv3, y, prior, cc, "xgb_hybrid_v3 (reference)")

    # pairwise sweep: ours × main's
    print("\n--- pairwise log-blend: our_greedy × main_lgbmxgb ---")
    best = {"tuned": -1.0}
    for a in np.linspace(0.0, 1.0, 21):
        blend = log_blend([ours, main_], [a, 1 - a])
        blend_te = log_blend([ours_te, main_te], [a, 1 - a])
        bias, tuned = tune_log_bias(blend, y, prior, cc)
        pred = (np.log(np.clip(blend, 1e-9, 1.0)) + bias).argmax(axis=1)
        pcr = per_class_recall(y, pred, cc)
        print(f"  w_ours={a:.2f}  w_main={1-a:.2f}  bal={tuned:.5f}  "
              f"rec_H={pcr['High']:.4f}  bias={np.round(bias, 3).tolist()}")
        if tuned > best["tuned"]:
            best = {"tuned": tuned, "a": a, "oof": blend, "test": blend_te,
                    "bias": bias, "recall": pcr}

    print(f"\n  PAIRWISE BEST: w_ours={best['a']:.2f}  tuned={best['tuned']:.5f}")

    # also try a 3-way with hv3
    print("\n--- 3-way log-blend: ours × main × hv3 ---")
    best3 = {"tuned": -1.0}
    for w_o in np.linspace(0.1, 0.8, 8):
        for w_m in np.linspace(0.1, 0.8 - w_o + 1e-9, 7):
            w_h = 1.0 - w_o - w_m
            if w_h < 0.05:
                continue
            blend = log_blend([ours, main_, hv3], [w_o, w_m, w_h])
            blend_te = log_blend([ours_te, main_te, hv3_te], [w_o, w_m, w_h])
            bias, tuned = tune_log_bias(blend, y, prior, cc)
            if tuned > best3["tuned"]:
                pred = (np.log(np.clip(blend, 1e-9, 1.0)) + bias).argmax(axis=1)
                pcr = per_class_recall(y, pred, cc)
                best3 = {"tuned": tuned, "w": (w_o, w_m, w_h),
                         "oof": blend, "test": blend_te,
                         "bias": bias, "recall": pcr}
    print(f"  3-WAY BEST: w=({best3['w'][0]:.2f},{best3['w'][1]:.2f},"
          f"{best3['w'][2]:.2f})  tuned={best3['tuned']:.5f}  "
          f"rec_H={best3['recall']['High']:.4f}  bias={np.round(best3['bias'], 3).tolist()}")

    # summary
    current_best = 0.97375
    threshold = current_best + 3e-4
    print(f"\n=== SUMMARY (beat {current_best} + 3e-4 = {threshold} to warrant LB probe) ===")
    cands = [
        ("cross_lineage_pairwise", best["tuned"], best["oof"], best["test"], best["bias"], best["recall"]),
        ("cross_lineage_3way",     best3["tuned"], best3["oof"], best3["test"], best3["bias"], best3["recall"]),
    ]
    cands.sort(key=lambda r: r[1], reverse=True)
    for name, sc, _, _, _, rec in cands:
        flag = "✓" if sc >= threshold else " "
        print(f"  {flag}  {name:30s}  bal={sc:.5f}  Δ={sc-current_best:+.5f}  "
              f"rec_H={rec['High']:.4f}")

    # write any submission that clears threshold
    for name, sc, oof_b, test_b, bias, _ in cands:
        if sc < threshold:
            continue
        pred = (np.log(np.clip(test_b, 1e-9, 1.0)) + bias).argmax(axis=1)
        fname = f"submission_blend_{name}.csv"
        pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred]}).to_csv(
            SUB / fname, index=False
        )
        print(f"  wrote {fname}")

    # always write the winner for review regardless of threshold
    best_cand = cands[0]
    if best_cand[2] is not None:
        pred = (np.log(np.clip(best_cand[3], 1e-9, 1.0)) + best_cand[4]).argmax(axis=1)
        fname = "submission_blend_cross_lineage_best.csv"
        pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred]}).to_csv(
            SUB / fname, index=False
        )
        print(f"  wrote {fname}  (winner={best_cand[0]} OOF={best_cand[1]:.5f})")

    with open(ART / "blend_cross_lineage_results.json", "w") as f:
        json.dump({
            "pairwise_best": {"a": best["a"], "tuned": best["tuned"],
                              "bias": best["bias"].tolist(),
                              "rec_H": best["recall"]["High"]},
            "three_way_best": {"w": list(best3["w"]), "tuned": best3["tuned"],
                               "bias": best3["bias"].tolist(),
                               "rec_H": best3["recall"]["High"]},
            "current_lb_best": {"name": "our_greedy", "oof": 0.97375, "lb": 0.97296},
        }, f, indent=2)
    print("\n--- done; NO auto-submission to LB ---")


if __name__ == "__main__":
    main()
