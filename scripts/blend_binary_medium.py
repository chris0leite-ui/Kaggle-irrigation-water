"""W3 blend analyzer: logit-add P(Medium) into LB-best 3-way, fixed bias.

Binhigh (2026-04-21) retuned log-bias after each blend variant and the
+0.00036 OOF lift evaporated on LB (−0.00084). W3 must be fixed-bias only.

Three parameterizations:
  - prob_mix  : (1-w)*p3[:,1] + w*p_med  (column-only update + renorm)
  - geo_mix   : p3[:,1]^(1-w) * p_med^w  (geometric blend on Medium col)
  - logit_add : p3[:,1] += lam * logit(p_med)  (log-space bump + softmax)

Gate: monotone positive AND peak Δ ≥ +0.0002. If both hold, emit CSV.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASSES, fast_bal_acc, load_oof_pair, log_blend


ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
EMIT_GATE = 0.0002  # below fold-std; plan W3 threshold was +0.0002


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def eval_at_bias(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray]:
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    return fast_bal_acc(y, pred, class_counts=cc), pred


def per_class_recall(pred: np.ndarray, y: np.ndarray) -> dict:
    return {CLASSES[k]: float(((pred == k) & (y == k)).sum() / max((y == k).sum(), 1))
            for k in range(3)}


def prob_mix(p3: np.ndarray, pmed: np.ndarray, w: float) -> np.ndarray:
    """Mix on Medium column only; rescale Low/High so rows sum to 1."""
    new = p3.copy()
    new_med = np.clip((1 - w) * p3[:, 1] + w * pmed, 1e-9, 1 - 1e-9)
    denom = np.clip(1 - p3[:, 1], 1e-9, 1.0)
    scale = (1 - new_med) / denom
    new[:, 0] = p3[:, 0] * scale
    new[:, 1] = new_med
    new[:, 2] = p3[:, 2] * scale
    new /= new.sum(1, keepdims=True)
    return new


def geo_mix(p3: np.ndarray, pmed: np.ndarray, w: float) -> np.ndarray:
    new = p3.copy()
    pm = np.clip(p3[:, 1], 1e-9, 1 - 1e-9) ** (1 - w) * np.clip(pmed, 1e-9, 1 - 1e-9) ** w
    pm = np.clip(pm, 1e-9, 1 - 1e-9)
    denom = np.clip(1 - p3[:, 1], 1e-9, 1.0)
    scale = (1 - pm) / denom
    new[:, 0] = p3[:, 0] * scale
    new[:, 1] = pm
    new[:, 2] = p3[:, 2] * scale
    new /= new.sum(1, keepdims=True)
    return new


def logit_add(p3: np.ndarray, pmed: np.ndarray, lam: float) -> np.ndarray:
    logp = np.log(np.clip(p3, 1e-9, 1.0))
    lg = np.log(np.clip(pmed, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - pmed, 1e-9, 1.0))
    logp[:, 1] += lam * lg
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def is_monotone(points: list[tuple[float, float]], baseline_y: float) -> bool:
    """Check the sweep is monotone-positive: strictly increasing from α=0 to
    the peak, with all Δ ≥ 0 up to and including the peak."""
    sorted_pts = sorted(points, key=lambda t: t[0])
    peak_idx = int(np.argmax([y for _, y in sorted_pts]))
    if peak_idx == 0:
        return False
    for i in range(1, peak_idx + 1):
        if sorted_pts[i][1] < baseline_y - 1e-8:
            return False
    return True


def main() -> None:
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(res["log_bias"], dtype=np.float64)
    log(f"fixed anchor bias = {bias.round(4).tolist()}")

    comps = [load_oof_pair("recipe_full_te"),
             load_oof_pair("recipe_pseudolabel"),
             load_oof_pair("recipe_pseudolabel_seed7labeler")]
    ws = np.array([0.25, 0.35, 0.40])
    lb_oof = log_blend([c[0] for c in comps], ws)
    lb_test = log_blend([c[1] for c in comps], ws)
    lb_ba, lb_pred = eval_at_bias(lb_oof, y, bias)
    lb_rec = per_class_recall(lb_pred, y)
    log(f"LB-best 3-way: bal={lb_ba:.5f}  errs={(lb_pred!=y).sum()}  rec={lb_rec}")

    oof_bin = np.load(ART / "oof_xgb_bin_medium.npy")
    test_bin = np.load(ART / "test_xgb_bin_medium.npy")
    log(f"binary-Medium OOF mean={oof_bin.mean():.4f}  Medium prior={(y==1).mean():.4f}")

    sample = pd.read_csv("data/sample_submission.csv")
    summary = {"lb_best_ba": float(lb_ba), "lb_best_rec": lb_rec, "sweeps": {}, "emitted": []}

    for kind, fn, grid in [
        ("prob_mix",  prob_mix,  np.linspace(0.0, 0.80, 17)),
        ("geo_mix",   geo_mix,   np.linspace(0.0, 0.80, 17)),
        ("logit_add", logit_add, np.linspace(-0.20, 1.00, 25)),
    ]:
        log(f"\n=== {kind} sweep @ fixed bias ===")
        pts = []
        for g in grid:
            blend_oof = fn(lb_oof, oof_bin, g)
            ba, pred = eval_at_bias(blend_oof, y, bias)
            pts.append((float(g), float(ba)))
            log(f"  g={g:+.3f}  ba={ba:.5f}  Δ={ba-lb_ba:+.5f}  errs={(pred!=y).sum()}")
        summary["sweeps"][kind] = pts
        # Emit check
        peak_g, peak_ba = max(pts, key=lambda t: t[1])
        delta = peak_ba - lb_ba
        monot = is_monotone(pts, lb_ba)
        log(f"  peak g={peak_g:+.3f}  ba={peak_ba:.5f}  Δ={delta:+.5f}  monotone_positive={monot}")
        if delta >= EMIT_GATE and monot and peak_g != 0.0:
            blend_test = fn(lb_test, test_bin, peak_g)
            pred_idx = (np.log(np.clip(blend_test, 1e-9, 1.0)) + bias).argmax(1)
            out = sample.copy()
            out["Irrigation_Need"] = [CLASSES[i] for i in pred_idx]
            path = SUB / f"submission_binmedium_{kind}_g{int(peak_g*1000):+05d}.csv"
            out.to_csv(path, index=False)
            summary["emitted"].append({"kind": kind, "g": float(peak_g),
                                       "delta": float(delta), "path": str(path)})
            log(f"  EMITTED {path}")
        else:
            log(f"  below gate ({delta:+.5f} < {EMIT_GATE:+.5f}) or non-monotone or g=0; no csv")

    with open(ART / "blend_binary_medium_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"saved blend_binary_medium_results.json")


if __name__ == "__main__":
    main()
