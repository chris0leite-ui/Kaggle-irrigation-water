"""Blend analysis: does xgb_spec_36 add signal on top of LB-best 3-way?

LB-best 3-way (LB 0.98005, OOF 0.98029):
  w_recipe=0.25 + w_pseudo_s1=0.35 + w_pseudo_s7=0.40 log-blend at
  recipe's tuned bias [1.4324, 1.4689, 3.4008].

Two integration strategies (both fixed-bias, no bias retune per
2026-04-21 binhigh rule):
  A) OVERRIDE: on score-{3,6} rows, replace 3-way's probs with
     spec-36's. Off-spec rows keep 3-way as-is.
  B) SOFT-MIX on spec rows only: log-blend 3-way × spec-36 at
     several alpha values on spec rows; off-spec rows keep 3-way.

Jaccard + error-count gate on whole-OOF predictions. LB-probe only if
Δ >= +0.0002 AND pattern is monotone (not single-point fluke).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CLASSES, add_distance_features, fast_bal_acc,
                    load_oof_pair, log_blend)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def eval_at_bias(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> tuple[float, np.ndarray]:
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    return fast_bal_acc(y, pred, class_counts=cc), pred


def per_class_recall(pred: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for k, name in enumerate(CLASSES):
        m = (y == k)
        out[name] = float((pred[m] == k).sum() / max(m.sum(), 1))
    return out


def jaccard(a_wrong: np.ndarray, b_wrong: np.ndarray) -> float:
    u = int((a_wrong | b_wrong).sum())
    return int((a_wrong & b_wrong).sum()) / u if u > 0 else 0.0


def main() -> None:
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    test = pd.read_csv("data/test.csv")

    # dgp_score for spec masks — must match spec-36's definition (from common.add_distance_features)
    tr_fe = add_distance_features(train[["Soil_Moisture", "Rainfall_mm",
                                         "Temperature_C", "Wind_Speed_kmh",
                                         "Mulching_Used", "Crop_Growth_Stage"]])
    te_fe = add_distance_features(test[["Soil_Moisture", "Rainfall_mm",
                                        "Temperature_C", "Wind_Speed_kmh",
                                        "Mulching_Used", "Crop_Growth_Stage"]])
    tr_score = tr_fe["dgp_score"].to_numpy()
    te_score = te_fe["dgp_score"].to_numpy()
    tr_spec = np.isin(tr_score, (3, 6))
    te_spec = np.isin(te_score, (3, 6))
    log(f"train spec rows: {tr_spec.sum()} ({tr_spec.mean()*100:.2f}%)")
    log(f"test  spec rows: {te_spec.sum()} ({te_spec.mean()*100:.2f}%)")

    # recipe bias = LB-best anchor
    res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(res["log_bias"], dtype=np.float64)
    log(f"anchor bias: {bias.round(4).tolist()}")

    # Reconstruct LB-best 3-way
    comps = {
        "recipe": load_oof_pair("recipe_full_te"),
        "p_s1":   load_oof_pair("recipe_pseudolabel"),
        "p_s7":   load_oof_pair("recipe_pseudolabel_seed7labeler"),
    }
    ws = np.array([0.25, 0.35, 0.40])
    lb_oof = log_blend([comps[k][0] for k in comps], ws)
    lb_test = log_blend([comps[k][1] for k in comps], ws)
    lb_ba, lb_pred = eval_at_bias(lb_oof, y, bias)
    lb_wrong = (lb_pred != y)
    log(f"LB-best 3-way OOF bal_acc = {lb_ba:.5f}  errs={lb_wrong.sum()}")
    log(f"  per-class recall: {per_class_recall(lb_pred, y)}")
    # spec-domain slice
    spec_y = y[tr_spec]
    spec_lb_pred = lb_pred[tr_spec]
    spec_lb_ba = fast_bal_acc(spec_y, spec_lb_pred)
    spec_lb_errs = int((spec_lb_pred != spec_y).sum())
    log(f"  LB-best on spec-domain only: bal={spec_lb_ba:.5f}  errs={spec_lb_errs}/{tr_spec.sum()}")

    # Load spec-36 (sparse carrier)
    spec_oof = np.load(ART / "oof_xgb_spec_36.npy")
    spec_test = np.load(ART / "test_xgb_spec_36.npy")
    spec_argmax_spec = spec_oof[tr_spec].argmax(1)
    spec_dom_ba = fast_bal_acc(spec_y, spec_argmax_spec)
    spec_dom_errs = int((spec_argmax_spec != spec_y).sum())
    log(f"  spec-36 on spec-domain only: bal={spec_dom_ba:.5f}  errs={spec_dom_errs}/{tr_spec.sum()}")
    log(f"  delta on spec-domain = {spec_dom_ba - spec_lb_ba:+.5f}")

    # --- Strategy A: OVERRIDE on spec rows ---
    ov_oof = lb_oof.copy()
    ov_oof[tr_spec] = spec_oof[tr_spec]
    ov_test = lb_test.copy()
    ov_test[te_spec] = spec_test[te_spec]
    ov_ba, ov_pred = eval_at_bias(ov_oof, y, bias)
    ov_wrong = (ov_pred != y)
    log(f"\n[A] OVERRIDE: OOF bal={ov_ba:.5f}  errs={ov_wrong.sum()}  Δ={ov_ba-lb_ba:+.5f}")
    log(f"  per-class: {per_class_recall(ov_pred, y)}")
    log(f"  Jaccard(err) vs LB-best: {jaccard(lb_wrong, ov_wrong):.4f}")

    # --- Strategy B: soft-mix on spec rows at fixed bias ---
    log("\n[B] SOFT-MIX alpha sweep on spec rows only (log-blend spec × 3way):")
    best_alpha = 0.0
    best_ba = lb_ba
    best_oof = lb_oof
    best_test = lb_test
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        mix_oof = lb_oof.copy()
        mix_test = lb_test.copy()
        # log_blend on spec rows only: mix = (1-a)*lb + a*spec (both prob, then renorm)
        w2 = np.array([1.0 - alpha, alpha])
        mix_oof[tr_spec] = log_blend([lb_oof[tr_spec], spec_oof[tr_spec]], w2)
        mix_test[te_spec] = log_blend([lb_test[te_spec], spec_test[te_spec]], w2)
        ba, pred = eval_at_bias(mix_oof, y, bias)
        delta = ba - lb_ba
        log(f"  alpha={alpha:.2f}  OOF={ba:.5f}  Δ={delta:+.5f}  errs={(pred!=y).sum()}  "
            f"Jacc={jaccard(lb_wrong, pred!=y):.4f}")
        if ba > best_ba:
            best_ba, best_alpha, best_oof, best_test = ba, alpha, mix_oof, mix_test

    log(f"\n[B] best alpha={best_alpha}  OOF={best_ba:.5f}  Δ={best_ba-lb_ba:+.5f}")

    # Emit submissions for OVERRIDE and best-alpha SOFT-MIX if gate clears
    EMIT_GATE = 0.0005
    sample = pd.read_csv("data/sample_submission.csv")
    emitted = []
    def emit_csv(test_probs: np.ndarray, tag: str, delta: float) -> None:
        pred_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(1)
        labels = [CLASSES[i] for i in pred_idx]
        out = sample.copy()
        out["Irrigation_Need"] = labels
        path = SUB / f"submission_spec36_{tag}.csv"
        out.to_csv(path, index=False)
        emitted.append({"tag": tag, "path": str(path), "delta": delta})
        log(f"    emitted {path}  (OOF Δ={delta:+.5f})")

    log("\n=== Emit decisions (gate +0.00050) ===")
    if ov_ba - lb_ba >= EMIT_GATE:
        emit_csv(ov_test, "override", ov_ba - lb_ba)
    else:
        log(f"  OVERRIDE below gate ({ov_ba-lb_ba:+.5f} < +{EMIT_GATE}); no csv")
    if best_ba - lb_ba >= EMIT_GATE and best_alpha > 0:
        emit_csv(best_test, f"softmix_a{int(best_alpha*100):03d}", best_ba - lb_ba)
    else:
        log(f"  SOFT-MIX below gate ({best_ba-lb_ba:+.5f} < +{EMIT_GATE}); no csv")

    with open(ART / "blend_spec_36_results.json", "w") as f:
        json.dump({
            "lb_best_oof": float(lb_ba),
            "lb_best_errs": int(lb_wrong.sum()),
            "lb_best_on_spec_domain": {"bal": float(spec_lb_ba), "errs": spec_lb_errs},
            "spec36_on_spec_domain": {"bal": float(spec_dom_ba), "errs": spec_dom_errs},
            "override": {"oof": float(ov_ba), "delta": float(ov_ba - lb_ba),
                         "errs": int(ov_wrong.sum()),
                         "jaccard_vs_lb": float(jaccard(lb_wrong, ov_wrong))},
            "best_softmix": {"alpha": float(best_alpha), "oof": float(best_ba),
                             "delta": float(best_ba - lb_ba)},
            "emitted": emitted,
        }, f, indent=2)
    log(f"saved blend_spec_36_results.json")


if __name__ == "__main__":
    main()
