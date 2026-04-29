"""Blend gate + natural-cal drift diagnostic for the natural-cal RealMLP probe.

Runs AFTER `kaggle kernels output chrisleitescha/irrigation-realmlp-natural
-p scripts/artifacts/` downloads the 3 outputs (oof / test / json).

Tests three hypotheses jointly:

  (1) Natural-cal verdict — does the (no class-balance + ORIG_ROW_WEIGHT=0.5
      + TE cv=5) regime produce raw probs at macro-recall optimum?
      Diagnostic: drift = tuned_bias - (-log(prior)). Natural-cal target
      is |drift| <= 0.3 each class. Baseline RealMLP n_ens=1 drift was
      [0.70, 0.50, 0.00].

  (2) Magnitude-trap verdict — does natural-cal training shrink the +358
      err overshoot from baseline RealMLP n_ens=1 to <= +0.05*anchor?
      Anchor = LB-best 4-stack (PRIMARY at LB 0.98094, errs ~9415).

  (3) Bank-add viability — is this a viable input to the RF natural meta
      bank that produced LB 0.98129? Gate: standalone tuned >= 0.974,
      Jaccard < 0.80 vs RF natural standalone.

If all three pass, the next stage is to retrain sklearn_rf_meta_natural
with this 8th component added to the 7-component bank.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, BIAS

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _errmask(probs, y, bias):
    return (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1) != y


def _eval(probs, y, bias):
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    return fast_bal_acc(y, pred, class_counts=cc), int((pred != y).sum())


def _per_class_recall(probs, y, bias):
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    out = []
    for k in range(3):
        out.append(((pred == k) & (y == k)).sum() / max(cc[k], 1))
    return np.array(out)


def main():
    oof_path = ART / "oof_realmlp_natural.npy"
    if not oof_path.exists():
        raise SystemExit(
            f"{oof_path} not found — run "
            "`kaggle kernels output chrisleitescha/irrigation-realmlp-natural "
            "-p scripts/artifacts/` first"
        )

    log("=== loading inputs ===")
    nat_oof = np.load(oof_path)
    nat_test = np.load(ART / "test_realmlp_natural.npy")
    base_oof = np.load(ART / "oof_realmlp.npy")  # baseline n_ens=1
    base_test = np.load(ART / "test_realmlp.npy")
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy")
    raw_test = np.load(ART / "test_rawashishsin_2600.npy")
    rf_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy")
    rf_test = np.load(ART / "test_sklearn_rf_meta_natural.npy")

    y = load_y()
    prior = np.bincount(y, minlength=3) / len(y)
    neg_log_prior = -np.log(prior)
    bias_recipe = BIAS  # [1.4324, 1.4689, 3.4008]

    # ========================= Hypothesis 1: drift =========================
    log("=== H1: natural-cal drift diagnostic ===")
    bias_nat, ba_nat_tuned = tune_log_bias(nat_oof, y, prior)
    bias_base, ba_base_tuned = tune_log_bias(base_oof, y, prior)
    bias_raw, ba_raw_tuned = tune_log_bias(raw_oof, y, prior)
    bias_rf, ba_rf_tuned = tune_log_bias(rf_oof, y, prior)

    drift_nat = bias_nat - neg_log_prior
    drift_base = bias_base - neg_log_prior
    drift_raw = bias_raw - neg_log_prior
    drift_rf = bias_rf - neg_log_prior

    log(f"  -log(prior)        = {neg_log_prior.round(3).tolist()}")
    log(f"  natural-cal RealMLP bias={bias_nat.round(3).tolist()} "
        f"drift={drift_nat.round(3).tolist()} max|drift|={np.abs(drift_nat).max():.3f}")
    log(f"  baseline   RealMLP bias={bias_base.round(3).tolist()} "
        f"drift={drift_base.round(3).tolist()} max|drift|={np.abs(drift_base).max():.3f}")
    log(f"  rawashishsin v3    bias={bias_raw.round(3).tolist()} "
        f"drift={drift_raw.round(3).tolist()} max|drift|={np.abs(drift_raw).max():.3f}")
    log(f"  RF natural meta    bias={bias_rf.round(3).tolist()} "
        f"drift={drift_rf.round(3).tolist()} max|drift|={np.abs(drift_rf).max():.3f}")
    h1_natcal_pass = float(np.abs(drift_nat).max()) <= 0.3
    h1_improvement = float(np.abs(drift_base).max() - np.abs(drift_nat).max())
    log(f"  H1 natural-cal verdict: {'PASS' if h1_natcal_pass else 'FAIL'} "
        f"(threshold |drift|<=0.3); drift_max_change vs baseline = "
        f"{h1_improvement:+.3f} (positive = improved)")

    # ========================= Hypothesis 2: magnitude trap =========================
    log("=== H2: magnitude-trap diagnostic ===")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    # Reconstruct LB-best 4-stack (xgb_metastack_iso @ alpha=0.30)
    meta_oof = np.load(ART / "oof_xgb_metastack.npy")
    meta_test = np.load(ART / "test_xgb_metastack.npy")
    meta_oof_iso, meta_test_iso = iso_cal(meta_oof, meta_test, y)
    lb4_oof = log_blend([lb3_oof, meta_oof_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, meta_test_iso], np.array([0.7, 0.3]))
    ba_lb4, n_lb4 = _eval(lb4_oof, y, bias_recipe)
    ba_nat_at_lb, n_nat_at_lb = _eval(nat_oof, y, bias_recipe)
    ba_base_at_lb, n_base_at_lb = _eval(base_oof, y, bias_recipe)
    ratio_nat = n_nat_at_lb / max(n_lb4, 1)
    ratio_base = n_base_at_lb / max(n_lb4, 1)
    log(f"  LB-best 4-stack         bal={ba_lb4:.5f}  errs={n_lb4}")
    log(f"  natural-cal @ LB bias    bal={ba_nat_at_lb:.5f}  errs={n_nat_at_lb}  "
        f"ratio={ratio_nat:.3f}")
    log(f"  baseline    @ LB bias    bal={ba_base_at_lb:.5f}  errs={n_base_at_lb}  "
        f"ratio={ratio_base:.3f}")
    h2_mag_pass = ratio_nat <= 1.05
    log(f"  H2 magnitude verdict: {'PASS' if h2_mag_pass else 'FAIL'} "
        f"(threshold errs <= 1.05x anchor); ratio change vs baseline = "
        f"{ratio_base - ratio_nat:+.3f} (positive = improved)")

    # ========================= Hypothesis 3: RF bank-add viability =========================
    log("=== H3: RF natural bank-add viability ===")
    h3_standalone_pass = ba_nat_tuned >= 0.974
    nat_iso_oof, _ = iso_cal(nat_oof, nat_test, y)
    rf_iso_oof, _ = iso_cal(rf_oof, rf_test, y)
    errs_nat = _errmask(nat_iso_oof, y, bias_recipe)
    errs_rf = _errmask(rf_iso_oof, y, bias_recipe)
    inter = int((errs_nat & errs_rf).sum())
    union = int((errs_nat | errs_rf).sum())
    jacc_rf = inter / max(union, 1)
    h3_jaccard_pass = jacc_rf < 0.80
    log(f"  standalone tuned bal_acc = {ba_nat_tuned:.5f} "
        f"({'PASS' if h3_standalone_pass else 'FAIL'} threshold 0.974)")
    log(f"  Jaccard(natural-cal_iso, RF_natural_iso) = {jacc_rf:.4f} "
        f"({'PASS' if h3_jaccard_pass else 'FAIL'} threshold < 0.80)")
    h3_bank_add_pass = h3_standalone_pass and h3_jaccard_pass
    log(f"  H3 bank-add verdict: {'PASS' if h3_bank_add_pass else 'FAIL'}")

    # ========================= sweeps + per-class trade =========================
    log("=== fixed-bias log-blend alpha sweep vs 3 anchors ===")
    alphas = np.arange(0.0, 0.55, 0.025)
    rows = []
    rf_nat_anchor_oof = rf_oof
    rf_nat_anchor_test = rf_test
    for name, a_oof, a_test, a_bal in [
        ("LB-best 4-stack", lb4_oof, lb4_test, ba_lb4),
        ("rawashishsin v3", raw_oof, raw_test, _eval(raw_oof, y, bias_recipe)[0]),
        ("RF natural meta", rf_nat_anchor_oof, rf_nat_anchor_test,
         _eval(rf_nat_anchor_oof, y, bias_recipe)[0]),
    ]:
        best = (-1.0, 0.0, None, None)
        for a in alphas:
            if a == 0:
                bo, bt = a_oof, a_test
            else:
                bo = log_blend([a_oof, nat_oof], np.array([1 - a, a]))
                bt = log_blend([a_test, nat_test], np.array([1 - a, a]))
            ba, _ = _eval(bo, y, bias_recipe)
            if ba > best[0]:
                best = (ba, float(a), bo, bt)
        delta = best[0] - a_bal
        # G4 net rare-class direction (anchor argmax → blend argmax shifts)
        if best[1] > 0:
            anchor_pred = (np.log(np.clip(a_oof, 1e-9, 1.0)) + bias_recipe).argmax(1)
            blend_pred = (np.log(np.clip(best[2], 1e-9, 1.0)) + bias_recipe).argmax(1)
            add_h = int(((anchor_pred != 2) & (blend_pred == 2)).sum())
            rem_h = int(((anchor_pred == 2) & (blend_pred != 2)).sum())
            net_h = add_h - rem_h
            churn_h = add_h + rem_h
            ratio = abs(net_h) / max(churn_h, 1)
            direction = "ADD-H" if net_h > 0 else "REM-H" if net_h < 0 else "balanced"
        else:
            net_h = 0; churn_h = 0; ratio = 0.0; direction = "n/a"
        log(f"  vs {name}: peak alpha={best[1]:.3f}  bal={best[0]:.5f}  "
            f"d={delta:+.5f}  net_H={net_h:+d}  churn={churn_h}  ratio={ratio:.2f}  "
            f"({direction})")
        rows.append({
            "anchor": name,
            "peak_alpha": best[1],
            "peak_bal": best[0],
            "anchor_bal": float(a_bal),
            "delta": float(delta),
            "net_h_flip": net_h,
            "h_churn": churn_h,
            "asym_ratio": float(ratio),
            "direction": direction,
        })

    # ========================= summary =========================
    log("=== SUMMARY ===")
    log(f"  H1 natural-cal drift  : {'PASS' if h1_natcal_pass else 'FAIL'} "
        f"(max|drift|={np.abs(drift_nat).max():.3f})")
    log(f"  H2 magnitude trap     : {'PASS' if h2_mag_pass else 'FAIL'} "
        f"(errs ratio={ratio_nat:.3f})")
    log(f"  H3 RF bank-add        : {'PASS' if h3_bank_add_pass else 'FAIL'} "
        f"(tuned={ba_nat_tuned:.5f}, jacc_rf={jacc_rf:.3f})")

    overall_pass = h1_natcal_pass and h2_mag_pass and h3_bank_add_pass
    log(f"  OVERALL: {'PASS — proceed to RF natural bank retrain' if overall_pass else 'FAIL'}")
    if h2_mag_pass and not h1_natcal_pass:
        log("  partial-PASS: H2 cleared without H1 → magnitude trap is "
            "structural, not calibration-driven; document this and reconsider")
    if h1_natcal_pass and not h2_mag_pass:
        log("  partial-PASS: H1 cleared without H2 → calibration improved but "
            "errs still too high; deeper architectural change needed (e.g. "
            "smoothed TE alone insufficient)")

    out = {
        "h1_natural_cal_pass": h1_natcal_pass,
        "h1_drift_max_change_vs_baseline": h1_improvement,
        "h2_magnitude_pass": h2_mag_pass,
        "h2_errs_ratio_change_vs_baseline": float(ratio_base - ratio_nat),
        "h3_bank_add_pass": h3_bank_add_pass,
        "h3_jaccard_vs_rf_natural": jacc_rf,
        "h3_standalone_tuned_bal": float(ba_nat_tuned),
        "overall_pass": overall_pass,
        "drift_natural_cal": drift_nat.tolist(),
        "drift_baseline": drift_base.tolist(),
        "drift_rawashishsin": drift_raw.tolist(),
        "drift_rf_natural": drift_rf.tolist(),
        "errs_natural_at_lb_bias": int(n_nat_at_lb),
        "errs_baseline_at_lb_bias": int(n_base_at_lb),
        "errs_lb_best_4stack": int(n_lb4),
        "tuned_bias_natural": bias_nat.tolist(),
        "tuned_bias_baseline": bias_base.tolist(),
        "sweeps": rows,
    }
    out_path = ART / "blend_realmlp_natural_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
