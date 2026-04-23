"""Blend pairwise OvO boundary specialists into the current best baseline.

Strategy: use the two OvO specialist outputs to shift the Low-vs-Medium
and Medium-vs-High log-odds of the baseline's 3-class probabilities,
gated by dgp_score:

  For rows with dgp_score in {2,3,4} -> add lam_LM * log-odds shift on L/M.
  For rows with dgp_score in {5,6,7} -> add lam_MH * log-odds shift on M/H.
  Otherwise: baseline unchanged.

Concretely, define specialist log-prob contributions (within the relevant
two-class sub-problem):
  log_s_L(x) = log(s_LM(x))          if s in LOWMED band, else 0
  log_s_M_lm(x) = log(1 - s_LM(x))    (same gate)
  log_s_M_mh(x) = log(s_MH(x))        if s in MEDHIGH band, else 0
  log_s_H(x) = log(1 - s_MH(x))       (same gate)

New log-prob:
  lp_L_new = lp_L_base + lam_LM * log_s_L
  lp_M_new = lp_M_base + lam_LM * log_s_M_lm + lam_MH * log_s_M_mh
  lp_H_new = lp_H_base + lam_MH * log_s_H

Followed by softmax across the 3 classes.

Gates: specialist contributions are ZERO outside their bands, so this
is a targeted overlay -- rows with dgp_score in {0,1,8,9} are untouched.

Sweep (lam_LM, lam_MH) independently on a grid, report OOF tuned
bal_acc with the baseline's FIXED log-bias (no retune per sweep point
-- honoring the 2026-04-21 rule that manufactured-bias retunes blow
up the OOF->LB gap).

Baselines tested:
  A) digit-XGB standalone            OOF 0.97449 (reproducible on this branch)
  B) digits-OTE x digit-XGB @ alpha=0.40 (LB-calibrated 2-way, OOF ~0.97477)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

LOWMED_SCORES = {2, 3, 4}
MEDHIGH_SCORES = {5, 6, 7}

# Grid: 0 = no specialist, 1 = full log-prob contribution.
LAM_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend_list(probs_list, weights) -> np.ndarray:
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(
                    y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def apply_ovo_overlay(
    base_probs: np.ndarray,
    s_lm: np.ndarray,          # shape (N,)
    s_mh: np.ndarray,          # shape (N,)
    score: np.ndarray,         # shape (N,)
    lam_lm: float,
    lam_mh: float,
) -> np.ndarray:
    """Compose baseline probs with OvO specialist outputs, gated by band.

    Each row's log-prob is adjusted only if its dgp_score falls in the
    relevant band. Returns softmax-normalised 3-class probs.
    """
    lp = np.log(np.clip(base_probs, 1e-9, 1.0)).copy()
    lp_sL = np.zeros_like(s_lm)
    lp_sMlm = np.zeros_like(s_lm)
    lp_sMmh = np.zeros_like(s_mh)
    lp_sH = np.zeros_like(s_mh)

    lm_gate = np.isin(score, list(LOWMED_SCORES))
    mh_gate = np.isin(score, list(MEDHIGH_SCORES))

    s_lm_clip = np.clip(s_lm, 1e-9, 1.0 - 1e-9)
    s_mh_clip = np.clip(s_mh, 1e-9, 1.0 - 1e-9)
    lp_sL[lm_gate] = np.log(s_lm_clip[lm_gate])
    lp_sMlm[lm_gate] = np.log(1.0 - s_lm_clip[lm_gate])
    lp_sMmh[mh_gate] = np.log(s_mh_clip[mh_gate])
    lp_sH[mh_gate] = np.log(1.0 - s_mh_clip[mh_gate])

    lp[:, 0] = lp[:, 0] + lam_lm * lp_sL                       # Low
    lp[:, 1] = lp[:, 1] + lam_lm * lp_sMlm + lam_mh * lp_sMmh  # Medium
    lp[:, 2] = lp[:, 2] + lam_mh * lp_sH                       # High

    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def fixed_bias_bal_acc(probs, y, bias):
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(axis=1)))


def sweep(base_probs_oof, base_probs_test, y, bias, s_lm_oof, s_mh_oof,
          s_lm_test, s_mh_test, score_oof, score_test, name):
    log(f"\n--- {name}: OvO overlay sweep (fixed bias) ---")
    base_ba = fixed_bias_bal_acc(base_probs_oof, y, bias)
    log(f"  baseline at fixed bias: {base_ba:.5f}")

    results = []
    best = (0.0, 0.0, base_ba)
    for lam_lm in LAM_GRID:
        for lam_mh in LAM_GRID:
            probs = apply_ovo_overlay(
                base_probs_oof, s_lm_oof, s_mh_oof, score_oof,
                lam_lm, lam_mh,
            )
            ba = fixed_bias_bal_acc(probs, y, bias)
            results.append({
                "lam_lm": lam_lm, "lam_mh": lam_mh, "oof_bal_acc": ba,
                "delta": ba - base_ba,
            })
            if ba > best[2]:
                best = (lam_lm, lam_mh, ba)

    log(f"  {'lam_LM':>8}  {'lam_MH':>8}  {'OOF':>9}  {'delta':>9}")
    for r in results:
        marker = "  <-- peak" if (r["lam_lm"], r["lam_mh"]) == (best[0], best[1]) else ""
        if r["delta"] >= -1e-4 or (r["lam_lm"], r["lam_mh"]) == (best[0], best[1]):
            log(f"  {r['lam_lm']:>8.3f}  {r['lam_mh']:>8.3f}  "
                f"{r['oof_bal_acc']:>9.5f}  {r['delta']:+.5f}{marker}")

    log(f"  BEST: lam_LM={best[0]:.3f}, lam_MH={best[1]:.3f}, OOF={best[2]:.5f}, "
        f"delta={best[2] - base_ba:+.5f}")
    return {
        "baseline_oof": base_ba,
        "best_lam_lm": best[0],
        "best_lam_mh": best[1],
        "best_oof": best[2],
        "delta": best[2] - base_ba,
        "sweep": results,
    }


def main() -> None:
    log("loading artefacts")
    oof_digit = np.load(ART / "oof_xgb_dist_digits.npy")
    test_digit = np.load(ART / "test_xgb_dist_digits.npy")
    oof_dote = np.load(ART / "oof_xgb_dist_digits_ote.npy")
    test_dote = np.load(ART / "test_xgb_dist_digits_ote.npy")
    oof_lm = np.load(ART / "oof_xgb_ovo_lowmed.npy")
    test_lm = np.load(ART / "test_xgb_ovo_lowmed.npy")
    oof_mh = np.load(ART / "oof_xgb_ovo_medhigh.npy")
    test_mh = np.load(ART / "test_xgb_ovo_medhigh.npy")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # dgp_score for gating
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dgp_formula import dgp_score
    score_oof = dgp_score(tr).astype(int)
    score_test = dgp_score(te).astype(int)

    # Baseline A: digit-XGB standalone (its tuned bias is LB-calibrated)
    digit_res = json.loads((ART / "xgb_dist_digits_results.json").read_text())
    bias_digit = np.array(digit_res["log_bias"])
    log(f"digit-XGB tuned bias = {bias_digit.round(4).tolist()}")

    # Baseline B: digits-OTE x digit-XGB 2-way at alpha=0.40 (LB-calibrated, old LB best)
    # Same bias (log-blend doesn't shift operating point)
    oof_2way = log_blend_list([oof_dote, oof_digit], [0.4, 0.6])
    test_2way = log_blend_list([test_dote, test_digit], [0.4, 0.6])

    # Also build a 3-way greedy-style blend using xgb_nonrule if available
    extras = {}
    for name, oof_p, test_p in [
        ("xgb_nonrule", "oof_xgb_nonrule.npy", "test_xgb_nonrule.npy"),
        ("xgb_corn",    "oof_xgb_corn.npy",    "test_xgb_corn.npy"),
    ]:
        if (ART / oof_p).exists():
            extras[name] = (np.load(ART / oof_p), np.load(ART / test_p))

    # Baseline C: digit_xgb 0.68 / xgb_nonrule 0.17 / lgbm_digit_ote 0.15
    # (the 3-way that's captured in greedy_full_bank_results.json on this branch)
    oof_3way = None
    test_3way = None
    try:
        oof_lgbm_ote = np.load(ART / "oof_lgbm_dist_digits_ote.npy")
        test_lgbm_ote = np.load(ART / "test_lgbm_dist_digits_ote.npy")
        if "xgb_nonrule" in extras:
            oof_3way = log_blend_list(
                [oof_digit, extras["xgb_nonrule"][0], oof_lgbm_ote],
                [0.68, 0.17, 0.15],
            )
            test_3way = log_blend_list(
                [test_digit, extras["xgb_nonrule"][1], test_lgbm_ote],
                [0.68, 0.17, 0.15],
            )
    except FileNotFoundError:
        pass

    # --- sweep on each baseline ---
    all_results = {}

    res_A = sweep(
        oof_digit, test_digit, y, bias_digit,
        oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
        name="A) digit-XGB standalone",
    )
    all_results["A_digit_xgb"] = res_A

    res_B = sweep(
        oof_2way, test_2way, y, bias_digit,
        oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
        name="B) digits-OTE x digit-XGB (alpha=0.40)",
    )
    all_results["B_dote_x_digit_2way"] = res_B

    if oof_3way is not None:
        res_C = sweep(
            oof_3way, test_3way, y, bias_digit,
            oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
            name="C) digit x nonrule x lgbm-digit-ote (3-way greedy)",
        )
        all_results["C_3way_greedy"] = res_C

    # --- pick the best configuration, emit submission if delta >= +5e-4 vs baseline_B ---
    # Reference: baseline B is the LB-best blend we can reproduce.
    reference = "B_dote_x_digit_2way"
    candidates = []
    for key, r in all_results.items():
        candidates.append((key, r["best_oof"], r["best_lam_lm"], r["best_lam_mh"]))
    candidates.sort(key=lambda x: -x[1])
    best_key, best_oof, best_lam_lm, best_lam_mh = candidates[0]
    log(f"\n=== best configuration ===")
    log(f"  base={best_key}  lam_LM={best_lam_lm:.3f}  lam_MH={best_lam_mh:.3f}  "
        f"OOF={best_oof:.5f}")

    ref_oof = all_results[reference]["baseline_oof"]
    delta_vs_ref = best_oof - ref_oof
    log(f"  vs baseline B (reproducible LB-best): {delta_vs_ref:+.5f}")

    # Rebuild best test probs and emit submission if above +5e-4 gate
    if best_key == "A_digit_xgb":
        base_oof, base_test = oof_digit, test_digit
    elif best_key == "B_dote_x_digit_2way":
        base_oof, base_test = oof_2way, test_2way
    else:
        base_oof, base_test = oof_3way, test_3way

    final_oof = apply_ovo_overlay(base_oof, oof_lm, oof_mh, score_oof,
                                   best_lam_lm, best_lam_mh)
    final_test = apply_ovo_overlay(base_test, test_lm, test_mh, score_test,
                                    best_lam_lm, best_lam_mh)

    cm = confusion_matrix(
        y, (np.log(np.clip(final_oof, 1e-9, 1.0)) + bias_digit).argmax(axis=1)
    )
    log(f"\nOOF confusion matrix at best OvO config:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Per-class recall
    per_class_rec = cm.diagonal() / cm.sum(axis=1)
    log(f"Per-class recall: " + "  ".join(
        [f"{c}={r:.5f}" for c, r in zip(CLASSES, per_class_rec)]
    ))

    # Error Jaccard vs baseline B
    ref_final_pred = (np.log(np.clip(oof_2way, 1e-9, 1.0)) + bias_digit).argmax(axis=1)
    ovo_final_pred = (np.log(np.clip(final_oof, 1e-9, 1.0)) + bias_digit).argmax(axis=1)
    ref_err = ref_final_pred != y
    ovo_err = ovo_final_pred != y
    jacc = (ref_err & ovo_err).sum() / max(1, (ref_err | ovo_err).sum())
    log(f"\nerror count: baseline_B={ref_err.sum()}  OvO={ovo_err.sum()}")
    log(f"Jaccard vs baseline_B = {jacc:.4f}")

    # Gating rule: ONLY emit submission if delta >= +5e-4 (the LB-probe gate).
    action = "no_submission"
    LB_PROBE_GATE = 5e-4
    if delta_vs_ref >= LB_PROBE_GATE:
        preds = (np.log(np.clip(final_test, 1e-9, 1.0)) + bias_digit).argmax(axis=1)
        sub_path = SUB / "submission_ovo_boundary_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"\nwrote {sub_path}  (delta {delta_vs_ref:+.5f} >= LB-probe gate)")
        action = "submission_ready"
    else:
        log(f"\nNo submission: delta {delta_vs_ref:+.5f} below LB-probe gate "
            f"{LB_PROBE_GATE:+.5f}")

    np.save(ART / "oof_ovo_boundary_blend.npy", final_oof)
    np.save(ART / "test_ovo_boundary_blend.npy", final_test)

    summary = {
        "baseline_A_digit_xgb_oof": res_A["baseline_oof"],
        "baseline_B_2way_oof": res_B["baseline_oof"],
        "all_results": all_results,
        "best": {
            "key": best_key,
            "lam_lm": best_lam_lm,
            "lam_mh": best_lam_mh,
            "oof": best_oof,
            "delta_vs_B": delta_vs_ref,
            "jaccard_vs_B": float(jacc),
            "error_count": int(ovo_err.sum()),
            "per_class_recall": {c: float(r) for c, r in zip(CLASSES, per_class_rec)},
        },
        "action": action,
    }
    with open(ART / "blend_ovo_boundary_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/blend_ovo_boundary_results.json")


if __name__ == "__main__":
    main()
