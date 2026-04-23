"""Non-rule OvO boundary blend sweep.

Tests the non-rule-feature-only OvO specialists (AUC 0.68 Low-vs-Med,
AUC 0.79 Med-vs-High on in-domain) as log-prob shifts on top of the
baseline digit-XGB.

Architecturally orthogonal by feature-view: the main digit-XGB uses
the full 89-feature dist+digits set, the specialists use only 13
non-rule features. Same band-gated composition as
`blend_ovo_boundary.py`:

  log_pL_new = log_pL + lam_LM * log(s_LM)      if score in {2,3,4}
  log_pM_new = log_pM + lam_LM * log(1 - s_LM) + lam_MH * log(s_MH)
  log_pH_new = log_pH + lam_MH * log(1 - s_MH)  if score in {5,6,7}
  followed by row softmax.

Fixed baseline log-bias, sweep (lam_LM, lam_MH) on a grid.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from dgp_formula import dgp_score as compute_dgp_score

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

LAM_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend_list(probs_list, weights) -> np.ndarray:
    logs = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        logs = logs + w * np.log(np.clip(p, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def apply_ovo_overlay(base_probs, s_lm, s_mh, score, lam_lm, lam_mh):
    lp = np.log(np.clip(base_probs, 1e-9, 1.0)).copy()
    lm_gate = np.isin(score, list(LOWMED_SCORES))
    mh_gate = np.isin(score, list(MEDHIGH_SCORES))
    s_lm_c = np.clip(s_lm, 1e-9, 1.0 - 1e-9)
    s_mh_c = np.clip(s_mh, 1e-9, 1.0 - 1e-9)

    lp_sL = np.where(lm_gate, np.log(s_lm_c), 0.0)
    lp_sMlm = np.where(lm_gate, np.log(1.0 - s_lm_c), 0.0)
    lp_sMmh = np.where(mh_gate, np.log(s_mh_c), 0.0)
    lp_sH = np.where(mh_gate, np.log(1.0 - s_mh_c), 0.0)

    lp[:, 0] = lp[:, 0] + lam_lm * lp_sL
    lp[:, 1] = lp[:, 1] + lam_lm * lp_sMlm + lam_mh * lp_sMmh
    lp[:, 2] = lp[:, 2] + lam_mh * lp_sH
    lp -= lp.max(1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(1, keepdims=True)


def fixed_bias_bal_acc(probs, y, bias):
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    return float(balanced_accuracy_score(y, (lp + bias).argmax(axis=1)))


def sweep(base_oof, base_test, y, bias, s_lm_oof, s_mh_oof, s_lm_test, s_mh_test,
          score_oof, score_test, name):
    log(f"\n--- {name} ---")
    base_ba = fixed_bias_bal_acc(base_oof, y, bias)
    log(f"  baseline OOF @ fixed bias: {base_ba:.5f}")

    results = []
    best = (0.0, 0.0, base_ba)
    for lam_lm in LAM_GRID:
        for lam_mh in LAM_GRID:
            p = apply_ovo_overlay(base_oof, s_lm_oof, s_mh_oof, score_oof,
                                   lam_lm, lam_mh)
            ba = fixed_bias_bal_acc(p, y, bias)
            results.append({"lam_lm": lam_lm, "lam_mh": lam_mh,
                            "oof": ba, "delta": ba - base_ba})
            if ba > best[2]:
                best = (lam_lm, lam_mh, ba)

    # Print positive or near-zero deltas
    log(f"  {'lam_LM':>8}  {'lam_MH':>8}  {'OOF':>9}  {'delta':>9}")
    for r in results:
        if r["delta"] >= -5e-5 or (r["lam_lm"], r["lam_mh"]) == (best[0], best[1]):
            marker = "  <-- peak" if (r["lam_lm"], r["lam_mh"]) == (best[0], best[1]) else ""
            log(f"  {r['lam_lm']:>8.3f}  {r['lam_mh']:>8.3f}  "
                f"{r['oof']:>9.5f}  {r['delta']:+.5f}{marker}")

    log(f"  BEST: lam_LM={best[0]}, lam_MH={best[1]}, OOF={best[2]:.5f}, "
        f"delta={best[2] - base_ba:+.5f}")
    return {"baseline_oof": base_ba, "best_lam_lm": best[0],
            "best_lam_mh": best[1], "best_oof": best[2],
            "delta": best[2] - base_ba, "sweep": results}


def main() -> None:
    log("loading artefacts")
    oof_digit = np.load(ART / "oof_xgb_dist_digits.npy")
    test_digit = np.load(ART / "test_xgb_dist_digits.npy")
    oof_dote = np.load(ART / "oof_xgb_dist_digits_ote.npy")
    test_dote = np.load(ART / "test_xgb_dist_digits_ote.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")

    oof_lm = np.load(ART / "oof_xgb_ovo_lowmed_nonrule.npy")
    oof_mh = np.load(ART / "oof_xgb_ovo_medhigh_nonrule.npy")
    test_lm = np.load(ART / "test_xgb_ovo_lowmed_nonrule.npy")
    test_mh = np.load(ART / "test_xgb_ovo_medhigh_nonrule.npy")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    digit_res = json.loads((ART / "xgb_dist_digits_results.json").read_text())
    bias = np.array(digit_res["log_bias"])
    log(f"digit-XGB tuned bias = {bias.round(4).tolist()}")

    score_oof = compute_dgp_score(tr).astype(int)
    score_test = compute_dgp_score(te).astype(int)

    # Reproducible LB-best 2-way (digits_ote 0.4 / digit_xgb 0.6)
    oof_2way = log_blend_list([oof_dote, oof_digit], [0.4, 0.6])
    test_2way = log_blend_list([test_dote, test_digit], [0.4, 0.6])

    # Also try 3-way: digits_ote x digit_xgb x nonrule, per CLAUDE.md's greedy_full_bank
    # components available on this branch (digit_xgb + xgb_nonrule)
    # We try digit_xgb + xgb_nonrule at 0.85/0.15 (the nonrule greedy add logged at
    # LB+0.00056 was at 0.20 alpha on top of greedy; use it as a strong baseline proxy).
    oof_dn = log_blend_list([oof_digit, oof_nonrule], [0.85, 0.15])
    test_dn = log_blend_list([test_digit, test_nonrule], [0.85, 0.15])

    all_res = {}
    all_res["A_digit_xgb"] = sweep(
        oof_digit, test_digit, y, bias,
        oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
        "A) digit-XGB + nonrule-OvO overlay",
    )
    all_res["B_2way"] = sweep(
        oof_2way, test_2way, y, bias,
        oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
        "B) 2-way (dote 0.4 / digit 0.6) + nonrule-OvO overlay",
    )
    all_res["C_digit_nonrule"] = sweep(
        oof_dn, test_dn, y, bias,
        oof_lm, oof_mh, test_lm, test_mh, score_oof, score_test,
        "C) digit-XGB (0.85) x xgb_nonrule (0.15) + nonrule-OvO overlay",
    )

    # Pick best
    best_key, best_r = max(all_res.items(), key=lambda kv: kv[1]["best_oof"])
    log(f"\n=== best overall ===")
    log(f"  {best_key}: lam_LM={best_r['best_lam_lm']:.3f}, "
        f"lam_MH={best_r['best_lam_mh']:.3f}, "
        f"OOF={best_r['best_oof']:.5f}, delta={best_r['delta']:+.5f}")

    if best_key == "A_digit_xgb":
        base_oof, base_test = oof_digit, test_digit
    elif best_key == "B_2way":
        base_oof, base_test = oof_2way, test_2way
    else:
        base_oof, base_test = oof_dn, test_dn

    final_oof = apply_ovo_overlay(base_oof, oof_lm, oof_mh, score_oof,
                                   best_r["best_lam_lm"], best_r["best_lam_mh"])
    final_test = apply_ovo_overlay(base_test, test_lm, test_mh, score_test,
                                    best_r["best_lam_lm"], best_r["best_lam_mh"])

    cm = confusion_matrix(
        y, (np.log(np.clip(final_oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    per_rec = cm.diagonal() / cm.sum(axis=1)
    log(f"\nConfusion matrix at best:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
    log("Per-class recall: " + "  ".join(
        [f"{c}={r:.5f}" for c, r in zip(CLASSES, per_rec)]
    ))

    # Emit submission only if delta vs baseline_B is >= +5e-4 (honest LB-probe gate)
    ref_oof = all_res["B_2way"]["baseline_oof"]
    delta_vs_ref = best_r["best_oof"] - ref_oof
    action = "no_submission"
    LB_PROBE_GATE = 5e-4
    if delta_vs_ref >= LB_PROBE_GATE:
        preds = (np.log(np.clip(final_test, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub_path = SUB / "submission_ovo_nonrule_blend.csv"
        pd.DataFrame({ID: te[ID],
                      TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub_path, index=False)
        log(f"\nwrote {sub_path}  (delta {delta_vs_ref:+.5f} >= gate)")
        action = "submission_ready"
    else:
        log(f"\nNo submission: delta {delta_vs_ref:+.5f} < gate {LB_PROBE_GATE:+.5f}")

    np.save(ART / "oof_ovo_nonrule_blend.npy", final_oof)
    np.save(ART / "test_ovo_nonrule_blend.npy", final_test)

    with open(ART / "blend_ovo_boundary_nonrule_results.json", "w") as f:
        json.dump({
            "digit_xgb_bias": bias.tolist(),
            "all_results": all_res,
            "best_key": best_key,
            "best": {
                "lam_lm": best_r["best_lam_lm"],
                "lam_mh": best_r["best_lam_mh"],
                "oof": best_r["best_oof"],
                "delta_vs_B_2way": delta_vs_ref,
                "per_class_recall": {c: float(r) for c, r in zip(CLASSES, per_rec)},
            },
            "action": action,
        }, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
