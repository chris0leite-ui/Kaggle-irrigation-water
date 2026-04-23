"""Fixed-bias blend sweep: digit+OTE XGB into the three LB-relevant baselines.

Adds `xgb_dist_digits_ote` (OOF in xgb_dist_digits_ote_results.json) as
a new leg in log space on top of:

  A. greedy            : OOF 0.97375  LB 0.97296
  B. greedy + nonrule  : OOF 0.97421  LB 0.97352
  C. digit-XGB         : OOF 0.97449  LB 0.97468   (current LB best)

Each sweep keeps the greedy-fitted log-bias unchanged — same design as
`blend_digits.py` to avoid the binhigh-style stage-wise OOF selection
overfit (CLAUDE.md 2026-04-21).

Decision rule:
  * α=0 peak or delta < 1e-5 → no submission, lever null
  * delta ∈ [1e-5, 5e-4]     → write submission, flag borderline, do NOT auto-submit
  * delta >= 5e-4            → write submission, worth a user-approved LB probe

For the digit-XGB-vs-OTE-XGB comparison (C), the bias used is the
DIGIT model's tuned bias (not greedy's), since digit-XGB standalone
is the strongest LB candidate.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)

VARIANT = os.environ.get("OTE_VARIANT", "default")
SUFFIX = "" if VARIANT == "default" else f"_{VARIANT}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_blend2(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    logs = la + lb
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def sweep(name: str, p_new_oof: np.ndarray, p_base_oof: np.ndarray,
          p_new_test: np.ndarray, p_base_test: np.ndarray,
          y: np.ndarray, bias: np.ndarray, baseline_oof: float,
          alphas) -> dict:
    log(f"--- {name}: fixed-bias α sweep (new at α, baseline at 1-α) ---")
    results = []
    for a in alphas:
        blend = log_blend2(p_new_oof, p_base_oof, a)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
        delta = ba - baseline_oof
        results.append({"alpha": float(a), "oof": float(ba),
                        "delta_vs_baseline": float(delta)})
        log(f"  α={a:.3f}  OOF={ba:.5f}  Δ={delta:+.5f}")
    best = max(results, key=lambda d: d["oof"])
    log(f"  best α={best['alpha']:.3f}  OOF={best['oof']:.5f}  "
        f"Δ={best['delta_vs_baseline']:+.5f}")

    blend = log_blend2(p_new_oof, p_base_oof, best["alpha"])
    lp = np.log(np.clip(blend, 1e-9, 1.0))
    cm = confusion_matrix(y, (lp + bias).argmax(axis=1))
    log(f"  CM at best α:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    test_blend = log_blend2(p_new_test, p_base_test, best["alpha"])
    return {
        "sweep": results,
        "best": best,
        "cm_at_best": cm.tolist(),
        "test_blend_at_best_alpha": test_blend,
    }


def emit_submission(test_blend, bias, te_df, sub_path: Path,
                    label: str, delta: float) -> str:
    lp = np.log(np.clip(test_blend, 1e-9, 1.0))
    preds = (lp + bias).argmax(axis=1)
    pd.DataFrame({ID: te_df[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
        sub_path, index=False
    )
    flag = "BORDERLINE" if delta < 5e-4 else "READY"
    log(f"wrote {sub_path}  ({label}, Δ={delta:+.5f}, {flag})")
    return flag


def main() -> None:
    log(f"blend variant: {VARIANT}")
    log("loading OOFs")
    oof_new = np.load(ART / f"oof_xgb_dist_digits_ote{SUFFIX}.npy")
    test_new = np.load(ART / f"test_xgb_dist_digits_ote{SUFFIX}.npy")

    # --- baseline A: greedy
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    greedy_oof = greedy_res["greedy_tuned_oof"]
    log(f"greedy baseline OOF = {greedy_oof:.5f}  "
        f"bias = {bias_greedy.round(4).tolist()}")

    # --- baseline B: greedy + nonrule (LB-best until digit landed)
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_lbb = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_lbb = log_blend2(test_nonrule, test_greedy, 0.15)

    # --- baseline C: digit-XGB standalone (current LB best)
    oof_digit = np.load(ART / "oof_xgb_dist_digits.npy")
    test_digit = np.load(ART / "test_xgb_dist_digits.npy")
    digit_res = json.loads((ART / "xgb_dist_digits_results.json").read_text())
    bias_digit = np.array(digit_res["log_bias"])
    digit_oof_tuned = digit_res["tuned_bal_acc"]

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    # OOF-at-fixed-bias for each baseline (so deltas in the sweep are honest).
    lp_lbb = np.log(np.clip(oof_lbb, 1e-9, 1.0))
    lbb_oof_at_fixed_bias = balanced_accuracy_score(
        y, (lp_lbb + bias_greedy).argmax(axis=1)
    )
    log(f"LB-best (greedy+nonrule α=0.15) OOF at greedy bias = {lbb_oof_at_fixed_bias:.5f}")
    lp_d = np.log(np.clip(oof_digit, 1e-9, 1.0))
    digit_oof_at_digit_bias = balanced_accuracy_score(
        y, (lp_d + bias_digit).argmax(axis=1)
    )
    log(f"digit-XGB OOF at its own tuned bias = {digit_oof_at_digit_bias:.5f}")

    # Standalone diagnostic on the new model.
    argmax_new = balanced_accuracy_score(y, oof_new.argmax(axis=1))
    log(f"OTE-XGB standalone argmax OOF = {argmax_new:.5f}")
    new_at_greedy_bias = balanced_accuracy_score(
        y, (np.log(np.clip(oof_new, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    )
    new_at_digit_bias = balanced_accuracy_score(
        y, (np.log(np.clip(oof_new, 1e-9, 1.0)) + bias_digit).argmax(axis=1)
    )
    log(f"OTE-XGB OOF at greedy bias = {new_at_greedy_bias:.5f}  "
        f"at digit bias = {new_at_digit_bias:.5f}")

    # Error-Jaccard diagnostic vs each baseline.
    new_argmax = oof_new.argmax(axis=1)
    e_new = new_argmax != y
    greedy_preds = (np.log(np.clip(oof_greedy, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    lbb_preds = (lp_lbb + bias_greedy).argmax(axis=1)
    digit_preds = (lp_d + bias_digit).argmax(axis=1)
    e_g = greedy_preds != y
    e_lbb = lbb_preds != y
    e_d = digit_preds != y
    j_g = (e_new & e_g).sum() / max(1, (e_new | e_g).sum())
    j_lbb = (e_new & e_lbb).sum() / max(1, (e_new | e_lbb).sum())
    j_d = (e_new & e_d).sum() / max(1, (e_new | e_d).sum())
    log(f"error counts:  OTE={e_new.sum()}  greedy={e_g.sum()}  "
        f"LB-best={e_lbb.sum()}  digit={e_d.sum()}")
    log(f"Jaccard:  vs greedy={j_g:.4f}  vs LB-best={j_lbb:.4f}  vs digit={j_d:.4f}")

    alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    summary = {
        "ote_xgb_standalone_argmax": float(argmax_new),
        "ote_xgb_at_greedy_bias": float(new_at_greedy_bias),
        "ote_xgb_at_digit_bias": float(new_at_digit_bias),
        "greedy_oof": float(greedy_oof),
        "lb_best_oof_fixed_bias": float(lbb_oof_at_fixed_bias),
        "digit_oof_at_own_bias": float(digit_oof_at_digit_bias),
        "error_count_ote": int(e_new.sum()),
        "error_count_greedy": int(e_g.sum()),
        "error_count_lb_best": int(e_lbb.sum()),
        "error_count_digit": int(e_d.sum()),
        "jaccard_vs_greedy": float(j_g),
        "jaccard_vs_lb_best": float(j_lbb),
        "jaccard_vs_digit": float(j_d),
    }

    r_a = sweep("OTE-XGB vs greedy", oof_new, oof_greedy,
                test_new, test_greedy, y, bias_greedy, greedy_oof, alphas)
    r_b = sweep("OTE-XGB vs greedy+nonrule (prior LB-best)",
                oof_new, oof_lbb, test_new, test_lbb, y, bias_greedy,
                lbb_oof_at_fixed_bias, alphas)
    r_c = sweep("OTE-XGB vs digit-XGB (CURRENT LB BEST)",
                oof_new, oof_digit, test_new, test_digit, y, bias_digit,
                digit_oof_at_digit_bias, alphas)

    summary["sweep_vs_greedy"] = {k: v for k, v in r_a.items() if k != "test_blend_at_best_alpha"}
    summary["sweep_vs_lb_best"] = {k: v for k, v in r_b.items() if k != "test_blend_at_best_alpha"}
    summary["sweep_vs_digit"] = {k: v for k, v in r_c.items() if k != "test_blend_at_best_alpha"}

    # --- emit submissions, gated by α > 0 AND delta > 1e-5
    actions: dict = {}

    if r_b["best"]["alpha"] > 0 and r_b["best"]["delta_vs_baseline"] > 1e-5:
        flag = emit_submission(
            r_b["test_blend_at_best_alpha"], bias_greedy, te,
            OUT / f"submission_greedy_nonrule_ote{SUFFIX}_blend.csv",
            f"vs LB-best α={r_b['best']['alpha']}",
            r_b["best"]["delta_vs_baseline"],
        )
        actions["vs_lb_best"] = flag
    else:
        actions["vs_lb_best"] = "no_submission"

    if r_c["best"]["alpha"] > 0 and r_c["best"]["delta_vs_baseline"] > 1e-5:
        flag = emit_submission(
            r_c["test_blend_at_best_alpha"], bias_digit, te,
            OUT / f"submission_digit_ote{SUFFIX}_blend.csv",
            f"vs digit α={r_c['best']['alpha']}",
            r_c["best"]["delta_vs_baseline"],
        )
        actions["vs_digit"] = flag
    else:
        actions["vs_digit"] = "no_submission"

    # Always emit the OTE-XGB standalone tuned submission for reference (not auto-submit).
    # The training script already writes submission_xgb_dist_digits_ote_tuned.csv.

    summary["actions"] = actions

    out_json = ART / f"blend_digits_ote{SUFFIX}_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    main()
