"""Fixed-bias blend sweep: digit-XGB into greedy and greedy+nonrule.

Adds `xgb_dist_digits` (OOF 0.XXXX, see xgb_dist_digits_results.json) as
a new leg in log space on top of our two LB-validated baselines:

  A. greedy            : OOF 0.97375  LB 0.97296
  B. greedy + nonrule  : OOF 0.97421  LB 0.97352    (current LB best)

Uses the greedy's fitted log-bias unchanged throughout — avoids the
stage-wise OOF selection overfit that killed the binhigh experiment.
Same α grid as nonrule_features_only, same 5-fold split (seed=42).

Decision rule (from CLAUDE.md):
  * α=0 peak or delta < 1e-5 → no submission, lever null
  * delta ∈ [1e-5, 5e-4]     → write submission, flag borderline, do NOT auto-submit
  * delta >= 5e-4            → write submission, worth a user-approved LB probe
"""
from __future__ import annotations

import json
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

    # Confusion matrix at best α for diagnostic visibility.
    blend = log_blend2(p_new_oof, p_base_oof, best["alpha"])
    lp = np.log(np.clip(blend, 1e-9, 1.0))
    cm = confusion_matrix(y, (lp + bias).argmax(axis=1))
    log(f"  CM at best α:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Build test blend at best α (caller decides whether to emit submission).
    test_blend = log_blend2(p_new_test, p_base_test, best["alpha"])

    return {
        "sweep": results,
        "best": best,
        "cm_at_best": cm.tolist(),
        "test_blend_at_best_alpha": test_blend,
    }


def main() -> None:
    log("loading OOFs")
    oof_new = np.load(ART / "oof_xgb_dist_digits.npy")
    test_new = np.load(ART / "test_xgb_dist_digits.npy")

    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    greedy_oof = greedy_res["greedy_tuned_oof"]
    log(f"greedy baseline OOF = {greedy_oof:.5f}  "
        f"bias = {bias_greedy.round(4).tolist()}")

    # Greedy + nonrule (LB-best) — build by log-blending at alpha=0.15 per the CLAUDE.md log.
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_best = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_best = log_blend2(test_nonrule, test_greedy, 0.15)

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    lp = np.log(np.clip(oof_best, 1e-9, 1.0))
    best_oof_at_fixed_bias = balanced_accuracy_score(
        y, (lp + bias_greedy).argmax(axis=1)
    )
    log(f"LB-best (greedy + nonrule α=0.15) OOF at fixed greedy bias = "
        f"{best_oof_at_fixed_bias:.5f}")

    # Standalone diagnostic on the new model.
    argmax_new = balanced_accuracy_score(y, oof_new.argmax(axis=1))
    log(f"digit-XGB standalone argmax OOF = {argmax_new:.5f}")

    # Error-Jaccard diagnostic — early warning for blend magnitude mismatch.
    best_preds = (np.log(np.clip(oof_best, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    greedy_preds = (np.log(np.clip(oof_greedy, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    new_argmax = oof_new.argmax(axis=1)
    e_new = new_argmax != y
    e_greedy = greedy_preds != y
    e_best = best_preds != y
    jacc_g = (e_new & e_greedy).sum() / max(1, (e_new | e_greedy).sum())
    jacc_b = (e_new & e_best).sum() / max(1, (e_new | e_best).sum())
    log(f"error count: digit-XGB={e_new.sum()} greedy={e_greedy.sum()} "
        f"best={e_best.sum()}")
    log(f"error Jaccard vs greedy={jacc_g:.4f}  vs LB-best={jacc_b:.4f}")

    alphas = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    summary = {
        "digit_xgb_standalone_argmax": float(argmax_new),
        "greedy_tuned_oof": float(greedy_oof),
        "greedy_bias": bias_greedy.tolist(),
        "lb_best_oof_fixed_bias": float(best_oof_at_fixed_bias),
        "error_count_digit_xgb": int(e_new.sum()),
        "error_count_greedy": int(e_greedy.sum()),
        "error_count_lb_best": int(e_best.sum()),
        "error_jaccard_vs_greedy": float(jacc_g),
        "error_jaccard_vs_lb_best": float(jacc_b),
    }

    r_a = sweep("digit-XGB vs greedy", oof_new, oof_greedy,
                test_new, test_greedy, y, bias_greedy, greedy_oof, alphas)
    r_b = sweep("digit-XGB vs greedy+nonrule (LB-best)", oof_new, oof_best,
                test_new, test_best, y, bias_greedy, best_oof_at_fixed_bias,
                alphas)

    # Drop test_blend from JSON but keep for submission emission.
    sweep_a = {**{k: v for k, v in r_a.items() if k != "test_blend_at_best_alpha"}}
    sweep_b = {**{k: v for k, v in r_b.items() if k != "test_blend_at_best_alpha"}}
    summary["sweep_vs_greedy"] = sweep_a
    summary["sweep_vs_lb_best"] = sweep_b

    # Emit submission only if vs-LB-best best α > 0 AND delta > 1e-5.
    best_b = r_b["best"]
    if best_b["alpha"] > 0 and best_b["delta_vs_baseline"] > 1e-5:
        test_blend = r_b["test_blend_at_best_alpha"]
        lp = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_nonrule_digits_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"wrote {sub} (α={best_b['alpha']}, OOF-lift={best_b['delta_vs_baseline']:+.5f})")
        if best_b["delta_vs_baseline"] < 5e-4:
            summary["action"] = "borderline_no_submit"
            log("BORDERLINE: OOF lift below 0.0005 LB-probe threshold — do not submit without user approval")
        else:
            summary["action"] = "ready_to_submit"
        summary["submission_path"] = str(sub)
    else:
        summary["action"] = "no_submission"
        log("no α > 0 gave a positive delta — digit-XGB adds no signal to LB-best.")

    with open(ART / "blend_digits_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {ART}/blend_digits_results.json")


if __name__ == "__main__":
    main()
