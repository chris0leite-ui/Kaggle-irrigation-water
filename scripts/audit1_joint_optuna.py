"""Audit-#1: Joint Optuna optimization of (α_meta, cw_L, cw_M, cw_H) on LB-best 4-stack.

Mechanism: kernels 3 + 6 from round-6 audit jointly tune blend weights AND
per-class multiplicative class weights via Optuna (200-650 trials). We've
tuned these SEQUENTIALLY (greedy α, then coord-ascent log-bias).

Math: argmax(p × cw / Z) = argmax(log p + log cw) — multiplicative cw is
mathematically equivalent to additive log-bias. But the SEARCH STRATEGY
differs: Optuna global TPE sampler may find a different local optimum
than our greedy α + coord-ascent bias.

Compare against:
  baseline: 0.7 × LB3 + 0.3 × meta_iso, log-bias [1.43, 1.47, 3.40]
            → OOF 0.98084 (LB 0.98094 verified)

Search space:
  α_meta  ∈ [0.10, 0.50]  (currently 0.30)
  cw_L, cw_M, cw_H ∈ [0.1, 50.0]  (currently exp(bias)=[4.18, 4.35, 30.0])

Scored: macro-recall on OOF + per-class guardrail check.

Diagnostic-only: emit submission ONLY if Optuna OOF Δ ≥ +0.0003 vs baseline
AND per-class recall guardrail PASSES (each class ≥ baseline - 5e-4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main() -> None:
    print("[1] Loading components...")
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)

    # Baseline: standard PRIMARY (LB 0.98094)
    p_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    p_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))
    pred_base = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
    base_macro = balanced_accuracy_score(y, pred_base)
    base_rec = recall_score(y, pred_base, average=None)
    print(f"    baseline PRIMARY OOF={base_macro:.5f}  rec={base_rec.round(5)}")

    print("[2] Optuna joint optimization...")
    # Two scoring modes:
    #   mode A: alpha + cw, NO log-bias. Tests if (alpha, cw) alone beats baseline.
    #   mode B: alpha + cw + log-bias jointly. Allows decision-rule recombination.
    # We focus on mode A first as the diagnostic — does Optuna find a better operating
    # point in (alpha, cw) space that beats coord-ascent log-bias at fixed alpha=0.30?

    def objective_modeA(trial):
        alpha = trial.suggest_float("alpha", 0.10, 0.50)
        cw_l = trial.suggest_float("cw_L", 0.1, 50.0, log=True)
        cw_m = trial.suggest_float("cw_M", 0.1, 50.0, log=True)
        cw_h = trial.suggest_float("cw_H", 0.1, 50.0, log=True)
        cw = np.array([cw_l, cw_m, cw_h], dtype=np.float32)
        blend_o = log_blend([s3_o, ms_o_iso], np.array([1 - alpha, alpha]))
        # Apply multiplicative class weights, renormalize, argmax
        adj = blend_o * cw
        pred = adj.argmax(1)
        return balanced_accuracy_score(y, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_a = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_a.enqueue_trial({"alpha": 0.30, "cw_L": float(np.exp(BIAS[0])),
                            "cw_M": float(np.exp(BIAS[1])), "cw_H": float(np.exp(BIAS[2]))})
    study_a.optimize(objective_modeA, n_trials=500, show_progress_bar=False)
    bp = study_a.best_params; bv = study_a.best_value
    print(f"  ModeA best macro = {bv:.5f}  Δ vs baseline = {bv - base_macro:+.5f}")
    print(f"  best params: alpha={bp['alpha']:.4f} cw=[{bp['cw_L']:.3f}, {bp['cw_M']:.3f}, {bp['cw_H']:.3f}]")
    # Equivalent log-bias representation
    eq_bias = np.log(np.array([bp['cw_L'], bp['cw_M'], bp['cw_H']]))
    print(f"  equivalent log-bias: [{eq_bias[0]:.4f}, {eq_bias[1]:.4f}, {eq_bias[2]:.4f}]")

    # Compute per-class recall + diff
    blend_o = log_blend([s3_o, ms_o_iso], np.array([1 - bp['alpha'], bp['alpha']]))
    cw = np.array([bp['cw_L'], bp['cw_M'], bp['cw_H']], dtype=np.float32)
    pred_opt = (blend_o * cw).argmax(1)
    rec_opt = recall_score(y, pred_opt, average=None)
    drec = (rec_opt - base_rec).round(6)
    guard = bool((drec >= -5e-4).all())
    print(f"  per-class rec: {rec_opt.round(5)}  Δ {drec}  guardrail={'PASS' if guard else 'FAIL'}")

    print("\n[3] Top 5 trials by OOF macro:")
    sorted_trials = sorted([t for t in study_a.trials if t.value is not None],
                            key=lambda t: -t.value)[:5]
    for i, t in enumerate(sorted_trials):
        p = t.params
        eq = np.log(np.array([p["cw_L"], p["cw_M"], p["cw_H"]]))
        print(f"  #{i+1} OOF={t.value:.5f}  alpha={p['alpha']:.3f}  "
              f"eq_bias=[{eq[0]:.3f}, {eq[1]:.3f}, {eq[2]:.3f}]")

    # Decision: emit submission only if Δ ≥ +0.0003 AND guardrail passes
    delta = bv - base_macro
    emit = guard and delta >= 3e-4
    print(f"\n[4] DECISION: Δ={delta:+.5f}, guardrail={guard}, emit={emit}")
    if emit:
        # Build test blend
        blend_t = log_blend([s3_t, ms_t_iso], np.array([1 - bp['alpha'], bp['alpha']]))
        pred_test = (blend_t * cw).argmax(1)
        # diff vs PRIMARY
        pt_orig = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))
        pred_orig = (np.log(np.clip(pt_orig, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_test != pred_orig).sum())
        test_df = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test_df["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_test]})
        fname = f"submission_audit1_optuna_a{int(bp['alpha']*1000):03d}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"  test_diff_vs_PRIMARY={n_diff}")
        print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")

    out = {"baseline_oof": float(base_macro),
           "optuna_best_oof": float(bv),
           "delta": float(delta),
           "best_params": bp,
           "equivalent_log_bias": eq_bias.tolist(),
           "per_class_rec": rec_opt.tolist(),
           "drec": drec.tolist(),
           "guardrail_pass": guard,
           "emit": emit,
           "n_trials": 500}
    out_path = ART / "audit1_joint_optuna_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
