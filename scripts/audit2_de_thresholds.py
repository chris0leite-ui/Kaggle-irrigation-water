"""Audit-#2: Differential Evolution prob-space thresholds (kernel 1, wguesdon).

Mechanism: per-class threshold subtraction in PROB space:
  pred = argmax(p - threshold)

Mathematically distinct from log-bias (additive in log-space) and from LP
quotas (cardinality-constrained). DE optimizer (scipy) for global search.

Two variants:
  A. PURE thresholds (no log-bias): argmax(p - t), t in [0, 0.5]
  B. CASCADED: log-bias first, then thresholds: argmax(softmax(log p + b) - t)

Compare against:
  baseline: log-bias [1.43, 1.47, 3.40] → OOF 0.98084 (LB 0.98094)

Diagnostic-only: emit submission ONLY if Δ ≥ +0.0003 vs baseline AND
per-class recall guardrail PASSES.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
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
    print("[1] Loading PRIMARY components...")
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)

    p_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    p_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))

    pred_base = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
    base_macro = balanced_accuracy_score(y, pred_base)
    base_rec = recall_score(y, pred_base, average=None)
    print(f"    baseline OOF={base_macro:.5f}  rec={base_rec.round(5)}")

    print("\n[2] Variant A: PURE prob-space thresholds (no log-bias)")
    print("    pred = argmax(p - t),  t in [0, 0.5] per class")

    def neg_macro_modeA(t):
        adj = p_o - t.reshape(1, 3).astype(np.float32)
        return -balanced_accuracy_score(y, adj.argmax(1))

    boundsA = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
    resA = differential_evolution(neg_macro_modeA, boundsA, seed=42, maxiter=80,
                                    tol=1e-6, popsize=20, polish=True)
    macroA = -resA.fun; tA = resA.x
    predA = (p_o - tA.reshape(1, 3).astype(np.float32)).argmax(1)
    recA = recall_score(y, predA, average=None)
    drecA = (recA - base_rec).round(6)
    guardA = bool((drecA >= -5e-4).all())
    print(f"    DE-A best macro={macroA:.5f}  Δ={macroA - base_macro:+.5f}")
    print(f"    thresholds={tA.round(5)}  rec={recA.round(5)}  Δrec={drecA}")
    print(f"    guardrail={'PASS' if guardA else 'FAIL'}")

    print("\n[3] Variant B: CASCADED (log-bias → softmax → DE thresholds)")
    print("    pred = argmax(softmax(log p + B) - t),  t in [-0.3, 0.3] per class")
    # Apply our log-bias first
    z = np.log(np.clip(p_o, 1e-12, 1)) + BIAS
    e = np.exp(z - z.max(1, keepdims=True))
    p_post = e / e.sum(1, keepdims=True)

    def neg_macro_modeB(t):
        adj = p_post - t.reshape(1, 3).astype(np.float32)
        return -balanced_accuracy_score(y, adj.argmax(1))

    boundsB = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]
    resB = differential_evolution(neg_macro_modeB, boundsB, seed=42, maxiter=80,
                                    tol=1e-6, popsize=20, polish=True)
    macroB = -resB.fun; tB = resB.x
    predB = (p_post - tB.reshape(1, 3).astype(np.float32)).argmax(1)
    recB = recall_score(y, predB, average=None)
    drecB = (recB - base_rec).round(6)
    guardB = bool((drecB >= -5e-4).all())
    print(f"    DE-B best macro={macroB:.5f}  Δ={macroB - base_macro:+.5f}")
    print(f"    thresholds={tB.round(5)}  rec={recB.round(5)}  Δrec={drecB}")
    print(f"    guardrail={'PASS' if guardB else 'FAIL'}")

    # Decision: emit submission only if Δ ≥ +0.0003 AND guardrail passes
    print("\n[4] DECISION")
    best_mode = None; best_delta = -1
    for mode, macro, t, drec, guard in [("A", macroA, tA, drecA, guardA),
                                          ("B", macroB, tB, drecB, guardB)]:
        d = macro - base_macro
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"  Mode {mode}: Δ={d:+.5f}, guard={guard}, emit={emit}{marker}")
        if emit and d > best_delta:
            best_mode = mode; best_delta = d

    if best_mode == "A":
        pred_test = (p_t - tA.reshape(1, 3).astype(np.float32)).argmax(1)
    elif best_mode == "B":
        z_t = np.log(np.clip(p_t, 1e-12, 1)) + BIAS
        e_t = np.exp(z_t - z_t.max(1, keepdims=True))
        p_post_t = e_t / e_t.sum(1, keepdims=True)
        pred_test = (p_post_t - tB.reshape(1, 3).astype(np.float32)).argmax(1)
    else:
        pred_test = None
        print("  No variant passes both gates; no submission emitted.")

    if pred_test is not None:
        # Diff vs PRIMARY
        pred_orig = (np.log(np.clip(p_t, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_test != pred_orig).sum())
        test_df = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test_df["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_test]})
        fname = f"submission_audit2_de_mode{best_mode}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"  test_diff_vs_PRIMARY={n_diff}")
        print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")

    out = {"baseline_oof": float(base_macro),
           "modeA": {"oof": float(macroA), "delta": float(macroA - base_macro),
                      "thresholds": tA.tolist(), "rec": recA.tolist(),
                      "drec": drecA.tolist(), "guard": guardA},
           "modeB": {"oof": float(macroB), "delta": float(macroB - base_macro),
                      "thresholds": tB.tolist(), "rec": recB.tolist(),
                      "drec": drecB.tolist(), "guard": guardB},
           "best_emit": best_mode}
    out_path = ART / "audit2_de_thresholds_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
