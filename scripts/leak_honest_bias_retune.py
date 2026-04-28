"""Coord-ascent log-bias retune on the leak-honest primary OOF.

If the leak-honest primary's optimal bias differs materially from
recipe's [1.4324, 1.4689, 3.4008], every prior gate evaluation that
pinned bias to recipe's value was using the wrong operating point for
the leak-honest surface. Re-run the 4-gate filter at the new bias.

Outputs:
  scripts/artifacts/leak_honest_bias_results.json
  submissions/submission_leak_honest_primary_retuned.csv (always emitted
                                                          as a probe candidate)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
SEED = 42
N_FOLDS = 5

CANDIDATES = [
    "sklearn_rf_meta",
    "mlp_metastack",
    "recipe_full_te_macrorec_T1_lam03",
    "recipe_full_te_dropdet",
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_perfold(oof, test, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oo = np.zeros_like(oof, dtype=np.float32)
    for tr_idx, va_idx in skf.split(oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            oo[va_idx, c] = ir.predict(oof[va_idx, c])
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


def bal_at_bias(p, y, bias):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))


def coord_ascent_bias(oof, y, init=None, step_init=0.5, step_min=0.01, max_iter=200):
    """Standard coord-ascent log-bias tune (matches recipe pipeline)."""
    bias = (init.copy() if init is not None else np.zeros(3))
    best = bal_at_bias(oof, y, bias)
    step = step_init
    for it in range(max_iter):
        improved = False
        for c in range(3):
            for delta in (+step, -step):
                trial = bias.copy()
                trial[c] += delta
                s = bal_at_bias(oof, y, trial)
                if s > best + 1e-7:
                    bias = trial
                    best = s
                    improved = True
        if not improved:
            step /= 2
            if step < step_min:
                break
    return bias, best


def per_class_recall(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(3)])


def jaccard_err(y, pred_a, pred_b):
    e_a = pred_a != y; e_b = pred_b != y
    return float((e_a & e_b).sum() / max((e_a | e_b).sum(), 1))


def four_gate_at_bias(anchor_o, cand_o, y, alpha, bias):
    blend_o = log_blend([anchor_o, cand_o], np.array([1 - alpha, alpha]))
    pred_anchor = (np.log(np.clip(anchor_o, 1e-12, 1)) + bias).argmax(1)
    pred_blend  = (np.log(np.clip(blend_o,  1e-12, 1)) + bias).argmax(1)

    bal_anchor = balanced_accuracy_score(y, pred_anchor)
    bal_blend  = balanced_accuracy_score(y, pred_blend)
    pcr_anchor = per_class_recall(y, pred_anchor)
    pcr_blend  = per_class_recall(y, pred_blend)
    pcr_delta  = pcr_blend - pcr_anchor
    jac        = jaccard_err(y, pred_blend, pred_anchor)

    add_h = int(((pred_blend == 2) & (pred_anchor != 2)).sum())
    rem_h = int(((pred_anchor == 2) & (pred_blend != 2)).sum())
    net_h = add_h - rem_h
    churn = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn, 1)

    g1 = (bal_blend - bal_anchor) >= 2e-4
    g2 = bool((pcr_delta >= -5e-4).all())
    g4 = (net_h > 0) and (g4_ratio >= 0.5)

    return {
        "alpha": float(alpha),
        "delta_oof": float(bal_blend - bal_anchor),
        "errs_anchor": int((pred_anchor != y).sum()),
        "errs_blend":  int((pred_blend  != y).sum()),
        "pcr_delta":   [float(x) for x in pcr_delta],
        "jaccard":     jac,
        "net_h":       net_h,
        "g1": bool(g1), "g2": bool(g2), "g4": bool(g4),
        "n_pass_no_g3": int(g1) + int(g2) + int(g4),
    }


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    log("loading leak-honest primary")
    prim_o = normed(np.load(ART / "leak_honest_primary_oof.npy"))
    prim_t = normed(np.load(ART / "leak_honest_primary_test.npy"))
    bal_recipe_bias = bal_at_bias(prim_o, y, RECIPE_BIAS)
    log(f"  OOF @ recipe bias [1.4324, 1.4689, 3.4008] = {bal_recipe_bias:.5f}")

    log("\ncoord-ascent bias tune on leak-honest primary OOF")
    bias_opt, bal_opt = coord_ascent_bias(prim_o, y, init=RECIPE_BIAS.copy())
    log(f"  optimal bias = [{bias_opt[0]:.4f}, {bias_opt[1]:.4f}, {bias_opt[2]:.4f}]")
    log(f"  OOF @ optimal bias = {bal_opt:.5f}")
    log(f"  Δ vs recipe bias = {bal_opt - bal_recipe_bias:+.5f}")

    # Sanity: find optimal also from zero
    log("\nsanity: coord-ascent from zero")
    bias_z, bal_z = coord_ascent_bias(prim_o, y)
    log(f"  zero-init optimal = [{bias_z[0]:.4f}, {bias_z[1]:.4f}, {bias_z[2]:.4f}]  OOF = {bal_z:.5f}")

    # Probe candidates at the new optimal bias
    log("\n=== gate sweep at leak-honest optimal bias ===")
    ALPHAS = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {
        "recipe_bias": [float(x) for x in RECIPE_BIAS],
        "bal_at_recipe_bias": float(bal_recipe_bias),
        "optimal_bias":  [float(x) for x in bias_opt],
        "bal_at_optimal_bias": float(bal_opt),
        "delta_recipe_to_optimal": float(bal_opt - bal_recipe_bias),
        "candidates": {},
    }

    for cand_name in CANDIDATES:
        log(f"\n--- {cand_name} ---")
        cand_o = normed(np.load(ART / f"oof_{cand_name}.npy").astype(np.float32))
        cand_t = normed(np.load(ART / f"test_{cand_name}.npy").astype(np.float32))
        cand_o_iso, cand_t_iso = iso_perfold(cand_o, cand_t, y)

        sweep = []
        for a in ALPHAS:
            row = four_gate_at_bias(prim_o, cand_o_iso, y, a, bias_opt)
            sweep.append(row)
            verdict = "PASS" if row["n_pass_no_g3"] == 3 else f"FAIL({row['n_pass_no_g3']}/3)"
            log(f"  α={a:.3f}  Δ={row['delta_oof']:+.5f}  errs={row['errs_blend']}  "
                f"PCR=[{row['pcr_delta'][0]:+.5f}, {row['pcr_delta'][1]:+.5f}, {row['pcr_delta'][2]:+.5f}]  "
                f"net_H={row['net_h']:+d}  G124={verdict}")

        gate_pass = [r for r in sweep if r["n_pass_no_g3"] == 3]
        best = (max(gate_pass, key=lambda r: r["delta_oof"])
                if gate_pass else max(sweep, key=lambda r: r["delta_oof"]))
        results["candidates"][cand_name] = {
            "sweep": sweep, "best": best, "any_pass": bool(gate_pass)
        }

    # Emit retuned-bias submission (the leak-honest primary at its OPTIMAL bias)
    pred_retuned = (np.log(np.clip(prim_t, 1e-12, 1)) + bias_opt).argmax(1)
    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = sample.copy()
    sub[TARGET] = [CLASSES[i] for i in pred_retuned]
    sub_path = SUB / "submission_leak_honest_primary_retuned.csv"
    sub.to_csv(sub_path, index=False)
    log(f"\nsubmission_leak_honest_primary_retuned.csv written")

    out_path = ART / "leak_honest_bias_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")
    log(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
