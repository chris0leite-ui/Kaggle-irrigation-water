"""B1: per-component iso isolation diagnostic.

Question: which component contributes the +0.00010 iso inflation in
the LB-best primary — xgb_nonrule, xgb_metastack, or both?

Builds 4 architecture variants:
  A) nonrule=full-OOF-iso, metastack=full-OOF-iso  (= LB-best primary)
  B) nonrule=PER-FOLD-iso, metastack=full-OOF-iso  (B1's "fix nonrule")
  C) nonrule=full-OOF-iso, metastack=PER-FOLD-iso  (B1's "fix metastack")
  D) nonrule=PER-FOLD-iso, metastack=PER-FOLD-iso  (full leak-honest)

For each: report OOF at recipe bias + at retuned bias + test row diff
vs current PRIMARY.

Then compute marginal inflation contribution:
  inflation(nonrule)   = OOF(A) - OOF(B)
  inflation(metastack) = OOF(A) - OOF(C)
  inflation(combined)  = OOF(A) - OOF(D)

If both >> 0 individually but combined ~= sum: each contributes
independently. If combined < sum: interaction (e.g., metastack
already absorbs nonrule's inflation, so per-fold-iso on nonrule
alone yields most of the gain).

Output:
  scripts/artifacts/b1_per_component_iso_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score
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


def log(m): print(f"[{time.strftime('%H:%M:%S')}] B1: {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_full(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return normed(oo), normed(tt)


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


def coord_ascent(oof, y, init=None, step_init=0.5, step_min=0.01, max_iter=200):
    bias = (init.copy() if init is not None else np.zeros(3))
    best = bal_at_bias(oof, y, bias)
    step = step_init
    for _ in range(max_iter):
        improved = False
        for c in range(3):
            for delta in (+step, -step):
                trial = bias.copy(); trial[c] += delta
                s = bal_at_bias(oof, y, trial)
                if s > best + 1e-7:
                    bias = trial; best = s; improved = True
        if not improved:
            step /= 2
            if step < step_min:
                break
    return bias, best


def build_variant(nonrule_iso, nonrule_iso_t, metastack_iso, metastack_iso_t,
                  lb3_o, lb3_t, rm, rmt):
    """Build primary architecture from variant-specific iso components."""
    st1_o = log_blend([lb3_o, rm], np.array([0.80, 0.20]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.80, 0.20]))
    st2_o = log_blend([st1_o, nonrule_iso], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nonrule_iso_t], np.array([0.925, 0.075]))
    prim_o = log_blend([st2_o, metastack_iso], np.array([0.70, 0.30]))
    prim_t = log_blend([st2_t, metastack_iso_t], np.array([0.70, 0.30]))
    return prim_o, prim_t


def predict(p, bias):
    return (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)


def main():
    t0 = time.time()
    log("=== B1: per-component iso isolation ===")
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # Load shared components
    log("loading bank components")
    r  = normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t= normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t= normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = normed(np.load(ART / "oof_realmlp.npy"))
    rmt= normed(np.load(ART / "test_realmlp.npy"))
    nr = normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt= normed(np.load(ART / "test_xgb_nonrule.npy"))
    ms = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mst= normed(np.load(ART / "test_xgb_metastack.npy"))

    # 3-stack base (shared across all variants)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)

    # Iso variants of each component
    log("computing iso variants for nonrule + metastack")
    nr_full,  nrt_full  = iso_full(nr, nrt, y)
    nr_perf,  nrt_perf  = iso_perfold(nr, nrt, y)
    ms_full,  mst_full  = iso_full(ms, mst, y)
    ms_perf,  mst_perf  = iso_perfold(ms, mst, y)

    # Reference (current PRIMARY) test predictions for diff counts
    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred = primary_csv[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    variants = [
        ("A_full_full",   nr_full, nrt_full, ms_full, mst_full),
        ("B_perf_full",   nr_perf, nrt_perf, ms_full, mst_full),
        ("C_full_perf",   nr_full, nrt_full, ms_perf, mst_perf),
        ("D_perf_perf",   nr_perf, nrt_perf, ms_perf, mst_perf),
    ]

    results = {}
    for name, nro, nrto, mso, msto in variants:
        prim_o, prim_t = build_variant(nro, nrto, mso, msto, lb3_o, lb3_t, rm, rmt)

        bal_recipe = bal_at_bias(prim_o, y, RECIPE_BIAS)
        bias_opt, bal_opt = coord_ascent(prim_o, y, init=RECIPE_BIAS.copy())

        pred_recipe = predict(prim_t, RECIPE_BIAS)
        pred_opt    = predict(prim_t, bias_opt)
        diff_recipe = int((pred_recipe != primary_pred).sum())
        diff_opt    = int((pred_opt    != primary_pred).sum())

        log(f"\n  variant {name}:")
        log(f"    OOF @ recipe bias = {bal_recipe:.5f}")
        log(f"    OOF @ optimal     = {bal_opt:.5f}  bias={bias_opt}")
        log(f"    test diff vs PRIMARY @ recipe = {diff_recipe}")
        log(f"    test diff vs PRIMARY @ optimal = {diff_opt}")

        results[name] = {
            "bal_recipe_bias":  float(bal_recipe),
            "bal_optimal_bias": float(bal_opt),
            "optimal_bias":     [float(x) for x in bias_opt],
            "diff_vs_primary_recipe":  diff_recipe,
            "diff_vs_primary_optimal": diff_opt,
        }

    # Inflation decomposition
    A = results["A_full_full"]["bal_recipe_bias"]
    B = results["B_perf_full"]["bal_recipe_bias"]
    C = results["C_full_perf"]["bal_recipe_bias"]
    D = results["D_perf_perf"]["bal_recipe_bias"]

    inf_nonrule    = A - B
    inf_metastack  = A - C
    inf_combined   = A - D
    inf_sum        = inf_nonrule + inf_metastack
    interaction    = inf_combined - inf_sum

    log(f"\n=== inflation decomposition (OOF @ recipe bias) ===")
    log(f"  A (full,full)     OOF = {A:.5f}")
    log(f"  B (perf,full)     OOF = {B:.5f}  Δ = {B - A:+.5f}  (fix nonrule alone)")
    log(f"  C (full,perf)     OOF = {C:.5f}  Δ = {C - A:+.5f}  (fix metastack alone)")
    log(f"  D (perf,perf)     OOF = {D:.5f}  Δ = {D - A:+.5f}  (fix both)")
    log(f"")
    log(f"  inflation(nonrule)   = {inf_nonrule:+.5f}")
    log(f"  inflation(metastack) = {inf_metastack:+.5f}")
    log(f"  sum (independence)   = {inf_sum:+.5f}")
    log(f"  inflation(combined)  = {inf_combined:+.5f}")
    log(f"  interaction term     = {interaction:+.5f}")
    if abs(interaction) < 1e-5:
        log(f"  → independent: each component contributes its own iso inflation")
    elif interaction > 1e-5:
        log(f"  → super-additive: components compound (joint > sum of parts)")
    else:
        log(f"  → sub-additive: shared inflation absorbed by metastack")

    results["inflation_decomposition"] = {
        "inf_nonrule": float(inf_nonrule),
        "inf_metastack": float(inf_metastack),
        "inf_combined": float(inf_combined),
        "inf_sum_independent": float(inf_sum),
        "interaction": float(interaction),
    }
    out_path = ART / "b1_per_component_iso_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
