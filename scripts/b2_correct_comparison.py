"""Correct B2 comparison — architecture-matched.

Bug in b2_metastack_perfoldiso_inputs.py: build_lbbest_stack returns the
4-stack (3-stack + RealMLP + nonrule_iso); the comparison then ADDS
RealMLP + nonrule_iso AGAIN, making it a 6-stack. Reported numbers
("4-stack base 0.98056", "v1_meta@α=0.30 = 0.98067") are wrong.

This script reuses the saved B2 meta OOF/test artifacts and recomputes
all comparisons with the CORRECT primary architecture:

  4-stack = 3-stack(0.25/0.35/0.40 r/s1/s7)
          + RealMLP @ α=0.20
          + xgb_nonrule_iso(full-OOF) @ α=0.075       ← LB-validated 4-stack
  PRIMARY = 4-stack + xgb_metastack_iso(full-OOF) @ α=0.30

Variants compared:
  v1 full-iso         (= LB-validated PRIMARY, OOF 0.98084)
  v1 per-fold iso     (= leak-honest v1)
  B2 full-iso
  B2 per-fold iso     (= "fully leak-honest with B2 retrained meta")

For each: report OOF @ recipe bias, OOF @ optimal-bias, test diff vs PRIMARY,
4-gate verdict against LB-validated PRIMARY.

Outputs:
  scripts/artifacts/b2_correct_comparison_results.json
  submissions/submission_b2_full_iso_a030.csv (if Δ ≥ +2e-4)
  submissions/submission_b2_perfold_iso_a030.csv (if Δ ≥ +2e-4)
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
from common import log_blend, CLS2IDX  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_full(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
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


def per_class_recall(y, pred):
    return np.array([(pred[y == k] == k).mean() for k in range(3)])


def jaccard_err(y, pred_a, pred_b):
    e_a = pred_a != y; e_b = pred_b != y
    return float((e_a & e_b).sum() / max((e_a | e_b).sum(), 1))


def build_4stack(y):
    """Correct 4-stack: 3-stack + RealMLP @0.20 + xgb_nonrule_iso(full) @0.075."""
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
    nr_iso, nrt_iso = iso_full(nr, nrt, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.80, 0.20]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.80, 0.20]))
    st2_o = log_blend([st1_o, nr_iso], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nrt_iso], np.array([0.925, 0.075]))
    return st2_o, st2_t


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("building correct 4-stack base")
    base_o, base_t = build_4stack(y)
    log(f"  4-stack base OOF @ recipe = {bal_at_bias(base_o, y, RECIPE_BIAS):.5f}")

    # Reference: current PRIMARY submission
    primary_csv = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
    primary_pred = primary_csv[TARGET].map(CLS2IDX).to_numpy()

    # Load both v1 and B2 raw meta artifacts
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    b2_o = normed(np.load(ART / "oof_xgb_metastack_perfoldiso_inputs.npy").astype(np.float32))
    b2_t = normed(np.load(ART / "test_xgb_metastack_perfoldiso_inputs.npy").astype(np.float32))

    log(f"\nstandalone meta @ recipe-bias:")
    log(f"  v1 raw       = {bal_at_bias(v1_o, y, RECIPE_BIAS):.5f}")
    log(f"  B2 raw       = {bal_at_bias(b2_o, y, RECIPE_BIAS):.5f}")

    # Apply iso variants
    log("\napplying iso variants (full-OOF + per-fold)")
    v1_full_o, v1_full_t = iso_full(v1_o, v1_t, y)
    v1_pf_o,   v1_pf_t   = iso_perfold(v1_o, v1_t, y)
    b2_full_o, b2_full_t = iso_full(b2_o, b2_t, y)
    b2_pf_o,   b2_pf_t   = iso_perfold(b2_o, b2_t, y)

    log(f"\nstandalone meta_iso @ recipe-bias:")
    log(f"  v1 full-iso  = {bal_at_bias(v1_full_o, y, RECIPE_BIAS):.5f}")
    log(f"  v1 per-f iso = {bal_at_bias(v1_pf_o,   y, RECIPE_BIAS):.5f}")
    log(f"  B2 full-iso  = {bal_at_bias(b2_full_o, y, RECIPE_BIAS):.5f}")
    log(f"  B2 per-f iso = {bal_at_bias(b2_pf_o,   y, RECIPE_BIAS):.5f}")

    # Build 4 primary variants and compare at recipe + optimal bias
    variants = {
        "v1_full":    (v1_full_o, v1_full_t),  # = LB-validated PRIMARY
        "v1_perfold": (v1_pf_o,   v1_pf_t),
        "B2_full":    (b2_full_o, b2_full_t),
        "B2_perfold": (b2_pf_o,   b2_pf_t),
    }

    results = {}
    for name, (mo, mt) in variants.items():
        prim_o = log_blend([base_o, mo], np.array([0.70, 0.30]))
        prim_t = log_blend([base_t, mt], np.array([0.70, 0.30]))

        bal_recipe = bal_at_bias(prim_o, y, RECIPE_BIAS)
        bias_opt, bal_opt = coord_ascent(prim_o, y, init=RECIPE_BIAS.copy())

        pred_recipe = (np.log(np.clip(prim_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        pred_opt    = (np.log(np.clip(prim_t, 1e-12, 1)) + bias_opt).argmax(1)
        diff_recipe = int((pred_recipe != primary_pred).sum())
        diff_opt    = int((pred_opt    != primary_pred).sum())

        log(f"\n  variant {name}:")
        log(f"    OOF @ recipe bias  = {bal_recipe:.5f}")
        log(f"    OOF @ optimal      = {bal_opt:.5f}  bias={bias_opt}")
        log(f"    test diff @ recipe = {diff_recipe}")
        log(f"    test diff @ opt    = {diff_opt}")

        # Per-class recall + Jaccard vs current PRIMARY (computed on OOF, not test)
        pred_oof = (np.log(np.clip(prim_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        if name != "v1_full":
            v1_full_prim_o = log_blend([base_o, v1_full_o], np.array([0.70, 0.30]))
            pred_v1f_oof = (np.log(np.clip(v1_full_prim_o, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
            pcr_v1f = per_class_recall(y, pred_v1f_oof)
            pcr_this = per_class_recall(y, pred_oof)
            pcr_delta = pcr_this - pcr_v1f
            jac = jaccard_err(y, pred_oof, pred_v1f_oof)
            log(f"    PCR vs v1_full (OOF, recipe bias): L={pcr_delta[0]:+.5f} M={pcr_delta[1]:+.5f} H={pcr_delta[2]:+.5f}  Jac={jac:.4f}")

        results[name] = {
            "oof_recipe_bias":   float(bal_recipe),
            "oof_optimal_bias":  float(bal_opt),
            "optimal_bias":      [float(x) for x in bias_opt],
            "test_diff_recipe":  diff_recipe,
            "test_diff_optimal": diff_opt,
        }

        # Emit submissions for B2 variants if they offer ≥ +2e-4 over v1_full at recipe bias
        if name.startswith("B2"):
            v1_full_oof = bal_at_bias(log_blend([base_o, v1_full_o], np.array([0.70, 0.30])), y, RECIPE_BIAS)
            delta = bal_recipe - v1_full_oof
            if delta >= 2e-4:
                sample = pd.read_csv(DATA / "sample_submission.csv")
                sub = sample.copy()
                sub[TARGET] = [CLASSES[i] for i in pred_recipe]
                path = SUB / f"submission_{name.lower()}_a030.csv"
                sub.to_csv(path, index=False)
                log(f"    Δ={delta:+.5f} ≥ +2e-4 → wrote {path}")

    # Iso inflation decomposition (4 variants compared at recipe bias)
    v1f = results["v1_full"]["oof_recipe_bias"]
    v1p = results["v1_perfold"]["oof_recipe_bias"]
    b2f = results["B2_full"]["oof_recipe_bias"]
    b2p = results["B2_perfold"]["oof_recipe_bias"]

    log(f"\n=== ISO INFLATION DECOMPOSITION (recipe bias) ===")
    log(f"  v1 inflation (full-iso minus per-fold-iso) = {v1f - v1p:+.5f}")
    log(f"  B2 inflation (full-iso minus per-fold-iso) = {b2f - b2p:+.5f}")
    log(f"  B2 vs v1 (full-iso)     = {b2f - v1f:+.5f}")
    log(f"  B2 vs v1 (per-fold-iso) = {b2p - v1p:+.5f}")

    out = dict(
        recipe_bias=[float(x) for x in RECIPE_BIAS],
        base_4stack_oof=float(bal_at_bias(base_o, y, RECIPE_BIAS)),
        variants=results,
        iso_inflation_v1=float(v1f - v1p),
        iso_inflation_B2=float(b2f - b2p),
        b2_vs_v1_full=float(b2f - v1f),
        b2_vs_v1_perfold=float(b2p - v1p),
        elapsed_sec=float(time.time() - t0),
    )
    json_path = ART / "b2_correct_comparison_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {json_path}")
    log(f"elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
