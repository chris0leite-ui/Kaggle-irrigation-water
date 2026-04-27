"""R5 — per-fold iso re-fit of meta_3wnn (vs the existing full-OOF iso).

Existing tier1b_helpers.iso_cal fits 3 isotonic regressors on the
ENTIRE OOF (all 630k rows) as a one-shot operation, then applies
them to OOF and test. This has a subtle leak path: the iso fit on
fold i's val rows uses fold i's val rows as targets — leaking
fold-specific calibration noise back into the iso curves.

R5 instead does proper leak-safe per-fold iso:
  for each fold i in StratifiedKFold(seed=42):
    fit iso on rows ∉ fold_i (4 folds of OOF data)
    apply to fold_i's OOF rows
  test side: fit iso on full OOF (no leakage since test rows aren't
              in any fold)

Then sweeps α onto the LB-best 3-stack and reports peak with
guardrail. Only emits a submission if Δ ≥ +1e-4 OOF AND per-class
recall preserved.

Compares against the full-OOF iso version to quantify the leak.

Design rationale (from CLAUDE.md 2026-04-25 audit follow-up):
'tier1b_greedy_perfoldiso showed iso-on-full-OOF was contributing
~1bp inflation. The current primary's +0.00086 LB lift over LB-best
3-stack is mostly genuine signal, not iso-leak inflation.'
That study used meta_v1 as the iso source; R5 repeats it for
meta_3wnn (the bigger-bank meta with 837 fewer raw errs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, log, normed,
)

SEED = 42
N_FOLDS = 5
GUARDRAIL = 5e-4
EMIT_DELTA = 1e-4
ALPHA_GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                       0.40, 0.45, 0.50])


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def predict_with_bias(p):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def per_fold_iso_cal(oof: np.ndarray, test: np.ndarray, y: np.ndarray):
    """Per-fold leak-safe isotonic calibration.

    For each (oof_fold tr_idx, va_idx): fit iso on `oof[tr_idx]` against
    `y[tr_idx]`, predict on `oof[va_idx]`. Test side: fit iso on full
    OOF (test rows never in any fold).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oo = np.zeros_like(oof, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof, y)):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip",
                                    y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            oo[va_idx, c] = ir.predict(oof[va_idx, c])

    # Test: full-OOF iso (no leakage, test isn't in folds)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])

    # Renormalize rows
    oo = oo / np.clip(oo.sum(axis=1, keepdims=True), 1e-9, None)
    tt = tt / np.clip(tt.sum(axis=1, keepdims=True), 1e-9, None)
    return oo, tt


def sweep_blend(anchor_o, cand_o, y, anchor_bal, anchor_pcr, label):
    rows = []
    for a in ALPHA_GRID:
        if a == 0:
            blend_o = anchor_o
        else:
            blend_o = log_blend([anchor_o, cand_o], np.array([1 - a, a]))
        pred = predict_with_bias(blend_o)
        bal = balanced_accuracy_score(y, pred)
        pcr = per_class_recall(y, pred)
        guard = all(pcr[c] >= anchor_pcr[c] - GUARDRAIL for c in range(3))
        rows.append(dict(
            alpha=float(a), bal=float(bal),
            delta=float(bal - anchor_bal),
            pcr=pcr.tolist(),
            pcr_pass=bool(guard),
        ))
    log(f"=== {label} sweep ===")
    log(f"{'α':>5} {'bal':>9} {'Δ':>9} {'PCR_L':>7} {'PCR_M':>7} {'PCR_H':>7} {'guard':>6}")
    for r in rows:
        log(f"{r['alpha']:>5.3f} {r['bal']:>9.5f} {r['delta']:>+9.5f} "
            f"{r['pcr'][0]:>7.4f} {r['pcr'][1]:>7.4f} {r['pcr'][2]:>7.4f} "
            f"{'PASS' if r['pcr_pass'] else 'FAIL':>6}")
    return rows


def emit(label, anchor_t, cand_t, alpha, ids):
    blend_t = log_blend([anchor_t, cand_t], np.array([1 - alpha, alpha])) \
        if alpha > 0 else anchor_t
    pred = predict_with_bias(blend_t)
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids, "Irrigation_Need": [cls_map[i] for i in pred]})
    p = SUB / f"submission_r5_{label}_a{int(alpha*1000):03d}.csv"
    sub.to_csv(p, index=False)
    log(f"  EMIT {p}  pred dist: {dict(sub['Irrigation_Need'].value_counts())}")
    return str(p)


def main():
    t0 = time.time()
    y = load_y()
    log("loading LB-best 3-stack anchor (no meta yet)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    pred_lb3 = predict_with_bias(lb3_o)
    lb3_bal = balanced_accuracy_score(y, pred_lb3)
    lb3_pcr = per_class_recall(y, pred_lb3)
    log(f"  LB-best 3-stack OOF = {lb3_bal:.5f}  "
        f"PCR L={lb3_pcr[0]:.4f} M={lb3_pcr[1]:.4f} H={lb3_pcr[2]:.4f}")

    # PRIMARY (3-stack + meta_v1_iso α=0.30) for context
    m_v1_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    m_v1_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    mv1_iso_o, mv1_iso_t = iso_cal(m_v1_o, m_v1_t, y)
    primary_o = log_blend([lb3_o, mv1_iso_o], np.array([0.70, 0.30]))
    primary_t = log_blend([lb3_t, mv1_iso_t], np.array([0.70, 0.30]))
    primary_bal = balanced_accuracy_score(y, predict_with_bias(primary_o))
    log(f"  PRIMARY OOF (LB 0.98094) = {primary_bal:.5f}")

    log("loading meta_3wnn raw OOF/test")
    m_o = normed(np.load(ART / "oof_xgb_metastack_3wnn.npy").astype(np.float32))
    m_t = normed(np.load(ART / "test_xgb_metastack_3wnn.npy").astype(np.float32))

    log("computing FULL-OOF iso (existing approach, baseline)")
    full_iso_o, full_iso_t = iso_cal(m_o, m_t, y)

    log("computing PER-FOLD iso (R5 leak-safe)")
    pf_iso_o, pf_iso_t = per_fold_iso_cal(m_o, m_t, y)

    # Sanity check: standalone @ recipe bias
    full_iso_bal = balanced_accuracy_score(y, predict_with_bias(full_iso_o))
    pf_iso_bal = balanced_accuracy_score(y, predict_with_bias(pf_iso_o))
    raw_bal = balanced_accuracy_score(y, predict_with_bias(m_o))
    log(f"  meta_3wnn standalone @ recipe bias:")
    log(f"    raw OOF       = {raw_bal:.5f}")
    log(f"    full-OOF iso  = {full_iso_bal:.5f}")
    log(f"    per-fold iso  = {pf_iso_bal:.5f}  (Δ vs full = {pf_iso_bal-full_iso_bal:+.5f})")

    # Sweep α onto LB-best 3-stack: full-iso vs per-fold iso
    full_rows = sweep_blend(lb3_o, full_iso_o, y, lb3_bal, lb3_pcr,
                             "full-OOF iso vs LB-best 3-stack")
    pf_rows = sweep_blend(lb3_o, pf_iso_o, y, lb3_bal, lb3_pcr,
                           "per-fold iso vs LB-best 3-stack")

    # Compare deltas. The "per-fold honest" peak Δ is the true signal magnitude.
    # If per-fold peak < full-OOF peak by > 0.0001, that's iso-leak inflation.
    full_pass = [r for r in full_rows if r["pcr_pass"]]
    pf_pass = [r for r in pf_rows if r["pcr_pass"]]
    full_best = max(full_pass, key=lambda r: r["delta"]) if full_pass else None
    pf_best = max(pf_pass, key=lambda r: r["delta"]) if pf_pass else None
    log(f"\n=== leak quantification ===")
    log(f"full-OOF iso best gate-pass: α={full_best['alpha']:.3f} Δ={full_best['delta']:+.5f}"
        if full_best else "full: no gate-pass")
    log(f"per-fold iso best gate-pass: α={pf_best['alpha']:.3f} Δ={pf_best['delta']:+.5f}"
        if pf_best else "per-fold: no gate-pass")
    if full_best and pf_best:
        leak = full_best["delta"] - pf_best["delta"]
        log(f"iso-leak inflation = {leak:+.5f}  "
            f"({'small (<1bp)' if abs(leak)<1e-4 else 'material' if abs(leak)<5e-4 else 'large'})")

    # Compare PRIMARY (with full-iso meta_v1) → does per-fold iso meta_3wnn beat it?
    log(f"\n=== PRIMARY-arch comparison: 0.7*lb3 + 0.3*meta_3wnn (per-fold iso) ===")
    repl_o = log_blend([lb3_o, pf_iso_o], np.array([0.7, 0.3]))
    repl_t = log_blend([lb3_t, pf_iso_t], np.array([0.7, 0.3]))
    repl_bal = balanced_accuracy_score(y, predict_with_bias(repl_o))
    repl_pcr = per_class_recall(y, predict_with_bias(repl_o))
    repl_pass = all(repl_pcr[c] >= primary_bal_pcr - GUARDRAIL for c, primary_bal_pcr in
                    enumerate([0.9955, 0.9695, 0.9775]))
    log(f"  OOF = {repl_bal:.5f}  Δ vs PRIMARY = {repl_bal-primary_bal:+.5f}")
    log(f"  PCR = L{repl_pcr[0]:.4f} M{repl_pcr[1]:.4f} H{repl_pcr[2]:.4f}  "
        f"vs PRIMARY [0.9955/0.9695/0.9775]  guard={'PASS' if repl_pass else 'FAIL'}")

    # Emit submissions for the candidates
    ids = pd.read_csv("data/test.csv")["id"].values
    summary = dict(
        lb3_bal=float(lb3_bal),
        primary_bal=float(primary_bal),
        meta_3wnn_raw_bal=float(raw_bal),
        meta_3wnn_full_iso_bal=float(full_iso_bal),
        meta_3wnn_pf_iso_bal=float(pf_iso_bal),
        full_sweep=full_rows, pf_sweep=pf_rows,
        full_best=full_best, pf_best=pf_best,
        primary_arch_pf_iso=dict(bal=float(repl_bal),
                                  delta_vs_primary=float(repl_bal-primary_bal),
                                  pcr=repl_pcr.tolist(),
                                  guard_pass=bool(repl_pass)),
        emitted=[],
    )

    if pf_best and pf_best["delta"] >= EMIT_DELTA:
        path = emit(f"perfold_iso_lb3", lb3_t, pf_iso_t, pf_best["alpha"], ids)
        summary["emitted"].append(dict(label="perfold_iso_lb3", row=pf_best,
                                       submission=path))
    # Also emit the PRIMARY-arch swap if it passes
    if repl_pass and (repl_bal - primary_bal) >= EMIT_DELTA:
        path = emit("primaryarch_perfoldiso", lb3_t, pf_iso_t, 0.30, ids)
        summary["emitted"].append(dict(label="primaryarch_perfoldiso",
                                       delta_vs_primary=float(repl_bal-primary_bal),
                                       submission=path))

    out = ART / "r5_perfold_iso_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
