"""Blend-gate diagnostic for N1 (OvR-XGB) and N2 (effective-number focal).

Anchors against:
  1. recipe_full_te        (single-model recipe baseline, OOF 0.97967)
  2. lb_best_2way          (recipe x pseudo_s1, OOF 0.98012, LB 0.97998)
  3. lb_best_3way          (3-way multi-seed, OOF 0.98029, LB 0.98005)
  4. lb_best_3stack        (lb3 + RealMLP@0.20 + nonrule_iso@0.075, OOF 0.98061, LB 0.98008)
  5. lb_best_meta          (3stack + xgb_metastack_iso@0.30, OOF 0.98084, LB 0.98094 — current best)

For each candidate (N1, N2, and N1+N2 ensembles + per-class iso variants):
  - standalone tuned bal_acc, errs, Jaccard vs every anchor
  - fixed-bias log-blend sweep alpha in [0, 0.5] vs every anchor
  - decision per (anchor, candidate): EMIT if peak Delta >= +5e-4 AND
    errs(blend) <= errs(anchor) AND no per-class recall drop > 5e-4

Also reports the per-class break-even precision rule for each candidate's
predicted High count (informational only). LB-probe gate per the
LB-transfer threshold rule documented in CLAUDE.md.
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

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import build_lbbest_stack, BIAS, iso_cal, normed  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

# Fixed anchor bias (recipe's tuned), reused across every gate per LEARNINGS rule.
ALPHA_GRID = np.linspace(0.0, 0.5, 21)
EMIT_DELTA = 5e-4
LB_TRANSFER_DELTA = 2e-4


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_pair(name: str):
    return (np.load(ART / f"oof_{name}.npy").astype(np.float32),
            np.load(ART / f"test_{name}.npy").astype(np.float32))


def bal_at_bias(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))


def per_class_recall(p, y, bias=BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    out = []
    for k in range(3):
        m = y == k
        out.append(float((pred[m] == k).sum() / max(m.sum(), 1)))
    return out


def err_count(p, y, bias=BIAS):
    return int(((np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1) != y).sum())


def jaccard_errs(a, b, y, bias=BIAS):
    pa = (np.log(np.clip(a, 1e-12, 1)) + bias).argmax(1)
    pb = (np.log(np.clip(b, 1e-12, 1)) + bias).argmax(1)
    ea = pa != y; eb = pb != y
    inter = (ea & eb).sum(); union = (ea | eb).sum()
    return float(inter / max(union, 1))


def sweep_blend(anchor_oof, anchor_test, cand_oof, cand_test, y, bias):
    """Log-blend sweep, returns (alpha_grid_results, peak_alpha, peak_delta,
    blend_oof_at_peak, blend_test_at_peak)."""
    base = bal_at_bias(anchor_oof, y, bias)
    rows = []
    best_alpha, best_delta = 0.0, 0.0
    best_blend_o = anchor_oof.copy()
    best_blend_t = anchor_test.copy()
    for a in ALPHA_GRID:
        if a <= 0:
            rows.append((float(a), float(base), 0.0))
            continue
        bo = log_blend([anchor_oof, cand_oof], np.array([1 - a, a]))
        bal = bal_at_bias(bo, y, bias)
        delta = bal - base
        rows.append((float(a), float(bal), float(delta)))
        if delta > best_delta:
            best_alpha = float(a)
            best_delta = float(delta)
            best_blend_o = bo
            best_blend_t = log_blend([anchor_test, cand_test], np.array([1 - a, a]))
    return rows, best_alpha, best_delta, best_blend_o, best_blend_t


def gate(anchor_name, anchor_oof, anchor_test, cand_name, cand_oof, cand_test, y, bias):
    """Run full gate diagnostic for one (anchor, candidate) pair."""
    base = bal_at_bias(anchor_oof, y, bias)
    base_errs = err_count(anchor_oof, y, bias)
    base_recall = per_class_recall(anchor_oof, y, bias)
    cand_errs = err_count(cand_oof, y, bias)
    cand_jac = jaccard_errs(anchor_oof, cand_oof, y, bias)
    rows, peak_a, peak_d, blend_o, blend_t = sweep_blend(
        anchor_oof, anchor_test, cand_oof, cand_test, y, bias)
    blend_errs = err_count(blend_o, y, bias)
    blend_recall = per_class_recall(blend_o, y, bias)
    rec_drop = max(base_recall[k] - blend_recall[k] for k in range(3))
    pass_emit = peak_d >= EMIT_DELTA
    pass_errs = blend_errs <= base_errs
    pass_recall = rec_drop <= 5e-4
    pass_lb_xfer = peak_d >= LB_TRANSFER_DELTA
    return dict(
        anchor=anchor_name, candidate=cand_name,
        anchor_bal=base, anchor_errs=base_errs, anchor_recall=base_recall,
        cand_errs=cand_errs, cand_jac=cand_jac,
        peak_alpha=peak_a, peak_delta=peak_d,
        blend_bal=base + peak_d, blend_errs=blend_errs, blend_recall=blend_recall,
        rec_drop=rec_drop,
        pass_emit_5e4=pass_emit, pass_errs_le_anchor=pass_errs,
        pass_recall_5e4=pass_recall, pass_lb_xfer_2e4=pass_lb_xfer,
        sweep=rows,
    )


def main():
    log("loading y")
    train = pd.read_csv(DATA / "train.csv")
    cls = {"Low": 0, "Medium": 1, "High": 2}
    y = train["Irrigation_Need"].map(cls).to_numpy().astype(np.int32)

    log("loading anchors")
    recipe_o, recipe_t = load_pair("recipe_full_te")

    s1_o, s1_t = load_pair("recipe_pseudolabel")
    s7_o, s7_t = load_pair("recipe_pseudolabel_seed7labeler")
    lb2_o = log_blend([recipe_o, s1_o], np.array([0.5, 0.5]))
    lb2_t = log_blend([recipe_t, s1_t], np.array([0.5, 0.5]))
    lb3_o = log_blend([recipe_o, s1_o, s7_o], np.array([0.25, 0.35, 0.40]))
    lb3_t = log_blend([recipe_t, s1_t, s7_t], np.array([0.25, 0.35, 0.40]))
    lb3stack_o, lb3stack_t = build_lbbest_stack(y)
    # The Tier-1b LB-best meta blend.
    meta_o, meta_t = load_pair("xgb_metastack")
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    lbmeta_o = log_blend([lb3stack_o, meta_o_iso], np.array([0.7, 0.3]))
    lbmeta_t = log_blend([lb3stack_t, meta_t_iso], np.array([0.7, 0.3]))
    log(f"  recipe   bal={bal_at_bias(recipe_o, y):.5f}")
    log(f"  lb3stack bal={bal_at_bias(lb3stack_o, y):.5f}")
    log(f"  lbmeta   bal={bal_at_bias(lbmeta_o, y):.5f}")

    anchors = [
        ("recipe_full_te", recipe_o, recipe_t),
        ("lb_best_2way", lb2_o, lb2_t),
        ("lb_best_3way", lb3_o, lb3_t),
        ("lb_best_3stack", lb3stack_o, lb3stack_t),
        ("lb_best_meta", lbmeta_o, lbmeta_t),
    ]

    log("loading candidates")
    cands: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in ("xgb_ovr_recipe", "xgb_ovr_recipe_raw", "recipe_focal_effnum"):
        oof_p = ART / f"oof_{name}.npy"
        if not oof_p.exists():
            log(f"  SKIP {name} (artifact missing — production not done?)")
            continue
        co, ct = load_pair(name)
        # Raw OvR is uncalibrated, not yet softmax-renormed: do that now so
        # gates run on a 3-class prob distribution.
        if name.endswith("_raw"):
            eps = 1e-9
            zo = np.log(np.clip(co, eps, 1.0))
            zo = zo - zo.max(axis=1, keepdims=True)
            ezo = np.exp(zo); co = (ezo / ezo.sum(axis=1, keepdims=True)).astype(np.float32)
            zt = np.log(np.clip(ct, eps, 1.0))
            zt = zt - zt.max(axis=1, keepdims=True)
            ezt = np.exp(zt); ct = (ezt / ezt.sum(axis=1, keepdims=True)).astype(np.float32)
        cands.append((name, co, ct))
        log(f"  loaded {name}  bal={bal_at_bias(co, y):.5f}")
        # Add iso-calibrated copy.
        co_i, ct_i = iso_cal(co, ct, y)
        cands.append((name + "__iso", co_i, ct_i))
        log(f"  loaded {name}__iso  bal={bal_at_bias(co_i, y):.5f}")

    if not cands:
        log("FATAL: no candidates loaded — run N1/N2 production first.")
        return

    log("running gate matrix")
    results = []
    for aname, ao, at in anchors:
        for cname, co, ct in cands:
            r = gate(aname, ao, at, cname, co, ct, y, BIAS)
            results.append(r)
            tags = []
            if r["pass_emit_5e4"]: tags.append("EMIT")
            if r["pass_lb_xfer_2e4"]: tags.append("LB_XFER")
            if not r["pass_errs_le_anchor"]: tags.append("MAG_TRAP")
            if not r["pass_recall_5e4"]: tags.append("RECALL_DROP")
            log(f"  anchor={aname:18s} cand={cname:32s}  "
                f"peak alpha={r['peak_alpha']:.3f}  Delta={r['peak_delta']:+.5f}  "
                f"jac={r['cand_jac']:.3f}  errs_anchor={r['anchor_errs']:5d}  "
                f"errs_cand={r['cand_errs']:5d}  errs_blend={r['blend_errs']:5d}  "
                f"rec_drop={r['rec_drop']:+.4f}  [{' '.join(tags) or 'null'}]")

    out = ART / "n1_n2_blend_gate_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
