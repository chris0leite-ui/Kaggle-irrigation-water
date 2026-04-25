"""Audit F2 fix: per-fold isotonic calibration for the meta-stacker greedy.

Mirrors `tier1b_greedy_with_meta.py` but swaps `iso_cal` for a per-fold
honest version: for row i in fold k, the iso function applied to oof[i]
is fit on rows in folds != k. Test transform uses iso fit on full OOF
(test rows are never in OOF training, so no leak there).

Compares the resulting greedy OOF against the current primary's 0.98084.

  - If new OOF >= 0.98075:  iso-cal-on-full-OOF was contributing minimal
    inflation; current primary's lift is genuine. Lock primary.
  - If new OOF in [0.98050, 0.98075):  some inflation, but lift mostly
    survives. Still safe to lock primary; mention in audit final.
  - If new OOF < 0.98050:  inflation was material. Recompute test-side
    primary using per-fold iso pipeline + LB-probe one slot.

Reads same OOF files as tier1b_greedy_with_meta. Writes:
  oof_tier1b_greedy_meta_perfoldiso.npy
  test_tier1b_greedy_meta_perfoldiso.npy
  tier1b_greedy_meta_perfoldiso_results.json
  submissions/submission_tier1b_greedy_meta_perfoldiso.csv (only if
                                                             OOF lift >= +2e-4)
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
BIAS = np.array([1.4324, 1.4689, 3.4008])
SEED = 42
N_FOLDS = 5

EXCLUDE_FROM_POOL = {
    "soft_distill", "xgb_spec_678", "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6", "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction",
    "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region", "b2_groupkfold_crop",
    "step1_greedy_lbbest", "hybrid_binhigh", "meta_v3", "eb_cell",
    "spec_lm_v3_score3", "tta_recipe_baseline",
    "tier1b_greedy_meta", "tier1b_greedy_meta_perfoldiso",  # don't loop on selves
}
EXCLUDE_GREEDY_ADD = EXCLUDE_FROM_POOL | {
    "recipe_pseudolabel_seed7labeler",
    "recipe_pseudolabel_seed123labeler",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal_perfold(oof, test, y):
    """Honest per-fold isotonic. For each row i in fold k, iso function
    is fit on (oof[!=fold_k], y[!=fold_k]) and applied to oof[i]. Test
    uses full-OOF iso (no leak — test rows aren't in OOF training)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oo = np.zeros_like(oof, dtype=np.float32)
    for tr_idx, va_idx in skf.split(oof, y):
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip",
                                    y_min=1e-6, y_max=1 - 1e-6)
            ir.fit(oof[tr_idx, c], (y[tr_idx] == c).astype(np.float32))
            oo[va_idx, c] = ir.predict(oof[va_idx, c])
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def bal(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def build_anchor(y):
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    rmt = _normed(np.load(ART / "test_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    # *** PER-FOLD ISO (audit F2 fix) ***
    nr_iso_o, nr_iso_t = iso_cal_perfold(nr, nrt, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
    picked = {"recipe_full_te", "recipe_pseudolabel",
              "recipe_pseudolabel_seed7labeler", "realmlp", "xgb_nonrule"}
    return st2_o, st2_t, picked


def load_pool(y):
    pool = {}
    for p in sorted(ART.glob("oof_*.npy")):
        name = p.stem.replace("oof_", "", 1)
        if name in EXCLUDE_FROM_POOL:
            continue
        test_p = ART / f"test_{name}.npy"
        if not test_p.exists():
            continue
        try:
            o = np.load(p).astype(np.float32)
            t = np.load(test_p).astype(np.float32)
        except Exception:
            continue
        if o.ndim != 2 or o.shape[1] != 3:
            continue
        oof = _normed(o); test = _normed(t)
        # *** PER-FOLD ISO for every iso pool entry (audit F2 fix) ***
        oof_i, test_i = iso_cal_perfold(oof, test, y)
        pool[name] = (oof, test)
        pool[f"{name}__iso"] = (oof_i, test_i)
    return pool


def greedy(oof_cur, test_cur, picked, pool, y, max_steps=8):
    alphas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15,
              0.2, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.5]
    bal_cur = bal(oof_cur, y)
    chosen = []
    for step in range(1, max_steps + 1):
        best = None
        for key, (o_k, t_k) in pool.items():
            base = key.replace("__iso", "")
            if base in picked or base in EXCLUDE_GREEDY_ADD:
                continue
            for a in alphas:
                ot = log_blend([oof_cur, o_k], np.array([1 - a, a]))
                s = bal(ot, y)
                if best is None or s > best[0]:
                    best = (s, key, base, a, ot, t_k)
        if best is None:
            log("  no candidate; stop"); break
        s, key, base, a, ot, tt = best
        d = s - bal_cur
        log(f"  step{step}: + {key:50s} α={a:.3f}  OOF={s:.5f}  Δ={d:+.5f}")
        if d < 5e-5:
            log("  stop (below +5e-5 gate)"); break
        chosen.append((key, float(a)))
        picked.add(base)
        oof_cur = ot
        test_cur = log_blend([test_cur, tt], np.array([1 - a, a]))
        bal_cur = s
    return bal_cur, oof_cur, test_cur, chosen


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    log("=== AUDIT F2 FIX: per-fold isotonic ===")
    log("building LB-best 3-stack anchor (with PER-FOLD iso on nonrule)")
    oof_anchor, test_anchor, picked = build_anchor(y)
    anchor_oof = bal(oof_anchor, y)
    log(f"  anchor OOF (per-fold iso) = {anchor_oof:.5f}")
    log(f"  for ref, original full-OOF iso anchor = 0.98061 (per CLAUDE.md)")

    pool = load_pool(y)
    log(f"pool: {len(pool)//2} base components (+per-fold-iso copies)")
    in_pool = "xgb_metastack" in pool
    log(f"  xgb_metastack in pool: {in_pool}")
    assert in_pool, "meta-stacker artefact missing"

    log("\n=== greedy (per-fold iso pool) ===")
    bal_f, oof_f, test_f, chosen = greedy(
        oof_anchor, test_anchor, picked.copy(), pool, y)
    delta = bal_f - anchor_oof
    log(f"\nanchor OOF        = {anchor_oof:.5f}")
    log(f"final OOF         = {bal_f:.5f}  Δ vs anchor = {delta:+.5f}")
    log(f"chosen: {chosen}")

    # Compare against current primary OOF (0.98084) and LB (0.98094)
    primary_oof = 0.98084
    delta_vs_primary = bal_f - primary_oof
    log(f"\nΔ vs current primary OOF (0.98084) = {delta_vs_primary:+.5f}")
    if bal_f >= 0.98075:
        verdict = "GREEN: iso-on-full-OOF was contributing minimal inflation; current primary's lift is genuine."
    elif bal_f >= 0.98050:
        verdict = "YELLOW: some OOF inflation, but lift mostly survives. Lock primary; document in audit."
    else:
        verdict = "RED: meaningful inflation. Consider rebuilding test-side primary with per-fold iso + LB probe."
    log(f"\nVERDICT: {verdict}")

    out = dict(
        anchor_oof=float(anchor_oof),
        final_oof=float(bal_f),
        delta_anchor_to_final=float(delta),
        primary_oof_reference=primary_oof,
        delta_vs_primary=float(delta_vs_primary),
        chosen=chosen,
        verdict=verdict,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "tier1b_greedy_meta_perfoldiso_results.json").write_text(
        json.dumps(out, indent=2))

    np.save(ART / "oof_tier1b_greedy_meta_perfoldiso.npy",
            oof_f.astype(np.float32))
    np.save(ART / "test_tier1b_greedy_meta_perfoldiso.npy",
            test_f.astype(np.float32))
    log(f"wrote scripts/artifacts/oof_tier1b_greedy_meta_perfoldiso.npy + test")

    if delta_vs_primary >= 2e-4:
        # Only if the per-fold variant beats current primary by enough
        # to justify a swap candidate.
        pred = (np.log(np.clip(test_f, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / "submission_tier1b_greedy_meta_perfoldiso.csv"
        sub.to_csv(path, index=False)
        log(f"wrote {path}")
    else:
        log(f"Δ vs primary < +2e-4; no submission emitted (diagnostic only)")


if __name__ == "__main__":
    main()
