"""Final blend gate over the LB-best 4-stack + meta variants.

Loads:
- LB-best 4-stack (the LB 0.98094 candidate's anchor stack: lb_3way + RealMLP +
  xgb_nonrule_iso + xgb_metastack_iso). Reconstructs from saved components.
- Tier-1b v3 meta (cross-pollinated 5 new components into the v1 pool).
- Variants B and C (different XGB depth/seed/colsample on same pool).

Each meta is per-class isotonic-calibrated. We try:
1. v3 alone replacing the v1 meta in the LB-best 4-stack.
2. Equal-weight log-blend of {v1, v3, B, C} → iso → α-sweep into LB-best 3-stack.
3. Greedy forward selection over the 4 metas, anchored on the LB-best 3-stack.

Decision rule (all enforced):
- best_blend_OOF > LB-best 4-stack OOF (0.98084) by ≥ +2e-4
- Jaccard with LB-best 4-stack < 0.97 (avoid pure-noise rearrangement)
- per-class recall: no class drops by ≥ 0.0005 vs LB-best 4-stack
Emit submission only if all three pass. NEVER retune log-bias on the blend
(binhigh-rule).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET, bal_at_bias, build_lbbest_stack,
    iso_cal, load_y, log, normed,
)


def jaccard(p1, p2, y):
    a1 = (np.log(np.clip(p1, 1e-12, 1)) + BIAS).argmax(1)
    a2 = (np.log(np.clip(p2, 1e-12, 1)) + BIAS).argmax(1)
    e1 = a1 != y
    e2 = a2 != y
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return float(inter / max(union, 1))


def per_class_recall(p, y):
    a = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    return recall_score(y, a, average=None, labels=[0, 1, 2])


def emit_if_passes(blend_oof, blend_test, anchor_oof, anchor_test, name, y):
    rec_anchor = per_class_recall(anchor_oof, y)
    bal_anchor = bal_at_bias(anchor_oof, y)
    rec_blend = per_class_recall(blend_oof, y)
    bal_blend = bal_at_bias(blend_oof, y)
    delta = bal_blend - bal_anchor
    j = jaccard(blend_oof, anchor_oof, y)
    drop_class = ((rec_blend - rec_anchor) <= -0.0005).any()
    log(f"  {name}: OOF={bal_blend:.5f} Δ={delta:+.5f} J={j:.4f} "
        f"rec_L={rec_blend[0]:.4f} rec_M={rec_blend[1]:.4f} rec_H={rec_blend[2]:.4f}")
    if delta >= 2e-4 and j < 0.97 and not drop_class:
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in (np.log(np.clip(blend_test, 1e-12, 1)) + BIAS).argmax(1)]
        path = SUB / f"submission_tier1b_{name}.csv"
        sub.to_csv(path, index=False)
        log(f"    PASS → wrote {path}")
        return True
    return False


def lb_best_4stack(y, meta_v1_oof, meta_v1_test):
    """Reconstructs LB 0.98094 = LB-best-3-stack + meta_v1_iso @ α=0.30."""
    lb3_o, lb3_t = build_lbbest_stack(y)
    iso_o, iso_t = iso_cal(meta_v1_oof, meta_v1_test, y)
    o = log_blend([lb3_o, iso_o], np.array([0.7, 0.3]))
    t = log_blend([lb3_t, iso_t], np.array([0.7, 0.3]))
    return o, t


def main():
    t0 = time.time()
    log("loading y + LB-best 3-stack")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF = {bal_at_bias(lb3_o, y):.5f}")

    metas = {}
    for tag, name in [("v1", "xgb_metastack"), ("v3", "xgb_metastack_v3"),
                      ("B", "xgb_metastack_varB"), ("C", "xgb_metastack_varC")]:
        op = ART / f"oof_{name}.npy"
        tp = ART / f"test_{name}.npy"
        if op.exists() and tp.exists():
            o = normed(np.load(op).astype(np.float32))
            t = normed(np.load(tp).astype(np.float32))
            iso_o, iso_t = iso_cal(o, t, y)
            metas[tag] = (o, t, iso_o, iso_t)
            log(f"  loaded meta {tag} ({name}) OOF={bal_at_bias(o, y):.5f} iso={bal_at_bias(iso_o, y):.5f}")
        else:
            log(f"  meta {tag} ({name}) NOT FOUND — skipping")

    if "v1" not in metas:
        log("FATAL: v1 meta missing; cannot reconstruct LB-best 4-stack")
        return
    anchor_o, anchor_t = lb_best_4stack(y, metas["v1"][0], metas["v1"][1])
    anchor_bal = bal_at_bias(anchor_o, y)
    log(f"\n  LB-best 4-stack (anchor) OOF = {anchor_bal:.5f} (expected ≈0.98084)")

    log(f"\n=== Strategy 1: replace v1 with single new meta ===")
    for tag in ["v3", "B", "C"]:
        if tag not in metas:
            continue
        o = log_blend([lb3_o, metas[tag][2]], np.array([0.7, 0.3]))
        t = log_blend([lb3_t, metas[tag][3]], np.array([0.7, 0.3]))
        emit_if_passes(o, t, anchor_o, anchor_t, f"replace_{tag}_a030", y)

    log(f"\n=== Strategy 2: equal-weight ensemble of metas (iso) ===")
    keys = list(metas.keys())
    for combo in [("v1", "v3"), ("v1", "v3", "B"), ("v1", "v3", "B", "C")]:
        avail = [k for k in combo if k in metas]
        if len(avail) < 2:
            continue
        os_, ts_ = [metas[k][2] for k in avail], [metas[k][3] for k in avail]
        w = np.ones(len(avail)) / len(avail)
        ens_o = log_blend(os_, w)
        ens_t = log_blend(ts_, w)
        for a in [0.2, 0.25, 0.3, 0.35, 0.4]:
            o = log_blend([lb3_o, ens_o], np.array([1 - a, a]))
            t = log_blend([lb3_t, ens_t], np.array([1 - a, a]))
            tag = f"ens_{'_'.join(avail)}_a{int(a*1000):03d}"
            emit_if_passes(o, t, anchor_o, anchor_t, tag, y)

    log(f"\n=== Strategy 3: greedy forward selection over iso metas ===")
    candidates = {k: (metas[k][2], metas[k][3]) for k in keys}
    chosen: list[tuple[str, float]] = []
    cur_o, cur_t = lb3_o, lb3_t
    cur_bal = bal_at_bias(cur_o, y)
    for step in range(4):
        best_tag, best_a, best_b = None, 0.0, cur_bal
        for tag, (oo, tt) in candidates.items():
            if any(c[0] == tag for c in chosen):
                continue
            for a in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                bo = log_blend([cur_o, oo], np.array([1 - a, a]))
                bb = bal_at_bias(bo, y)
                if bb > best_b + 1e-5:
                    best_tag, best_a, best_b = tag, a, bb
        if best_tag is None:
            log(f"  step {step+1}: no candidate improves (>1e-5), stopping")
            break
        oo, tt = candidates[best_tag]
        cur_o = log_blend([cur_o, oo], np.array([1 - best_a, best_a]))
        cur_t = log_blend([cur_t, tt], np.array([1 - best_a, best_a]))
        cur_bal = best_b
        chosen.append((best_tag, best_a))
        log(f"  step {step+1}: + {best_tag} α={best_a:.3f} OOF={best_b:.5f}")
    if chosen:
        emit_if_passes(cur_o, cur_t, anchor_o, anchor_t,
                       f"greedy_{'_'.join(c[0] for c in chosen)}", y)

    log(f"\nelapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
