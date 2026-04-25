"""Next move 1: Greedy forward from the NEW LB-best stack (OOF 0.98084 / LB 0.98094).

Construction of the new anchor:
  lb3      = log_blend(recipe_full_te, recipe_pseudolabel, recipe_pseudolabel_seed7labeler;
                        0.25/0.35/0.40)
  stack1   = log_blend(lb3, realmlp;                    0.80/0.20)
  stack2   = log_blend(stack1, xgb_nonrule__iso;        0.925/0.075)
  stack3   = log_blend(stack2, xgb_metastack__iso;      0.700/0.300)   ← LB-best 0.98094

Pool: every saved OOF/test pair minus EXCLUDE_FROM_POOL/EXCLUDE_GREEDY_ADD.
Each component appears raw + isotonic-calibrated. Finer α grid (0.01..0.5).
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

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])
LB_NEW_BEST = 0.98094

EXCLUDE_FROM_POOL = {
    "soft_distill", "soft_distill_small", "soft_distill_tiny",
    "xgb_spec_678",
    "recipe_pseudolabel_stage2",
    "spec_mh_v3_score5", "spec_mh_v3_score6",
    "spec6_mh", "spec6_mh_v2",
    "xgb_bin_medium", "xgb_bin_high", "binhigh", "p_flip", "pflip",
    "missed_high", "flip_correction",
    "selective_router", "disagree_meta",
    "c0_safe_lb_best_2way", "c0_safe_recipe_full_te",
    "c0_v2_lb_best_2way", "c0_v2_lb_best_3way", "c0_v2_recipe_full_te",
    "c0_v3_lb_best_3way", "c0_v3_recipe_full_te",
    "b2_groupkfold_region", "step1_greedy_lbbest",
    "hybrid_binhigh", "meta_v3", "eb_cell",
    "spec_lm_v3_score3",  # binary, not 3-class
    "tta_recipe_baseline",  # = recipe_full_te
    "tier1b_greedy_meta",   # = the new anchor (would be circular)
    "step1_greedy_lbbest",
}
EXCLUDE_GREEDY_ADD = EXCLUDE_FROM_POOL | {
    "recipe_pseudolabel_seed7labeler",   # anchor ingredient
    "recipe_pseudolabel_seed123labeler",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
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
    nr_iso_o, nr_iso_t = iso_cal(nr, nrt, y)
    meta = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    metat = _normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_iso_o, meta_iso_t = iso_cal(meta, metat, y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
    st3_o = log_blend([st2_o, meta_iso_o], np.array([0.7, 0.3]))
    st3_t = log_blend([st2_t, meta_iso_t], np.array([0.7, 0.3]))

    picked = {"recipe_full_te", "recipe_pseudolabel",
              "recipe_pseudolabel_seed7labeler", "realmlp", "xgb_nonrule",
              "xgb_metastack"}
    return st3_o, st3_t, picked


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
        oof_i, test_i = iso_cal(oof, test, y)
        pool[name] = (oof, test)
        pool[f"{name}__iso"] = (oof_i, test_i)
    return pool


def greedy(oof_cur, test_cur, picked, pool, y, max_steps=8):
    alphas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15,
              0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
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
        log(f"  step{step}: + {key:60s} α={a:.3f}  OOF={s:.5f}  Δ={d:+.5f}")
        if d < 5e-5:
            log("  stop (below +5e-5 internal gate)"); break
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

    log("building NEW LB-best stack3 anchor (lb3 + realmlp + nonrule_iso + meta_iso)")
    oof_anchor, test_anchor, picked = build_anchor(y)
    log(f"  anchor OOF = {bal(oof_anchor, y):.5f}  (LB 0.98094)")

    pool = load_pool(y)
    log(f"pool: {len(pool)//2} base components (+iso copies)")

    log("\n=== greedy from NEW LB-best (finer α grid 0.005..0.5, +iso) ===")
    bal_f, oof_f, test_f, chosen = greedy(oof_anchor, test_anchor, picked.copy(), pool, y)
    start = bal(oof_anchor, y)
    delta = bal_f - start
    log(f"\nanchor OOF = {start:.5f}")
    log(f"final OOF  = {bal_f:.5f}  Δ={delta:+.5f}")
    log(f"chosen: {chosen}")

    out = dict(anchor_oof=float(start), final_oof=float(bal_f), delta=float(delta),
               chosen=chosen, elapsed_sec=float(time.time() - t0))
    (ART / "next_greedy_on_meta_stack_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote scripts/artifacts/next_greedy_on_meta_stack_results.json")

    if delta >= 2e-4:
        np.save(ART / "oof_next_greedy_meta_stack.npy", oof_f.astype(np.float32))
        np.save(ART / "test_next_greedy_meta_stack.npy", test_f.astype(np.float32))
        pred = (np.log(np.clip(test_f, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / "submission_next_greedy_meta_stack.csv"
        sub.to_csv(path, index=False)
        log(f"wrote {path}  (Δ={delta:+.5f} ≥ +2e-4 LB-transfer threshold)")
    elif delta >= 5e-5:
        log(f"Δ={delta:+.5f} ≥ +5e-5 internal gate but below +2e-4 LB-transfer threshold")
    else:
        log(f"Δ={delta:+.5f} below internal gate; no submission")


if __name__ == "__main__":
    main()
