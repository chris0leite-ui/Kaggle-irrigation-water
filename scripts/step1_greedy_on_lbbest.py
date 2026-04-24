"""Step 1: Greedy forward-selection ANCHORED ON the current LB-best 3-stack
(OOF 0.98061 / LB 0.98008), searching for a step-4 lift.

Anchor construction:
  lb3     = log_blend(recipe, pseudo_s1, pseudo_s7;           0.25/0.35/0.40)
  stack1  = log_blend(lb3, realmlp;                           0.80/0.20)
  stack2  = log_blend(stack1, xgb_nonrule__iso;               0.925/0.075)

Pool: every saved OOF/test pair in scripts/artifacts, minus EXCLUDE_FROM_POOL
+ EXCLUDE_GREEDY_ADD (consistent with c0_safe_greedy_v3 W5 guardrail).
Each component is made available raw and isotonic-calibrated.

Emit gate: final OOF Δ ≥ +1e-4 for a new submission file.
Per-step gate: stop below +1e-4.
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
LB_BEST_STACK_LB = 0.98008

EXCLUDE_FROM_POOL = {
    "soft_distill",
    "xgb_spec_678",
    "recipe_pseudolabel_stage2",
}
EXCLUDE_GREEDY_ADD = EXCLUDE_FROM_POOL | {
    "recipe_pseudolabel_seed7labeler",
    "recipe_pseudolabel_seed123labeler",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def errmask(p, y):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1) != y


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


def build_anchor(y):
    """Reconstruct the current LB-best 3-stack OOF + test."""
    recipe = (_normed(np.load(ART / "oof_recipe_full_te.npy")),
              _normed(np.load(ART / "test_recipe_full_te.npy")))
    pseudo_s1 = (_normed(np.load(ART / "oof_recipe_pseudolabel.npy")),
                 _normed(np.load(ART / "test_recipe_pseudolabel.npy")))
    pseudo_s7 = (_normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")),
                 _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")))
    realmlp = (_normed(np.load(ART / "oof_realmlp.npy")),
               _normed(np.load(ART / "test_realmlp.npy")))
    nonrule = (_normed(np.load(ART / "oof_xgb_nonrule.npy")),
               _normed(np.load(ART / "test_xgb_nonrule.npy")))

    # isotonic for nonrule (matches emit_realmlp_3stack)
    nr_iso_oof, nr_iso_test = iso_cal(nonrule[0], nonrule[1], y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_oof = log_blend([recipe[0], pseudo_s1[0], pseudo_s7[0]], w3)
    lb3_test = log_blend([recipe[1], pseudo_s1[1], pseudo_s7[1]], w3)

    stack1_oof = log_blend([lb3_oof, realmlp[0]], np.array([0.8, 0.2]))
    stack1_test = log_blend([lb3_test, realmlp[1]], np.array([0.8, 0.2]))

    stack2_oof = log_blend([stack1_oof, nr_iso_oof], np.array([0.925, 0.075]))
    stack2_test = log_blend([stack1_test, nr_iso_test], np.array([0.925, 0.075]))

    picked = {"recipe_full_te", "recipe_pseudolabel",
              "recipe_pseudolabel_seed7labeler",
              "realmlp", "xgb_nonrule"}
    log(f"anchor (current LB-best 3-stack): OOF={bal(stack2_oof, y):.5f}  "
        f"errs={int(errmask(stack2_oof, y).sum())}")
    return stack2_oof, stack2_test, picked


def load_pool(y):
    """Load every oof_*.npy + test_*.npy pair, minus EXCLUDE_FROM_POOL.
    Provide both raw and isotonic-calibrated copies."""
    pool = {}
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        if name in EXCLUDE_FROM_POOL:
            continue
        test_p = ART / f"test_{name}.npy"
        if not test_p.exists():
            continue
        try:
            raw_o = np.load(oof_p).astype(np.float32)
            raw_t = np.load(test_p).astype(np.float32)
        except Exception as e:
            log(f"  skip {name}: {e}")
            continue
        # must be 3-class prob (binary/sparse carriers skipped)
        if raw_o.ndim != 2 or raw_o.shape[1] != 3:
            continue
        oof = _normed(raw_o)
        test = _normed(raw_t)
        oof_i, test_i = iso_cal(oof, test, y)
        pool[name] = (oof, test)
        pool[f"{name}__iso"] = (oof_i, test_i)
    log(f"  {len(pool)//2} components loaded (× 2 for raw/iso)")
    return pool


def greedy(anchor_oof, anchor_test, picked_bases, pool, y, max_steps=6):
    alphas = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    oof_cur, test_cur = anchor_oof, anchor_test
    bal_cur = bal(oof_cur, y)
    picked = set(picked_bases)
    chosen = []

    for step in range(1, max_steps + 1):
        best = None
        for key, (oof_k, test_k) in pool.items():
            base = key.replace("__iso", "")
            if base in picked or base in EXCLUDE_GREEDY_ADD:
                continue
            for a in alphas:
                ot = log_blend([oof_cur, oof_k], np.array([1 - a, a]))
                s = bal(ot, y)
                if best is None or s > best[0]:
                    best = (s, key, base, a, ot, test_k)
        if best is None:
            log("  no candidate remaining; stop")
            break
        s, key, base, a, ot, tt = best
        d = s - bal_cur
        log(f"  step{step}: + {key:50s} α={a:.3f}  OOF={s:.5f}  Δ={d:+.5f}")
        if d < 1e-4:
            log("  stop (below +1e-4 gate)")
            break
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

    anchor_oof, anchor_test, picked = build_anchor(y)
    pool = load_pool(y)

    log("=" * 70)
    log("greedy forward from LB-best 3-stack (OOF 0.98061)")
    bal_final, oof_final, test_final, chosen = greedy(
        anchor_oof, anchor_test, picked, pool, y
    )

    start = bal(anchor_oof, y)
    delta = bal_final - start
    log("=" * 70)
    log(f"anchor OOF = {start:.5f}")
    log(f"final OOF  = {bal_final:.5f}")
    log(f"Δ vs LB-best 3-stack = {delta:+.5f}")
    log(f"chosen: {chosen}")

    summary = dict(
        anchor_oof=float(start),
        final_oof=float(bal_final),
        delta=float(delta),
        chosen=chosen,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "step1_greedy_on_lbbest_results.json").write_text(
        json.dumps(summary, indent=2))
    log(f"wrote scripts/artifacts/step1_greedy_on_lbbest_results.json")

    if delta >= 1e-4:
        np.save(ART / "oof_step1_greedy_lbbest.npy", oof_final.astype(np.float32))
        np.save(ART / "test_step1_greedy_lbbest.npy", test_final.astype(np.float32))
        sample = pd.read_csv(DATA / "sample_submission.csv")
        pred = (np.log(np.clip(test_final, 1e-12, 1)) + BIAS).argmax(1)
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / "submission_step1_greedy_lbbest.csv"
        sub.to_csv(path, index=False)
        log(f"wrote {path}  (expected LB ~{LB_BEST_STACK_LB + delta * 0.5:.5f})")
        log(f"class dist: {sub[TARGET].value_counts().to_dict()}")
    else:
        log("no lift above anchor; no submission emitted.")


if __name__ == "__main__":
    main()
