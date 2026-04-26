"""Probe SUBSETS of LB-validated subs to find any independent diversity.

Most subs share the recipe-pseudo backbone. Only recipe (raw) and
catboost are model-family-distinct. Probe specific subsets:
  - {primary, recipe, catboost_iso}: 3 model views
  - {primary, catboost_iso}: 2 family-distinct
  - finer α-grid greedy forward from primary
  - blend-gate sweep onto primary for S3 hard-vote (close non-S5 result)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from own_ensemble_helpers import LB_SCORES, reconstruct_lb_validated_set  # noqa: E402
from tier1b_helpers import BIAS, iso_cal, load_y  # noqa: E402

ART = Path("scripts/artifacts")
EPS = 1e-12


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def predict(p):
    return (np.log(np.clip(p, EPS, 1.0)) + BIAS).argmax(1)


def bal(p, y):
    return balanced_accuracy_score(y, predict(p))


def errs(p, y):
    return int((y != predict(p)).sum())


def pcr(p, y):
    pp = predict(p)
    return [float((pp[y == c] == c).mean()) for c in range(3)]


def jaccard(p, q, y):
    err_p = predict(p) != y
    err_q = predict(q) != y
    inter = int((err_p & err_q).sum())
    union = int((err_p | err_q).sum())
    return inter / max(union, 1)


def main():
    y = load_y()
    subs = reconstruct_lb_validated_set(y)
    primary_o, primary_t = subs["primary_lb098094"]
    recipe_o, recipe_t = subs["recipe_lb097939"]
    cb_raw_o, cb_raw_t = subs["catboost_lb097935"]
    cb_o, cb_t = iso_cal(cb_raw_o, cb_raw_t, y)
    m2_o, m2_t = subs["m2_pseudo_lb097998"]
    s2_o, s2_t = subs["stack2_lb098008"]
    m3_o, m3_t = subs["m3_seed_lb098005"]

    primary_bal = bal(primary_o, y)
    primary_errs = errs(primary_o, y)
    primary_pcr = pcr(primary_o, y)
    log(f"PRIMARY: OOF {primary_bal:.6f}, errs {primary_errs}, PCR {primary_pcr}")
    log(f"recipe Jaccard vs primary: {jaccard(recipe_o, primary_o, y):.4f}")
    log(f"catboost_iso Jaccard vs primary: {jaccard(cb_o, primary_o, y):.4f}")
    log(f"m2_pseudo Jaccard vs primary: {jaccard(m2_o, primary_o, y):.4f}")
    log(f"stack2 Jaccard vs primary: {jaccard(s2_o, primary_o, y):.4f}")
    log(f"m3_seed Jaccard vs primary: {jaccard(m3_o, primary_o, y):.4f}")

    out = {"primary": {"oof": primary_bal, "errs": primary_errs, "pcr": primary_pcr},
           "experiments": []}

    # =====================================================================
    log("\n=== EXPERIMENT 1: 3-view {primary, recipe, catboost_iso} log-blend grid ===")
    # Restrict to primary getting the most weight; 2D grid over (w_recipe, w_cb)
    # subject to sum=1 → w_primary = 1 - w_recipe - w_cb
    print(f"  {'w_pri':>6} {'w_rec':>6} {'w_cb':>6}  {'OOF':>9}  {'Δ':>9}  {'errs':>6}  PCR")
    best_e1 = (-1, None, None)
    for w_rec in np.arange(0.0, 0.55, 0.05):
        for w_cb in np.arange(0.0, 0.55, 0.05):
            w_pri = 1 - w_rec - w_cb
            if w_pri < 0.30 or w_pri > 1.0:
                continue
            w = np.array([w_pri, w_rec, w_cb])
            blend = log_blend([primary_o, recipe_o, cb_o], w)
            b = bal(blend, y)
            if b > best_e1[0]:
                best_e1 = (b, (float(w_pri), float(w_rec), float(w_cb)), blend)
    b, weights, _ = best_e1
    blend = best_e1[2]
    print(f"  best (w_pri, w_rec, w_cb) = ({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}) "
          f"OOF {b:.6f} Δ={b - primary_bal:+.5f} errs={errs(blend, y)}")
    print(f"  PCR={[round(x, 4) for x in pcr(blend, y)]}")
    out["experiments"].append({"name": "3view_primary_recipe_cbiso",
                               "best_weights": weights, "oof": b,
                               "delta_vs_primary": b - primary_bal,
                               "errs": errs(blend, y), "pcr": pcr(blend, y)})
    # Save best
    test_blend = log_blend([primary_t, recipe_t, cb_t], np.array(weights))
    np.save(ART / "oof_own_3view.npy", blend.astype(np.float32))
    np.save(ART / "test_own_3view.npy", test_blend.astype(np.float32))

    # =====================================================================
    log("\n=== EXPERIMENT 2: greedy forward from primary, FINE α grid ===")
    candidates = [
        ("recipe", recipe_o, recipe_t),
        ("catboost_iso", cb_o, cb_t),
        ("m2_pseudo", m2_o, m2_t),
        ("m3_seed", m3_o, m3_t),
        ("stack2", s2_o, s2_t),
    ]
    base_o = primary_o.copy()
    base_t = primary_t.copy()
    base_bal = primary_bal
    history = []
    fine_alphas = np.concatenate([np.arange(0.005, 0.05, 0.005),
                                   np.arange(0.05, 0.55, 0.025)])
    remaining = list(range(len(candidates)))
    while remaining:
        best = None
        for i in remaining:
            for a in fine_alphas:
                blend_o = log_blend([base_o, candidates[i][1]], np.array([1 - a, a]))
                b = bal(blend_o, y)
                if best is None or b > best[0]:
                    best = (b, i, float(a))
        if best is None or best[0] <= base_bal + 1e-6:
            break
        b, i, a = best
        base_o = log_blend([base_o, candidates[i][1]], np.array([1 - a, a]))
        base_t = log_blend([base_t, candidates[i][2]], np.array([1 - a, a]))
        base_bal = b
        history.append({"step": len(history) + 1, "name": candidates[i][0],
                        "alpha": a, "oof": b, "delta": b - primary_bal})
        log(f"  step {len(history)}: + {candidates[i][0]} α={a:.3f} → "
            f"OOF {b:.6f} Δ={b - primary_bal:+.5f}")
        remaining.remove(i)
    log(f"  Final greedy OOF: {base_bal:.6f} (Δ {base_bal - primary_bal:+.5f}, errs {errs(base_o, y)})")
    log(f"  PCR: {pcr(base_o, y)}")
    out["experiments"].append({"name": "greedy_fine",
                               "history": history,
                               "final_oof": base_bal,
                               "delta_vs_primary": base_bal - primary_bal,
                               "errs": errs(base_o, y), "pcr": pcr(base_o, y)})
    np.save(ART / "oof_own_greedy_fine.npy", base_o.astype(np.float32))
    np.save(ART / "test_own_greedy_fine.npy", base_t.astype(np.float32))

    # =====================================================================
    log("\n=== EXPERIMENT 3: Hard-vote across {primary, recipe, catboost_iso} ===")
    # Per-row majority of 3 argmaxes; tie-break to primary
    p_pri = predict(primary_o)
    p_rec = predict(recipe_o)
    p_cb = predict(cb_o)
    pri_pred = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        votes = [p_pri[i], p_rec[i], p_cb[i]]
        cnt = np.bincount(votes, minlength=3)
        if cnt.max() >= 2:
            pri_pred[i] = cnt.argmax()
        else:
            pri_pred[i] = p_pri[i]   # tie → primary
    one_hot = np.zeros((len(y), 3), dtype=np.float32)
    one_hot[np.arange(len(y)), pri_pred] = 1.0
    b = bal(one_hot, y)
    e = errs(one_hot, y)
    j = jaccard(one_hot, primary_o, y)
    c = pcr(one_hot, y)
    log(f"  hard-vote 3-view: OOF {b:.6f} Δ={b - primary_bal:+.5f} errs={e} Jacc={j:.4f} PCR={c}")
    out["experiments"].append({"name": "3view_hard_vote",
                               "oof": b, "delta_vs_primary": b - primary_bal,
                               "errs": e, "jaccard_vs_primary": j, "pcr": c})

    (ART / "own_ensemble_subset_probe_results.json").write_text(json.dumps(out, indent=2))
    log("\nSaved subset-probe results JSON")


if __name__ == "__main__":
    main()
