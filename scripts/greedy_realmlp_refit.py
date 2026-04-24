"""Greedy refit with RealMLP in the OOF pool.

Diagnostic: run greedy forward-selection from two anchors (recipe and
LB-best 3-way) with the RealMLP OOF added to the candidate pool. Tests
whether the hand-picked α=0.375 in scripts/blend_realmlp.py missed a
better blend configuration, and whether RealMLP's unique Jaccard-0.62
errors pair productively with components other than recipe/pseudo.

No retraining, no LB spend. Reuses c0_safe_greedy_v3 pattern with:
  - EXCLUDE = {soft_distill, xgb_spec_678, pseudo_stage2}
    (all confirmed LB regressors in prior experiments)
  - + realmlp added to CANDIDATES
  - fixed recipe bias throughout
  - both raw and isotonic-calibrated variants in pool
  - greedy selects step if Δ ≥ +1e-4 OOF

Emits no submission; just prints the greedy path + best OOF.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
DATA = Path("data")
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
LB_BEST_3WAY_OOF = 0.98029
EXCLUDE = {"soft_distill", "xgb_spec_678", "recipe_pseudolabel_stage2"}

CANDIDATES = [
    "recipe_full_te", "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler", "recipe_full_te_seed7",
    "recipe_allpairs", "recipe_catboost", "recipe_lgbm", "recipe_171pair",
    "recipe_full_te_a01", "recipe_full_te_a10", "recipe_full_te_catboost",
    "recipe_full_te_lgbm", "recipe_full_te_cldrop",
    "recipe_no_ote", "recipe_no_digits", "recipe_no_combos", "recipe_no_orig",
    "em_uniform", "xgb_corn", "xgb_nonrule",
    "xgb_dist_digits", "lgbm_dist_digits",
    "xgb_dist_digits_ote", "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_pairs", "xgb_dist_digits_ote_digits_light",
    "xgb_dist_digits_ote_light", "lgbm_dist_digits_ote",
    "xgb_dist_routed_v3", "xgb_vanilla_dist",
    "catboost_optuna", "catboost_recipe_gpu",
    "extratrees_dist_digits", "extratrees_dist_digits_v2",
    "lgbm_competitor", "lgbm_te_orig", "tabpfn",
    "realmlp",  # <- NEW — first NN with Jaccard 0.62 vs LB-best 3-way
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend(probs, weights):
    eps = 1e-12
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    lp = np.zeros_like(probs[0], dtype=np.float64)
    for wi, p in zip(w, probs):
        lp += wi * np.log(np.clip(p, eps, 1))
    lp -= lp.max(axis=1, keepdims=True)
    ez = np.exp(lp)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)


def bal_bias(p, y):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + RECIPE_BIAS).argmax(1),
    )


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    oo = oo / np.clip(oo.sum(1, keepdims=True), 1e-9, None)
    tt = tt / np.clip(tt.sum(1, keepdims=True), 1e-9, None)
    return oo, tt


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}).to_numpy()

    candidates = [n for n in CANDIDATES if n not in EXCLUDE]
    log(f"greedy-refit with RealMLP → {len(candidates)} candidates "
        f"(EXCLUDE = {sorted(EXCLUDE)})")

    pool = {}
    for name in candidates:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            continue
        oof_raw = np.load(oof_p).astype(np.float32)
        test_raw = np.load(test_p).astype(np.float32)
        oof = oof_raw / np.clip(oof_raw.sum(1, keepdims=True), 1e-9, None)
        test = test_raw / np.clip(test_raw.sum(1, keepdims=True), 1e-9, None)
        oof_i, test_i = iso_cal(oof, test, y)
        pool[name] = (oof, test)
        pool[f"{name}__iso"] = (oof_i, test_i)
    log(f"  {len(pool)//2} components loaded "
        f"({'realmlp' in {k.replace('__iso', '') for k in pool}})")

    realmlp_included = "realmlp" in pool
    log(f"  realmlp present: {realmlp_included}")

    summary = dict(anchors={}, realmlp_included=realmlp_included)
    alphas = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    anchors = [
        ("recipe_full_te", [("recipe_full_te", 1.0)]),
        ("lb_best_3way",
         [("recipe_full_te", 0.25),
          ("recipe_pseudolabel", 0.35),
          ("recipe_pseudolabel_seed7labeler", 0.40)]),
    ]

    for anchor_name, anchor_def in anchors:
        log("=" * 70)
        log(f"Anchor: {anchor_name}")
        names, weights = zip(*anchor_def)
        oof_cur = log_blend([pool[n][0] for n in names], list(weights))
        picked = set(names)
        bal_cur = bal_bias(oof_cur, y)
        log(f"  start: bal={bal_cur:.5f}")
        chosen = []
        realmlp_picked_at_step = None
        for step in range(1, 10):
            best = None
            for key, (oof_k, _) in pool.items():
                base = key.replace("__iso", "")
                if base in picked:
                    continue
                for a in alphas:
                    ot = log_blend([oof_cur, oof_k], [1 - a, a])
                    s = bal_bias(ot, y)
                    if best is None or s > best[0]:
                        best = (s, key, base, a, ot)
            s, key, base, a, ot = best
            d = s - bal_cur
            flag = "  <-- REALMLP" if "realmlp" in key else ""
            log(f"  step{step}: + {key:50s} α={a:.3f}  "
                f"OOF={s:.5f}  Δ={d:+.5f}{flag}")
            if d < 1e-4:
                log("  stop (Δ < 1e-4)")
                break
            chosen.append((key, float(a), float(s), float(d)))
            picked.add(base)
            oof_cur = ot
            bal_cur = s
            if "realmlp" in key and realmlp_picked_at_step is None:
                realmlp_picked_at_step = step

        log(f"final[{anchor_name}]: {bal_cur:.5f}  "
            f"Δ vs LB3 0.98029 = {bal_cur - LB_BEST_3WAY_OOF:+.5f}  "
            f"realmlp_picked_at_step={realmlp_picked_at_step}")
        summary["anchors"][anchor_name] = dict(
            final_oof=float(bal_cur),
            delta_vs_3way=float(bal_cur - LB_BEST_3WAY_OOF),
            chosen=chosen,
            realmlp_picked_at_step=realmlp_picked_at_step,
        )

    summary["elapsed_sec"] = float(time.time() - t0)
    out_path = ART / "greedy_realmlp_refit_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
