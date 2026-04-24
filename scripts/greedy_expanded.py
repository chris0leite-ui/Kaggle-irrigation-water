"""Greedy forward-selection with FULL candidate pool post-today's-experiments.

Extends greedy_realmlp_refit.py by:
  * Adding today's new 3-class candidates: recipe_focal (g2h3),
    recipe_focal_invfreq (α=invfreq), soft_distill_small (d=3/r=1500).
  * EXCLUDE_GREEDY_ADD for LB-verified regressors — they can't be
    greedy-added, but are still valid as anchor ingredients:
      soft_distill, soft_distill_small, recipe_pseudolabel_stage2,
      xgb_spec_678.
  * Running from THREE anchors (recipe_full_te, LB-best 3-way,
    LB-best 3-way + realmlp + xgb_nonrule_iso — the new LB 0.98008
    stack) to see if different starting points surface new paths.

No retraining. ~10 min on CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
DATA = Path("data")
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
LB_BEST_OOF = 0.98061  # new best from RealMLP 3-stack blend

# Components valid as anchor ingredients but NOT greedy-add candidates
# (confirmed LB regressors — their OOF lift is overfit artifact).
EXCLUDE_GREEDY_ADD = {
    "soft_distill",                    # LB 0.97850, gap +0.00246
    "soft_distill_small",              # LB 0.97865, gap +0.00201
    "recipe_pseudolabel_stage2",       # null per 2026-04-23
    "xgb_spec_678",                    # sparse carrier, not blend
    "recipe_pseudolabel_seed7labeler", # LB 0.97969 (A/B)
    "recipe_pseudolabel_seed123labeler",
}

# Components to never load at all (broken shape or irrelevant).
EXCLUDE_POOL = {"xgb_bin_medium", "spec6_mh", "spec6_mh_v2"}

CANDIDATES = [
    # Recipe family
    "recipe_full_te", "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler", "recipe_full_te_seed7",
    "recipe_allpairs", "recipe_catboost", "recipe_lgbm", "recipe_171pair",
    "recipe_full_te_a01", "recipe_full_te_a10", "recipe_full_te_catboost",
    "recipe_full_te_lgbm", "recipe_full_te_cldrop",
    "recipe_no_ote", "recipe_no_digits", "recipe_no_combos", "recipe_no_orig",
    # Today's focal variants (both null standalone but have low Jaccards).
    "recipe_focal_g2h3", "recipe_focal_g2_invfreq",
    # Today's distill variant (LB regressor — kept in pool but
    # EXCLUDE_GREEDY_ADD blocks greedy from picking it).
    "soft_distill_small", "soft_distill",
    # Other / non-recipe
    "em_uniform", "xgb_corn", "xgb_nonrule",
    "xgb_dist_digits", "lgbm_dist_digits",
    "xgb_dist_digits_ote", "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_pairs", "xgb_dist_digits_ote_digits_light",
    "xgb_dist_digits_ote_light", "lgbm_dist_digits_ote",
    "xgb_dist_routed_v3", "xgb_vanilla_dist",
    "catboost_optuna", "catboost_recipe_gpu",
    "extratrees_dist_digits", "extratrees_dist_digits_v2",
    "lgbm_competitor", "lgbm_te_orig", "tabpfn",
    "realmlp",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend(probs, weights):
    eps = 1e-12
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
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

    candidates = [n for n in CANDIDATES if n not in EXCLUDE_POOL]
    log(f"greedy-expanded: {len(candidates)} components to try")
    log(f"EXCLUDE_GREEDY_ADD = {sorted(EXCLUDE_GREEDY_ADD)}")

    pool = {}
    for name in candidates:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            continue
        try:
            oof_raw = np.load(oof_p).astype(np.float32)
            test_raw = np.load(test_p).astype(np.float32)
            if oof_raw.ndim != 2 or oof_raw.shape[1] != 3:
                log(f"  skip {name}: shape {oof_raw.shape} not (N,3)")
                continue
            oof = oof_raw / np.clip(oof_raw.sum(1, keepdims=True), 1e-9, None)
            test = test_raw / np.clip(test_raw.sum(1, keepdims=True), 1e-9, None)
            oof_i, test_i = iso_cal(oof, test, y)
            pool[name] = (oof, test)
            pool[f"{name}__iso"] = (oof_i, test_i)
        except Exception as e:
            log(f"  skip {name}: {e}")
    log(f"pool: {len(pool)//2} components loaded as (raw, iso)")

    alphas = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    anchors = [
        ("recipe_full_te", [("recipe_full_te", 1.0)]),
        ("lb_best_3way",
         [("recipe_full_te", 0.25),
          ("recipe_pseudolabel", 0.35),
          ("recipe_pseudolabel_seed7labeler", 0.40)]),
        # LB-best stack (currently LB 0.98008) — explicit test of whether
        # further greedy additions lift beyond the known winner.
        ("lb_best_realmlp_stack",
         [("recipe_full_te", 0.25 * 0.725),
          ("recipe_pseudolabel", 0.35 * 0.725),
          ("recipe_pseudolabel_seed7labeler", 0.40 * 0.725),
          ("realmlp", 0.200),
          ("xgb_nonrule", 0.075)]),  # note: __iso version used in LB sub
    ]

    summary = dict(anchors={}, elapsed_sec=None,
                   lb_best_oof=LB_BEST_OOF,
                   pool_size=len(pool) // 2)

    for anchor_name, anchor_def in anchors:
        log("=" * 70)
        log(f"Anchor: {anchor_name}")
        names, weights = zip(*anchor_def)
        # Any missing anchor component = skip.
        missing = [n for n in names if n not in pool]
        if missing:
            log(f"  skip anchor (missing: {missing})")
            continue
        oof_cur = log_blend([pool[n][0] for n in names], list(weights))
        test_cur = log_blend([pool[n][1] for n in names], list(weights))
        picked_bases = set(names)
        bal_cur = bal_bias(oof_cur, y)
        log(f"  start: bal={bal_cur:.5f}  Δvs_LB_best={bal_cur - LB_BEST_OOF:+.5f}")
        chosen = []
        oof_chain = [oof_cur]
        test_chain = [test_cur]
        weight_chain = [1.0]

        for step in range(1, 8):
            best = None
            for key, (oof_k, test_k) in pool.items():
                base = key.replace("__iso", "")
                if base in picked_bases:
                    continue
                if base in EXCLUDE_GREEDY_ADD:
                    continue
                for a in alphas:
                    ot = log_blend([oof_cur, oof_k], [1 - a, a])
                    s = bal_bias(ot, y)
                    if best is None or s > best[0]:
                        best = (s, key, base, a, ot, test_k)
            s, key, base, a, ot, tt = best
            d = s - bal_cur
            log(f"  step{step}: + {key:50s}  α={a:.3f}  "
                f"OOF={s:.5f}  Δ={d:+.5f}")
            if d < 1e-4:
                log("  stop (Δ < 1e-4)")
                break
            chosen.append((key, float(a), float(s), float(d)))
            picked_bases.add(base)
            # Update running state.
            new_test = log_blend([test_cur, tt], [1 - a, a])
            oof_cur = ot
            test_cur = new_test
            bal_cur = s

        log(f"final[{anchor_name}]: {bal_cur:.5f}  "
            f"Δvs_LB_best={bal_cur - LB_BEST_OOF:+.5f}")

        summary["anchors"][anchor_name] = dict(
            start_oof=float(chosen[0][2] - chosen[0][3]) if chosen else float(bal_cur),
            final_oof=float(bal_cur),
            delta_vs_lb_best=float(bal_cur - LB_BEST_OOF),
            chosen=chosen,
        )

        # Emit submission IF blend beats LB_BEST_OOF by ≥ 1e-4 on OOF.
        if bal_cur - LB_BEST_OOF >= 1e-4:
            eps = 1e-9
            test_log = np.log(np.clip(test_cur, eps, 1.0))
            pred_idx = (test_log + RECIPE_BIAS).argmax(1)
            IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}
            test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()
            sub = pd.DataFrame({
                "id": test_ids,
                "Irrigation_Need": [IDX2CLS[i] for i in pred_idx],
            })
            out = Path("submissions") / f"submission_greedy_expanded_{anchor_name}.csv"
            sub.to_csv(out, index=False)
            log(f"  emitted {out}  dist={dict(sub['Irrigation_Need'].value_counts())}")

    summary["elapsed_sec"] = float(time.time() - t0)
    out_path = ART / "greedy_expanded_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
