"""v3: safe greedy with stage-2 EXCLUDED (in addition to soft_distill
and xgb_spec_678). The c0_v2 4-way probe just returned LB 0.97961
(gap 0.00089) — stage-2's OOF overfit poisoned the 4-way.

Mirrors c0_safe_greedy_v2.py but with stage-2 in EXCLUDE.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
SUB = Path("submissions")
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
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + RECIPE_BIAS).argmax(1))


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32); tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c]); tt[:, c] = ir.predict(test[:, c])
    oo = oo / np.clip(oo.sum(1, keepdims=True), 1e-9, None)
    tt = tt / np.clip(tt.sum(1, keepdims=True), 1e-9, None)
    return oo, tt


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()

    candidates = [n for n in CANDIDATES if n not in EXCLUDE]
    log(f"v3 excluding {sorted(EXCLUDE)} → {len(candidates)} candidates")
    pool = {}
    for name in candidates:
        oof_p = ART / f"oof_{name}.npy"; test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            continue
        oof = np.load(oof_p).astype(np.float32) / np.clip(np.load(oof_p).astype(np.float32).sum(1, keepdims=True), 1e-9, None)
        test = np.load(test_p).astype(np.float32) / np.clip(np.load(test_p).astype(np.float32).sum(1, keepdims=True), 1e-9, None)
        oof_i, test_i = iso_cal(oof, test, y)
        pool[name] = (oof, test); pool[f"{name}__iso"] = (oof_i, test_i)
    log(f"  {len(pool)//2} components loaded")

    summary = dict(anchors={})
    alphas = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    best_for_sub = None

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
        test_cur = log_blend([pool[n][1] for n in names], list(weights))
        picked = set(names)
        bal_cur = bal_bias(oof_cur, y)
        log(f"start: bal={bal_cur:.5f}")
        chosen = []
        for step in range(1, 8):
            best = None
            for key, (oof_k, test_k) in pool.items():
                base = key.replace("__iso", "")
                if base in picked: continue
                for a in alphas:
                    ot = log_blend([oof_cur, oof_k], [1 - a, a])
                    s = bal_bias(ot, y)
                    if best is None or s > best[0]:
                        best = (s, key, base, a, ot, test_k)
            s, key, base, a, ot, tt = best
            d = s - bal_cur
            log(f"  step{step}: + {key:50s} α={a:.3f}  OOF={s:.5f}  Δ={d:+.5f}")
            if d < 1e-4: log("  stop"); break
            chosen.append((key, float(a)))
            picked.add(base); oof_cur = ot
            test_cur = log_blend([test_cur, tt], [1 - a, a])
            bal_cur = s
        log(f"final[{anchor_name}]: {bal_cur:.5f}  Δ vs LB-best-3way 0.98029 = {bal_cur - LB_BEST_3WAY_OOF:+.5f}")
        summary["anchors"][anchor_name] = dict(
            final_oof=float(bal_cur),
            delta_vs_3way=float(bal_cur - LB_BEST_3WAY_OOF),
            chosen=chosen,
        )
        np.save(ART / f"oof_c0_v3_{anchor_name}.npy", oof_cur.astype(np.float32))
        np.save(ART / f"test_c0_v3_{anchor_name}.npy", test_cur.astype(np.float32))
        if bal_cur > LB_BEST_3WAY_OOF + 1e-4 and (best_for_sub is None or bal_cur > best_for_sub[0]):
            best_for_sub = (bal_cur, anchor_name, test_cur)

    summary["elapsed_sec"] = float(time.time() - t0)
    (ART / "c0_safe_greedy_v3_results.json").write_text(json.dumps(summary, indent=2))

    if best_for_sub is not None:
        bal, an, tt = best_for_sub
        pred = (np.log(np.clip(tt, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        classes = ["Low", "Medium", "High"]
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = pd.DataFrame({
            "id": sample["id"].values,
            "Irrigation_Need": [classes[i] for i in pred],
        })
        sub.to_csv(SUB / f"submission_c0_v3_{an}.csv", index=False)
        log(f"Wrote submission_c0_v3_{an}.csv  OOF={bal:.5f}")


if __name__ == "__main__":
    main()
