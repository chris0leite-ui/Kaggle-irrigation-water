"""P3 A/B: perturbed meta on the ORIGINAL 62-component bank used by the
LB-best primary. Isolates "perturbation lift" from "bank-extension OOF
overfit". Same XGB HPs, same K=3 bag, same noise σ as v1; only the
component pool is restricted to the 62 explicit names from
tier1b_xgb_metastack_results.json.

If perturbed-62 reaches OOF ≥ ~0.98100 standalone with similar blend
peak (Δ ≥ +0.00040), the perturbation mechanism is confirmed real and
the 111-component bank's extra components ADDED noise rather than
signal. If perturbed-62 lands close to original meta (0.98041
standalone), then the +0.00071 lift in the 111-bank version was
bank-extension overfit.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_xgb_metastack import build_lbbest_stack  # noqa: E402
from tier1b_helpers import ART, BIAS, CLASSES, DATA, SUB, TARGET, iso_cal, log, normed  # noqa: E402

SEED = 42
N_FOLDS = 5
ENGINEERED_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                   "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                   "min_boundary_dist", "min_axis_abs",
                   "score_dist_low_mid", "score_dist_mid_high"]

# Exact 62-component list from tier1b_xgb_metastack_results.json (LB-best primary's bank)
ORIG_62 = [
    "bagged_greedy_nonrule", "c0_greedy", "catboost_optuna", "catboost_recipe_gpu",
    "em_uniform", "extratrees_dist_digits", "extratrees_dist_digits_v2", "greedy_blend",
    "greedy_full_bank_6way", "hybrid_lgbmxgb_blend", "lb_best_fs123", "lb_best_fs7",
    "lgbm_competitor", "lgbm_dist_digits", "lgbm_dist_digits_ote", "lgbm_te_orig",
    "ovo_boundary_blend", "ovo_nonrule_blend", "p3_embed_propagate", "realmlp",
    "recipe_171pair", "recipe_allpairs", "recipe_catboost", "recipe_focal_g2h3",
    "recipe_full_te", "recipe_full_te_a01", "recipe_full_te_a10",
    "recipe_full_te_catboost", "recipe_full_te_cldrop", "recipe_full_te_dae",
    "recipe_full_te_fexboth", "recipe_full_te_gby", "recipe_full_te_lgbm",
    "recipe_full_te_seed123", "recipe_full_te_seed7", "recipe_lgbm",
    "recipe_no_combos", "recipe_no_digits", "recipe_no_orig", "recipe_no_ote",
    "recipe_pseudolabel", "recipe_pseudolabel_seed123labeler",
    "recipe_pseudolabel_seed7labeler", "recipe_pseudolabel_tau092",
    "tabpfn", "tta_recipe_baseline", "tta_recipe_s001", "tta_recipe_s005",
    "tta_recipe_s010", "tta_recipe_s020", "tta_recipe_s030",
    "xgb_corn", "xgb_dist_digits", "xgb_dist_digits_ote", "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_light", "xgb_dist_digits_ote_digits_pairs",
    "xgb_dist_digits_ote_light", "xgb_dist_routed_v3", "xgb_nonrule",
    "xgb_spec_36", "xgb_vanilla_dist",
]


def load_components(names):
    pool = {}
    for n in names:
        op = ART / f"oof_{n}.npy"
        tp = ART / f"test_{n}.npy"
        if not op.exists() or not tp.exists():
            log(f"  WARN missing: {n}")
            continue
        pool[n] = (normed(np.load(op).astype(np.float32)),
                   normed(np.load(tp).astype(np.float32)))
    return pool


def build_meta_features(train, test, lb_oof, lb_test, pool, names):
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[ENGINEERED_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[ENGINEERED_COLS].to_numpy(dtype=np.float32)
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in names if n in pool]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in names if n in pool]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    return X_tr, X_te


def train_perturbed_meta(X_tr, X_te, y, sigma, colsample, bag_k):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=colsample,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    n_eng_offset = 3 + len(ENGINEERED_COLS)  # cols 0..n_eng_offset are LB+engineered
    oof_meta = np.zeros((len(X_tr), 3), dtype=np.float32)
    test_meta_folds = []
    for bag in range(bag_k):
        rng = np.random.default_rng(SEED + 1000 + bag)
        oof_bag = np.zeros_like(oof_meta)
        test_bag = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
            t0 = time.time()
            noise = rng.standard_normal(X_tr.shape).astype(np.float32) * sigma
            noise[:, 3:n_eng_offset] = 0  # zero noise on engineered cols
            X_tr_n = X_tr + noise
            dtr = xgb.DMatrix(X_tr_n[tr_idx], label=y[tr_idx])
            dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])
            dte = xgb.DMatrix(X_te)
            booster = xgb.train(xgb_params, dtr, num_boost_round=3000,
                                evals=[(dva, "val")], early_stopping_rounds=200,
                                verbose_eval=0)
            bi = booster.best_iteration
            vp = booster.predict(dva, iteration_range=(0, bi + 1))
            oof_bag[va_idx] = vp
            test_bag.append(booster.predict(dte, iteration_range=(0, bi + 1)))
            log(f"  bag {bag} fold {fold + 1}/5 it={bi} "
                f"argmax={balanced_accuracy_score(y[va_idx], vp.argmax(1)):.5f} "
                f"wall={time.time() - t0:.0f}s")
        oof_meta += oof_bag / bag_k
        test_meta_folds.append(np.mean(test_bag, axis=0))
    return normed(oof_meta), normed(np.mean(test_meta_folds, axis=0).astype(np.float32))


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(np.int32)

    log("building LB-best 3-stack")
    lb_oof, lb_test = build_lbbest_stack(y)

    log(f"loading {len(ORIG_62)} original components")
    pool = load_components(ORIG_62)
    log(f"  found {len(pool)} of {len(ORIG_62)}")
    if len(pool) != 62:
        log(f"  WARN: missing {set(ORIG_62) - set(pool)}")

    X_tr, X_te = build_meta_features(train, test, lb_oof, lb_test, pool, ORIG_62)
    log(f"  shape={X_tr.shape}")

    log(f"\n=== perturbed-62 (σ=0.3, colsample=0.9, K=3) ===")
    oof, te = train_perturbed_meta(X_tr, X_te, y, sigma=0.3, colsample=0.9, bag_k=3)
    np.save(ART / "oof_meta_perturbed_62_v1.npy", oof)
    np.save(ART / "test_meta_perturbed_62_v1.npy", te)

    iso_o, iso_t = iso_cal(oof, te, y)
    lb_bal = bal(lb_oof, y)
    rows = []
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for use_iso in (False, True):
        m_o = iso_o if use_iso else oof
        m_t = iso_t if use_iso else te
        for a in alphas:
            blend = log_blend([lb_oof, m_o], np.array([1 - a, a]))
            b = bal(blend, y)
            rows.append(dict(iso=use_iso, alpha=a, oof=float(b),
                             delta=float(b - lb_bal)))
    best = max(rows, key=lambda r: r["delta"])

    log(f"\nstandalone @ bias = {bal(oof, y):.5f}  argmax={balanced_accuracy_score(y, oof.argmax(1)):.5f}")
    log(f"iso @ bias        = {bal(iso_o, y):.5f}")
    log(f"\nblend onto LB-3-stack peak: iso={best['iso']} α={best['alpha']:.3f} "
        f"OOF={best['oof']:.5f} Δ={best['delta']:+.5f}")

    out = dict(
        n_components=len(pool), feature_dim=X_tr.shape[1],
        standalone_at_bias=float(bal(oof, y)),
        standalone_argmax=float(balanced_accuracy_score(y, oof.argmax(1))),
        iso_at_bias=float(bal(iso_o, y)),
        sweep=rows, best=best,
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "p3_perturbed_62_results.json").write_text(json.dumps(out, indent=2))

    if best["delta"] >= 4e-4:
        a = best["alpha"]
        m_t = iso_t if best["iso"] else te
        blend_test = log_blend([lb_test, m_t], np.array([1 - a, a]))
        pred = (np.log(np.clip(blend_test, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / f"submission_p3_perturbed_62_a{int(a * 1000):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"  wrote {path}")
    log(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
