"""P3: perturbed-OOF meta-stacker. Amplify the negative OOF→LB gap.

Hypothesis: the LB-best 4-stack's OOF→LB gap of -0.00010 (LB > OOF) is
CV-pessimism — meta sees noisy fold hold-outs of 62 components; on test,
all components fire on unseen rows simultaneously, no fold-noise smearing.
Train meta on PERTURBED log-prob inputs (Gaussian noise) so it learns
robust signal channels, not fold-noise. Predict on UNNOISED test.

Pipeline mirrors tier1b_xgb_metastack EXACTLY (same EXCLUDE, same XGB HPs,
same 5-fold StratifiedKFold(seed=42), same iso-cal recipe-bias decision
rule). The only difference: noise added to log-prob columns at training
time, plus K=3 bag for stability and stronger column subsampling
(colsample_bytree=0.3) to compound component-dropout regularization.

Decision: emit submission only if iso-cal'd blend onto LB-best 3-stack
gives Δ OOF ≥ +0.00023 (the LB-best primary's iso-cal'd Δ at α=0.30).
Anything BELOW that just reproduces or weakens primary; anything above
might amplify the negative OOF→LB gap further.
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
from tier1b_xgb_metastack import EXCLUDE, build_lbbest_stack, load_pool  # noqa: E402
from tier1b_helpers import ART, BIAS, CLASSES, DATA, SUB, TARGET, iso_cal, log, normed  # noqa: E402

SEED = 42
N_FOLDS = 5
N_ENGINEERED = 14  # dgp_score, rule_pred, sm/rf/tc/ws_dist/abs, min_*, score_dist_*
ENGINEERED_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                   "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                   "min_boundary_dist", "min_axis_abs",
                   "score_dist_low_mid", "score_dist_mid_high"]


def build_meta_features(train, test, lb_oof, lb_test, pool):
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[ENGINEERED_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[ENGINEERED_COLS].to_numpy(dtype=np.float32)
    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1)
    # noise mask: True where noise should be applied (log-prob cols only)
    noise_mask = np.ones(X_tr.shape[1], dtype=bool)
    noise_mask[3:3 + N_ENGINEERED] = False  # skip engineered
    return X_tr, X_te, noise_mask, component_names


def train_meta_with_noise(X_tr, X_te, y, sigma, colsample, bag_k, max_rounds=3000):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    xgb_params = dict(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.9, colsample_bytree=colsample,
        reg_alpha=5.0, reg_lambda=5.0,
        tree_method="hist", verbosity=0, seed=SEED, nthread=-1,
    )
    n_tr = len(X_tr)
    oof_meta = np.zeros((n_tr, 3), dtype=np.float32)
    test_meta_folds = []

    for bag_seed in range(bag_k):
        rng = np.random.default_rng(SEED + 1000 + bag_seed)
        oof_bag = np.zeros((n_tr, 3), dtype=np.float32)
        test_bag_folds = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y)):
            t1 = time.time()
            # Noise applied per-fold per-bag — different realization each time.
            # noise_mask narrows to log-prob cols (engineered features stay clean).
            noise = rng.standard_normal(X_tr.shape).astype(np.float32) * sigma
            noise[:, 3:3 + N_ENGINEERED] = 0  # zero out engineered cols
            X_tr_noisy = X_tr + noise
            dtr = xgb.DMatrix(X_tr_noisy[tr_idx], label=y[tr_idx])
            dva = xgb.DMatrix(X_tr[va_idx], label=y[va_idx])  # CLEAN val
            dte = xgb.DMatrix(X_te)  # CLEAN test
            booster = xgb.train(
                xgb_params, dtr, num_boost_round=max_rounds,
                evals=[(dva, "val")], early_stopping_rounds=200,
                verbose_eval=0,
            )
            bi = booster.best_iteration
            vp = booster.predict(dva, iteration_range=(0, bi + 1))
            tp = booster.predict(dte, iteration_range=(0, bi + 1))
            oof_bag[va_idx] = vp.astype(np.float32)
            test_bag_folds.append(tp)
            log(f"  bag {bag_seed} fold {fold + 1}/{N_FOLDS} it={bi} "
                f"argmax={balanced_accuracy_score(y[va_idx], vp.argmax(1)):.5f} "
                f"wall={time.time() - t1:.0f}s")
        oof_meta += oof_bag / bag_k
        test_meta_folds.append(np.mean(test_bag_folds, axis=0))
    test_meta = np.mean(test_meta_folds, axis=0).astype(np.float32)
    return normed(oof_meta), normed(test_meta)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def evaluate_blend(meta_oof, meta_test, lb_oof, lb_test, y, tag):
    meta_iso_o, meta_iso_t = iso_cal(meta_oof, meta_test, y)
    lb_bal = bal(lb_oof, y)
    log(f"  meta standalone argmax={balanced_accuracy_score(y, meta_oof.argmax(1)):.5f}  "
        f"@bias={bal(meta_oof, y):.5f}  iso@bias={bal(meta_iso_o, y):.5f}")
    rows = []
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for use_iso in (False, True):
        m_o = meta_iso_o if use_iso else meta_oof
        m_t = meta_iso_t if use_iso else meta_test
        for a in alphas:
            blend = log_blend([lb_oof, m_o], np.array([1 - a, a]))
            b = bal(blend, y)
            rows.append({"iso": use_iso, "alpha": a, "oof": float(b),
                         "delta": float(b - lb_bal)})
    best = max(rows, key=lambda r: r["delta"])
    log(f"  {tag} sweep best: iso={best['iso']} α={best['alpha']:.3f} "
        f"OOF={best['oof']:.5f} Δ={best['delta']:+.5f}")
    return rows, best, meta_iso_o, meta_iso_t


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(np.int32)

    log("building LB-best 3-stack")
    lb_oof, lb_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF = {bal(lb_oof, y):.5f}")

    log("loading 62-component pool")
    pool = load_pool(y)
    log(f"  {len(pool)} components")

    log("constructing meta-features")
    X_tr, X_te, noise_mask, comp_names = build_meta_features(train, test, lb_oof, lb_test, pool)
    log(f"  shape={X_tr.shape}  noise_cols={noise_mask.sum()}/{X_tr.shape[1]}")

    # Two configs: noise-only-mild + noise+drop-strong (bag K=3 for stability).
    # Pick the strongest signal config based on early-stopping economics.
    configs = [
        dict(name="v1_noise03_csb09_k3", sigma=0.3, colsample=0.9, bag_k=3),
        dict(name="v2_noise05_csb05_k3", sigma=0.5, colsample=0.5, bag_k=3),
    ]

    all_results = {}
    for cfg in configs:
        log(f"\n=== {cfg['name']}: σ={cfg['sigma']} colsample={cfg['colsample']} bag={cfg['bag_k']} ===")
        oof, te = train_meta_with_noise(X_tr, X_te, y, cfg["sigma"], cfg["colsample"], cfg["bag_k"])
        rows, best, iso_o, iso_t = evaluate_blend(oof, te, lb_oof, lb_test, y, cfg["name"])
        np.save(ART / f"oof_meta_perturbed_{cfg['name']}.npy", oof)
        np.save(ART / f"test_meta_perturbed_{cfg['name']}.npy", te)
        all_results[cfg["name"]] = dict(cfg=cfg, sweep=rows, best=best,
                                        meta_argmax=float(balanced_accuracy_score(y, oof.argmax(1))),
                                        meta_at_bias=float(bal(oof, y)),
                                        meta_iso_at_bias=float(bal(iso_o, y)))
        # Emit submission if best blend Δ ≥ +0.00023 (the primary's iso α=0.30 lift)
        if best["delta"] >= 0.00023:
            a = best["alpha"]
            m_t = iso_t if best["iso"] else te
            blend_test = log_blend([lb_test, m_t], np.array([1 - a, a]))
            pred = (np.log(np.clip(blend_test, 1e-12, 1)) + BIAS).argmax(1)
            sample = pd.read_csv(DATA / "sample_submission.csv")
            sub = sample.copy()
            sub[TARGET] = [CLASSES[i] for i in pred]
            tag = f"{cfg['name']}_{'iso' if best['iso'] else 'raw'}_a{int(a * 1000):03d}"
            path = SUB / f"submission_p3_perturbed_{tag}.csv"
            sub.to_csv(path, index=False)
            log(f"  wrote {path}")

    out = dict(lb_best_3stack_oof=float(bal(lb_oof, y)),
               primary_oof_target=0.98084,
               primary_iso_alpha030_delta=0.00023,
               configs=all_results, elapsed_sec=float(time.time() - t0))
    (ART / "p3_perturbed_meta_results.json").write_text(json.dumps(out, indent=2))
    log(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
