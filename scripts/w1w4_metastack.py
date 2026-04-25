"""W1+W4 OOFs as meta-stacker INPUTS — retrain meta with expanded pool.

The W1 MLPs (3 configs, Jaccard 0.51-0.56) and W4 score-regression
(Jaccard 0.61) all nulled as direct blend legs (magnitude trap or
Pareto-frontier closure). But their UNPRECEDENTED orthogonality may
still help the XGB meta-stacker, which is robust to weak components
(it down-weights them and may use them sparingly to correct edge cases).

This script reuses tier1b_xgb_metastack pool-loading + EXCLUDE list
but saves outputs as `xgb_metastack_w1w4` so the LB-best reconstruction
that depends on the original `xgb_metastack` stays intact.

Then runs blend gate vs LB-best 4-stack: substitute new meta into the
4-stack's α=0.30 slot and compare.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, CLS2IDX
from tier1b_xgb_metastack import (
    EXCLUDE, _normed, build_lbbest_stack, load_pool, bal, log,
    ART, SUB, DATA, SEED, N_FOLDS, BIAS, TARGET,
)


def iso_cal_oof_test(oof, test, y):
    """Per-class isotonic, fit on full OOF, transform test."""
    oof_o = np.zeros_like(oof); test_o = np.zeros_like(test)
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        ir.fit(oof[:, k], (y == k).astype(float))
        oof_o[:, k] = ir.transform(oof[:, k])
        test_o[:, k] = ir.transform(test[:, k])
    oof_o = oof_o / oof_o.sum(1, keepdims=True).clip(1e-9)
    test_o = test_o / test_o.sum(1, keepdims=True).clip(1e-9)
    return oof_o, test_o


def lb_log(*pw):
    out = sum(w * np.log(p.clip(1e-12)) for p, w in pw)
    out = np.exp(out - out.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    test_ids = test["id"].values
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    # Original LB-best stack (uses OLD xgb_metastack)
    log("building LB-best 3-stack anchor (unchanged)")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    log(f"  LB-best 3-stack OOF = {bal(lb3_oof, y):.5f}")
    # 4-stack = LB-best 3-stack + xgb_metastack (OLD) iso-cal'd at 0.30 weight
    old_meta_oof = np.load(ART / "oof_xgb_metastack.npy")
    old_meta_test = np.load(ART / "test_xgb_metastack.npy")
    old_meta_oof_iso, old_meta_test_iso = iso_cal_oof_test(old_meta_oof, old_meta_test, y)
    final_lb_oof = lb_log((lb3_oof, .70), (old_meta_oof_iso, .30))
    final_lb_test = lb_log((lb3_test, .70), (old_meta_test_iso, .30))
    lb4_ba = bal(final_lb_oof, y)
    lb4_pred = (np.log(final_lb_oof.clip(1e-12)) + BIAS).argmax(1)
    lb4_errs = (lb4_pred != y).sum()
    log(f"  LB-best 4-stack OOF = {lb4_ba:.5f}, errs {lb4_errs}")

    # Load pool (auto-includes new W1 + W4 OOFs)
    log("loading pool (now with W1 tanh_*, W4 xgb_score_reg)")
    pool = load_pool(y)
    log(f"  {len(pool)} 3-class components in pool")
    new_components = [n for n in pool if n.startswith("w1_") or n == "xgb_score_reg"]
    log(f"  new since prior meta: {new_components}")
    if not new_components:
        log("  WARNING: no new components found — pool unchanged")

    # Build meta-features matrix (mirrors tier1b layout)
    log("constructing meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_cols = ["dgp_score", "rule_pred",
                 "sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
    meta_te = te_d[meta_cols].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]
    lb_log_tr = np.log(np.clip(lb3_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb3_test, 1e-9, 1.0))
    Xtr = np.hstack([meta_tr, lb_log_tr] + comp_tr).astype(np.float32)
    Xte = np.hstack([meta_te, lb_log_te] + comp_te).astype(np.float32)
    log(f"  Xtr shape {Xtr.shape}  Xte shape {Xte.shape}  ({len(component_names)} components × 3 + meta + lb_log)")

    # Train meta XGB (same heavy-reg HPs as original)
    log("training XGB meta-stacker (5-fold)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_meta = np.zeros((len(y), 3), dtype=np.float32)
    test_meta = np.zeros((len(test), 3), dtype=np.float32)
    params = dict(
        n_estimators=3000, max_depth=4, max_leaves=30,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5, max_bin=1024,
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED, verbosity=0,
        early_stopping_rounds=200,
    )
    fold_iters = []
    for f, (tr, va) in enumerate(skf.split(Xtr, y), 1):
        tf = time.time()
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr[tr], y[tr], eval_set=[(Xtr[va], y[va])], verbose=500)
        oof_meta[va] = m.predict_proba(Xtr[va])
        test_meta += m.predict_proba(Xte) / N_FOLDS
        fold_iters.append(m.best_iteration)
        ba_f = balanced_accuracy_score(y[va], oof_meta[va].argmax(1))
        log(f"  fold {f}: best_iter={m.best_iteration} argmax_bal={ba_f:.5f} ({time.time()-tf:.1f}s)")

    # Standalone meta diagnostic
    meta_argmax = oof_meta.argmax(1)
    meta_argmax_ba = balanced_accuracy_score(y, meta_argmax)
    meta_iso, meta_iso_test = iso_cal_oof_test(oof_meta, test_meta, y)
    meta_iso_ba_at_lb_bias = bal(meta_iso, y)
    log(f"new meta standalone argmax_bal_acc = {meta_argmax_ba:.5f}")
    log(f"new meta iso @ LB bias = {meta_iso_ba_at_lb_bias:.5f}")

    # Save with NEW name (don't clobber LB-best inputs)
    np.save(ART / "oof_xgb_metastack_w1w4.npy", oof_meta)
    np.save(ART / "test_xgb_metastack_w1w4.npy", test_meta)
    log("saved oof_xgb_metastack_w1w4.npy + test_*")

    # Substitute new meta into 4-stack at α=0.30 (same weight as old meta)
    new_4stack_oof = lb_log((lb3_oof, .70), (meta_iso, .30))
    new_4stack_test = lb_log((lb3_test, .70), (meta_iso_test, .30))
    new_4stack_ba = bal(new_4stack_oof, y)
    new_4stack_pred = (np.log(new_4stack_oof.clip(1e-12)) + BIAS).argmax(1)
    new_4stack_errs = (new_4stack_pred != y).sum()
    delta_4stack = new_4stack_ba - lb4_ba
    log(f"\n=== 4-stack substitution (α=0.30) ===")
    log(f"  old LB-best 4-stack: tuned {lb4_ba:.5f}  errs {lb4_errs}")
    log(f"  new w1w4   4-stack: tuned {new_4stack_ba:.5f}  errs {new_4stack_errs}  Δ={delta_4stack:+.5f}")

    # Per-class recall comparison
    rec_old = [((y == k) & (lb4_pred == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
    rec_new = [((y == k) & (new_4stack_pred == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
    print(f"  per-class recall (old):  L {rec_old[0]:.4f}  M {rec_old[1]:.4f}  H {rec_old[2]:.4f}")
    print(f"  per-class recall (new):  L {rec_new[0]:.4f}  M {rec_new[1]:.4f}  H {rec_new[2]:.4f}")
    rec_drops = [rec_new[k] - rec_old[k] for k in (0,1,2)]
    print(f"  recall delta:            L {rec_drops[0]:+.4f}  M {rec_drops[1]:+.4f}  H {rec_drops[2]:+.4f}")

    # α-sweep: try other weights for new meta
    log(f"\n=== α-sweep: lb3 + new_meta_iso ===")
    print(f"{'α':>6} {'tuned':>9} {'Δ vs lb4':>+10} {'errs':>6}")
    best_alpha = 0.0; best_delta = 0.0
    for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        b = lb_log((lb3_oof, 1-alpha), (meta_iso, alpha))
        ba = bal(b, y)
        d = ba - lb4_ba
        bp = (np.log(b.clip(1e-12)) + BIAS).argmax(1)
        errs = (bp != y).sum()
        flag = "  *** PASS ***" if d >= 0.0002 else ""
        print(f"{alpha:>6.2f} {ba:>9.5f} {d:>+10.5f} {errs:>6}{flag}")
        if d > best_delta:
            best_delta = d; best_alpha = alpha

    log(f"\nBest α={best_alpha:.2f}, Δ={best_delta:+.5f}")

    # Emit submission if PASS
    if best_delta >= 0.0002 and all(d >= -5e-4 for d in rec_drops):
        b_test = lb_log((lb3_test, 1 - best_alpha), (meta_iso_test, best_alpha))
        bp = (np.log(b_test.clip(1e-12)) + BIAS).argmax(1)
        labels = np.array(["Low", "Medium", "High"])[bp]
        sub_path = SUB / f"submission_lb4_w1w4meta_a{int(best_alpha*1000):03d}.csv"
        pd.DataFrame({"id": test_ids, "Irrigation_Need": labels}).to_csv(sub_path, index=False)
        log(f"\n*** EMITTED: {sub_path} ***")
        from collections import Counter
        c = Counter(labels.tolist())
        print(f"  pred dist: {dict(c)}")
        primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
        n_diff = (primary["Irrigation_Need"].values != labels).sum()
        print(f"  rows differing from LB-best primary: {n_diff} ({100*n_diff/len(labels):.3f}%)")
    else:
        log(f"\nNO PASS (Δ {best_delta:+.5f} < +0.0002 OR class guardrail failed)")

    out = {
        "lb4_ba_old": float(lb4_ba), "lb4_errs_old": int(lb4_errs),
        "meta_argmax_ba": float(meta_argmax_ba),
        "meta_iso_ba_at_lb_bias": float(meta_iso_ba_at_lb_bias),
        "new_4stack_ba": float(new_4stack_ba),
        "new_4stack_errs": int(new_4stack_errs),
        "new_components_added": new_components,
        "n_components": len(component_names),
        "best_alpha": float(best_alpha), "best_delta": float(best_delta),
        "fold_best_iters": fold_iters,
        "wall_s": time.time() - t0,
    }
    out_path = ART / "w1w4_metastack_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"results → {out_path}, wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
