"""CMA-ES direct optimization of macro-recall over a simplex of components.

Why this is structurally different from greedy / LR / convex QP:
  - Convex log-loss surrogates (J6 QP) misalign with macro-recall + fixed bias.
  - Greedy is myopic; LR's `class_weight='balanced'` fits log-loss not bal_acc.
  - CMA-ES is gradient-free; we optimize the ACTUAL macro-recall metric at
    the fixed recipe bias [1.4324, 1.4689, 3.4008] over the K-simplex.
  - Honest 5-fold nested CV: optimize on 4 train-fold rows, score on held-out.
  - This gives the mathematical upper bound on what any constant-weight
    blend over the chosen components can achieve under the LB-best decision rule.

Components (K=7 anchors): we pick the strongest from each model family.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cma
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                            load_y, normed)


ART = Path("scripts/artifacts")
SEED = 42
N_FOLDS = 5
EPS = 1e-12


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def softmax_w(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def neg_bal_acc(z, oofs_list, y_idx, y_true, bias):
    """Objective: log-blend at simplex weights softmax(z), eval bal_acc, negate."""
    w = softmax_w(z)
    P = log_blend(oofs_list, w)
    pred = (np.log(np.clip(P, EPS, 1.0)) + bias).argmax(1)
    pred_idx = pred[y_idx] if y_idx is not None else pred
    y_eval = y_true[y_idx] if y_idx is not None else y_true
    return -balanced_accuracy_score(y_eval, pred_idx)


def fit_one_fold(oofs_list_tr, y_tr, bias, K, sigma0=0.5, maxiter=80):
    """Run CMA-ES on the train fold; returns optimal softmax(z) weights."""
    z0 = np.zeros(K - 1)  # K-1 free dims; we anchor z[0]=0 to break softmax invariance
    # Use full K dims but fix initial population symmetric — softmax invariance
    # over additive constant means optimization is over K-1 effective dims, but
    # CMA can handle the redundancy; just keep dim=K and add small L2 in z.
    es = cma.CMAEvolutionStrategy(
        np.zeros(K), sigma0,
        {"maxiter": maxiter, "verbose": -9, "seed": SEED, "popsize": 14},
    )
    while not es.stop():
        zs = es.ask()
        scores = [neg_bal_acc(z, oofs_list_tr, None, y_tr, bias) for z in zs]
        es.tell(zs, scores)
    return es.result.xbest


def main():
    log("Loading y + experts")
    y = load_y()
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    realmlp_o, realmlp_t = L("realmlp")
    nr_raw_o, nr_raw_t = L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr_raw_o, nr_raw_t, y)
    leaf_o, leaf_t = L("leaf_ote_meta_v2")
    dig_o, dig_t = L("xgb_dist_digits")
    recipe_o, recipe_t = L("recipe_full_te")

    components = {
        "lb_best_3stack": (lb3_o, lb3_t),
        "xgb_metastack_iso": (meta_iso_o, meta_iso_t),
        "realmlp": (realmlp_o, realmlp_t),
        "xgb_nonrule_iso": (nr_o, nr_t),
        "leaf_ote_meta_v2": (leaf_o, leaf_t),
        "xgb_dist_digits": (dig_o, dig_t),
        "recipe_full_te": (recipe_o, recipe_t),
    }
    names = list(components.keys())
    K = len(names)
    oofs = [components[n][0] for n in names]
    tests = [components[n][1] for n in names]
    log(f"Components K={K}: {names}")

    # 1. In-sample full-fit for upper bound
    log("Phase 1: in-sample CMA-ES full-fit for upper bound")
    z_full = fit_one_fold(oofs, y, BIAS, K, sigma0=0.5, maxiter=120)
    w_full = softmax_w(z_full)
    P_full = log_blend(oofs, w_full)
    bal_full = balanced_accuracy_score(
        y, (np.log(np.clip(P_full, EPS, 1)) + BIAS).argmax(1))
    log(f"  In-sample best bal_acc={bal_full:.6f}")
    log(f"  Weights: {dict(zip(names, [round(float(x), 4) for x in w_full]))}")

    # 2. Honest 5-fold CV: fit on tr_idx, predict on va_idx
    log("Phase 2: nested 5-fold CV (honest)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_cv = np.zeros((len(y), 3), dtype=np.float32)
    fold_records = []
    test_acc = np.zeros(tests[0].shape, dtype=np.float32)
    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        oofs_tr = [o[tr] for o in oofs]
        z = fit_one_fold(oofs_tr, y[tr], BIAS, K, sigma0=0.5, maxiter=80)
        w = softmax_w(z)
        # Predict on val + test
        oofs_va = [o[va] for o in oofs]
        Pv = log_blend(oofs_va, w)
        oof_cv[va] = Pv
        Pte = log_blend(tests, w)
        test_acc += Pte / N_FOLDS
        fold_records.append({"fold": fi + 1, "weights": [float(x) for x in w],
                             "wall_s": round(time.time() - t0, 2)})
        log(f"  fold {fi+1}: weights={[round(float(x),3) for x in w]} "
            f"wall={time.time()-t0:.1f}s")

    bal_cv = balanced_accuracy_score(
        y, (np.log(np.clip(oof_cv, EPS, 1)) + BIAS).argmax(1))
    log(f"Nested CV bal_acc={bal_cv:.6f}")
    overfit_gap = bal_full - bal_cv
    log(f"Overfit gap (full-fit - CV): {overfit_gap:+.5f}")

    # Save results + artifacts
    np.save(ART / "oof_cmaes_blend.npy", oof_cv)
    np.save(ART / "test_cmaes_blend.npy", test_acc)
    out = {
        "components": names,
        "K": K,
        "in_sample_bal_acc": bal_full,
        "in_sample_weights": dict(zip(names, [float(x) for x in w_full])),
        "cv_bal_acc": bal_cv,
        "overfit_gap": overfit_gap,
        "fold_records": fold_records,
        "lb_best_3stack_baseline": balanced_accuracy_score(
            y, (np.log(np.clip(lb3_o, EPS, 1)) + BIAS).argmax(1)),
    }
    (ART / "cmaes_macro_recall_results.json").write_text(json.dumps(out, indent=2))
    log("Saved oof_cmaes_blend.npy + test + results JSON")


if __name__ == "__main__":
    main()
