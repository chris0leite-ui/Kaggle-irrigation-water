"""Pseudo-labeling v2 on greedy+nonrule base.

Prior attempt (2026-04-21 pseudo_label_hybrid) used tau=0.95 on the
weaker hybrid_v3 base and compounded boundary errors (Medium<->High
mistakes got encoded in pseudo-labels, pushing decision surface in
the wrong direction). Lesson logged: "pseudo-labeling compounds
boundary errors when the labeler is systematically wrong on the
boundary."

v2 design changes:
  1. Stronger base: greedy+nonrule (LB-best 0.97421 vs hybrid_v3's
     0.97352). Fewer systematic boundary errors to propagate.
  2. Higher threshold: tau=0.99 (vs 0.95). Only rows the base is
     nearly certain about get pseudo-labels.
  3. Class-restricted: LOW-ONLY pseudo-labels. From error analysis:
     - Low class has 99.58% recall at greedy+nonrule's operating point
       (368,267 correct / 369,917 total).
     - Low errors are tiny (1,650 misclassified out of 369k).
     - Pseudo-labeling Low only (tau=0.99) gives us ~90% of test's
       Low rows with minimal error injection.
     - AVOIDS Medium/High where boundary errors live (~10k misclass).

  This is essentially "use only the clean-class predictions as
  pseudo-labels". Expected lift: +0.0002-0.001.

Protocol:
  1. Generate test pseudo-labels using greedy+nonrule @ its tuned bias.
  2. Filter: only rows where pred_class==0 (Low) AND max_prob > 0.99.
  3. Augment train: 630k real + N pseudo-Low (retrain XGB-dist).
  4. Compare retrained-XGB-dist OOF to vanilla.
  5. If lift, re-run hybrid + greedy pipeline with augmented base.

Simpler path (what we do here): augment the XGB-nonrule leg, since
it's the only architecturally-diverse component, and re-blend.
Expected delta is small but measurable.

Artefacts:
  scripts/artifacts/oof_xgb_nonrule_pl.npy
  scripts/artifacts/test_xgb_nonrule_pl.npy
  scripts/artifacts/pseudo_label_v2_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

RULE_COLS = {
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Mulching_Used", "Crop_Growth_Stage",
}
DROP_COLS = {ID, TARGET}

TAU = 0.99
ALLOWED_PSEUDO_CLASSES = [0]  # Low only

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            bals = []
            for g in grid:
                base[k] = bias[k] + g
                bals.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(bals))
            if bals[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = bals[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def main():
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    all_cols = [c for c in tr.columns if c not in DROP_COLS]
    nonrule_cols = [c for c in all_cols if c not in RULE_COLS]
    log(f"non-rule features ({len(nonrule_cols)}): {nonrule_cols}")

    log("loading greedy+nonrule test probs + greedy bias")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])

    # Generate pseudo-labels from greedy+nonrule @ alpha=0.15
    test_base = log_blend2(test_nonrule, test_greedy, 0.15)
    # Apply tuned bias in logit space, re-softmax for "calibrated" probs.
    lp = np.log(np.clip(test_base, 1e-9, 1.0)) + bias_greedy
    exp_lp = np.exp(lp - lp.max(axis=1, keepdims=True))
    test_probs_adj = exp_lp / exp_lp.sum(axis=1, keepdims=True)

    test_pred = test_probs_adj.argmax(axis=1)
    test_maxp = test_probs_adj.max(axis=1)
    log(f"  test pred distribution (argmax): "
        f"{np.bincount(test_pred, minlength=3).tolist()}")
    log(f"  max-prob distribution: min={test_maxp.min():.4f}  "
        f"med={np.median(test_maxp):.4f}  max={test_maxp.max():.4f}")

    # Filter: tau=0.99, Low only
    mask_pl = (test_maxp > TAU) & np.isin(test_pred, ALLOWED_PSEUDO_CLASSES)
    n_pl = int(mask_pl.sum())
    log(f"pseudo-labeled test rows (tau={TAU}, classes={ALLOWED_PSEUDO_CLASSES}): "
        f"{n_pl} / {len(te)} ({100*n_pl/len(te):.1f}%)")

    if n_pl < 1000:
        log("too few pseudo-labeled rows — null")
        with open(ART / "pseudo_label_v2_results.json", "w") as f:
            json.dump({"n_pseudo_labels": n_pl, "action": "no_submission",
                       "reason": "too_few_pl"}, f, indent=2)
        return

    # Build augmented train set for XGB-nonrule retrain
    X_tr = tr[nonrule_cols].copy()
    X_te = te[nonrule_cols].copy()

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in nonrule_cols if c not in num_cols]
    for c in cat_cols:
        mp = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X_tr[c] = tr[c].map(mp).astype("int32").astype("category")
        X_te[c] = te[c].map(mp).astype("int32").astype("category")

    y_tr = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y_tr) / len(y_tr)

    # Pseudo-labeled subset (use same numeric types)
    X_pl = X_te.loc[mask_pl].copy()
    y_pl = test_pred[mask_pl].astype(np.int32)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )

    log(f"retraining XGB-nonrule with pseudo-labels (real {len(tr)} + pl {n_pl})")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred_pl = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    dte = xgb.DMatrix(X_te, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr)):
        t0 = time.time()
        # Augment TRAIN (not val) with pseudo-labels
        X_comb = pd.concat([X_tr.iloc[tr_idx], X_pl], axis=0, ignore_index=True)
        y_comb = np.concatenate([y_tr[tr_idx], y_pl])
        # Pseudo-labels get weight 0.5 (discount for label uncertainty)
        w_comb = np.ones(len(y_comb), dtype=np.float32)
        w_comb[len(tr_idx):] = 0.5

        dtr = xgb.DMatrix(X_comb, label=y_comb, weight=w_comb, enable_categorical=True)
        dva = xgb.DMatrix(X_tr.iloc[va_idx], label=y_tr[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred_pl += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        bal = balanced_accuracy_score(y_tr[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  bal_acc={bal:.5f}  "
            f"({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y_tr, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y_tr, prior)
    log(f"pl-nonrule standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")

    np.save(ART / "oof_xgb_nonrule_pl.npy", oof)
    np.save(ART / "test_xgb_nonrule_pl.npy", test_pred_pl)

    # Compare to vanilla XGB-nonrule
    oof_vanilla = np.load(ART / "oof_xgb_nonrule.npy")
    _, vanilla_tuned = tune_log_bias(oof_vanilla, y_tr, prior)
    log(f"vanilla nonrule tuned: {vanilla_tuned:.5f}  "
        f"Δ_pl = {tuned_bal - vanilla_tuned:+.5f}")

    # Blend sweep vs greedy+(PL-nonrule) vs greedy+(vanilla-nonrule)
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")

    # LB-best reference: greedy + vanilla-nonrule @ 0.15
    oof_lbbest = log_blend2(oof_vanilla, oof_greedy, 0.15)
    lbbest_ba = balanced_accuracy_score(y_tr,
        (np.log(np.clip(oof_lbbest, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"LB-best ref OOF: {lbbest_ba:.5f}")

    log("sweep: greedy + PL-nonrule at various alphas (fixed greedy bias)")
    grid = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep = []
    best = {"alpha": 0.0, "oof": lbbest_ba, "delta": 0.0}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_greedy
        else:
            blend_oof = log_blend2(oof, oof_greedy, alpha)
        lp_b = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y_tr, (lp_b + bias_greedy).argmax(axis=1))
        delta = ba - lbbest_ba
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta": float(delta)}
            marker = "  <- best"
        sweep.append({"alpha": alpha, "oof": float(ba), "delta_vs_lbbest": float(delta)})
        log(f"  alpha_pl={alpha:.2f}  OOF={ba:.5f}  "
            f"Δ_lbbest={delta:+.5f}{marker}")

    results = {
        "tau": TAU,
        "allowed_pseudo_classes": ALLOWED_PSEUDO_CLASSES,
        "n_pseudo_labels": n_pl,
        "pl_nonrule_tuned": float(tuned_bal),
        "vanilla_nonrule_tuned": float(vanilla_tuned),
        "lbbest_reference_oof": float(lbbest_ba),
        "sweep": sweep,
        "best": best,
    }

    if best["delta"] < 1e-5:
        log("no lift — null")
        results["action"] = "no_submission"
    elif best["delta"] < 3e-4:
        log(f"Δ={best['delta']:+.5f} below +0.0003 — borderline")
        a = best["alpha"]
        test_blend = log_blend2(test_pred_pl, np.load(ART / "test_greedy_blend.npy"), a) \
            if a > 0 else np.load(ART / "test_greedy_blend.npy")
        lp_t = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds_out = (lp_t + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_pl_nonrule_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds_out]}).to_csv(
            sub, index=False)
        log(f"wrote borderline {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best["alpha"]
        test_greedy_arr = np.load(ART / "test_greedy_blend.npy")
        test_blend = log_blend2(test_pred_pl, test_greedy_arr, a) if a > 0 else test_greedy_arr
        lp_t = np.log(np.clip(test_blend, 1e-9, 1.0))
        preds_out = (lp_t + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_greedy_pl_nonrule_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds_out]}).to_csv(
            sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "pseudo_label_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/pseudo_label_v2_results.json")


if __name__ == "__main__":
    main()
