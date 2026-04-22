"""Follow-up A: soft flip-correction, architecturally distinct from
all TE-regression variants.

Pipeline (OOF-clean, leak-free on synthetic train):
  1. Binary XGB `P_flip(x) = P(y != rule_pred | x)`
     - 5-fold stratified on GLOBAL y (same seed=42 split used
       everywhere else, so OOFs align).
     - Full 630k training set with binary label y_flip.
     - 43-col dist features.
  2. 3-class XGB `P_y_given_flipped(x)`
     - Trained ONLY on the ~10,304 flipped rows.
     - 5-fold inner stratified split on FLIPPED-ROW y-classes.
     - On val rows of each outer fold, predicts 3-class probs.
  3. Composite corrected prob:
        P_y(c | x) = P_flip * P_y_given_flipped(c | x)
                   + (1 - P_flip) * onehot(rule_pred(x), c)
     Soft version of the 2026-04-20 `gated_v3` hard-gate null,
     which failed because threshold-routing let false-positive
     P_flip spill anti-rule predictions onto clean rows. Soft
     weighting downweights those spills automatically.
  4. Blend composite into greedy (0.97375) and LB-best greedy +
     nonrule @0.15 (0.97421), fixed greedy bias, log-space alpha
     sweep. LB-probe if Delta >= +0.0005.

Outputs:
  scripts/artifacts/oof_flip_correction.npy        (630_000, 3)
  scripts/artifacts/test_flip_correction.npy       (270_000, 3)
  scripts/artifacts/oof_pflip.npy                  (630_000,)
  scripts/artifacts/test_pflip.npy                 (270_000,)
  scripts/artifacts/flip_correction_results.json
  submissions/submission_flip_correction_vs_{greedy,lbbest}.csv  (if lift)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


def build_xy(tr: pd.DataFrame, te: pd.DataFrame):
    drop = {ID, TARGET}
    num_cols = [c for c in tr.select_dtypes(include=[np.number]).columns if c not in drop]
    cat_cols = [c for c in tr.columns if c not in num_cols and c not in drop]
    X = tr[num_cols + cat_cols].copy()
    X_test = te[num_cols + cat_cols].copy()
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        X[c] = tr[c].map(mapping).astype("int32").astype("category")
        X_test[c] = te[c].map(mapping).astype("int32").astype("category")
    return X, X_test, cat_cols


def tune_log_bias(p, y, prior, grid=None):
    if grid is None:
        grid = np.linspace(-3.0, 3.0, 61)
    log_p = np.log(np.clip(p, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    t0 = time.time()
    log("loading data + dist features")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    tr = add_dist_features(tr)
    te = add_dist_features(te)
    X, X_test, cat_cols = build_xy(tr, te)
    log(f"  features: {X.shape[1]}")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    rule_pred = tr["rule_pred"].values.astype(np.int32)
    rule_pred_test = te["rule_pred"].values.astype(np.int32)
    y_flip = (y != rule_pred).astype(np.int32)
    flip_rate = y_flip.mean()
    log(f"  synth flip rate = {flip_rate:.5f}  ({y_flip.sum():,} / {len(y):,})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # ---------- Step 1: binary P_flip, 5-fold stratified on GLOBAL y ----------
    log("step 1/3: binary P_flip, 5-fold stratified on y")
    oof_pflip = np.zeros(len(tr), dtype=np.float64)
    test_pflip = np.zeros(len(te), dtype=np.float64)
    flip_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
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
    dte_all = xgb.DMatrix(X_test, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_f = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y_flip[tr_idx],
                          enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y_flip[va_idx],
                          enable_categorical=True)
        booster = xgb.train(
            flip_params, dtr, num_boost_round=3000,
            evals=[(dva, "val")],
            early_stopping_rounds=80,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof_pflip[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pflip += booster.predict(dte_all, iteration_range=(0, bi + 1)) / N_FOLDS
        fold_auc = roc_auc_score(y_flip[va_idx], oof_pflip[va_idx])
        log(f"  P_flip fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"AUC={fold_auc:.4f}  ({time.time()-t_f:.1f}s)")
    oof_auc = roc_auc_score(y_flip, oof_pflip)
    log(f"  OOF P_flip AUC = {oof_auc:.5f}")
    np.save(ART / "oof_pflip.npy", oof_pflip)
    np.save(ART / "test_pflip.npy", test_pflip)

    # ---------- Step 2: 3-class prob on flipped rows only ----------
    log("step 2/3: 3-class P_y_given_flipped, trained only on flipped rows")
    oof_dir = np.zeros((len(tr), 3), dtype=np.float64)
    test_dir = np.zeros((len(te), 3), dtype=np.float64)
    dir_params = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )
    # Outer 5-fold split on GLOBAL y (same as step 1). Within each fold,
    # train the 3-class head on the flipped subset of the outer-train,
    # predict for ALL val rows (flipped or not). Predictions for clean
    # rows are garbage by design; the blend composite multiplies them by
    # tiny P_flip and they vanish.
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t_f = time.time()
        flipped_tr = tr_idx[y_flip[tr_idx] == 1]
        log(f"  dir fold {fold+1}/{N_FOLDS}  flipped_train={len(flipped_tr):,}  "
            f"val={len(va_idx):,}")
        dtr = xgb.DMatrix(X.iloc[flipped_tr], label=y[flipped_tr],
                          enable_categorical=True)
        # No native val for early stop: use a held-out 10 % of flipped_tr.
        n_hold = max(500, int(0.1 * len(flipped_tr)))
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(flipped_tr))
        hold = flipped_tr[perm[:n_hold]]
        rest = flipped_tr[perm[n_hold:]]
        dtr2 = xgb.DMatrix(X.iloc[rest], label=y[rest], enable_categorical=True)
        dhold = xgb.DMatrix(X.iloc[hold], label=y[hold], enable_categorical=True)
        booster = xgb.train(
            dir_params, dtr2, num_boost_round=1500,
            evals=[(dhold, "hold")],
            early_stopping_rounds=60,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        dva = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        oof_dir[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_dir += booster.predict(dte_all, iteration_range=(0, bi + 1)) / N_FOLDS
        log(f"     best_iter={bi}  ({time.time()-t_f:.1f}s)")

    # ---------- Step 3: composite corrected probabilities ----------
    log("step 3/3: composite P_y = P_flip * P_dir + (1-P_flip) * onehot(rule)")
    rule_oh_train = np.zeros((len(tr), 3), dtype=np.float64)
    rule_oh_train[np.arange(len(tr)), rule_pred] = 1.0
    rule_oh_test = np.zeros((len(te), 3), dtype=np.float64)
    rule_oh_test[np.arange(len(te)), rule_pred_test] = 1.0
    oof_comp = oof_pflip[:, None] * oof_dir + (1 - oof_pflip)[:, None] * rule_oh_train
    test_comp = test_pflip[:, None] * test_dir + (1 - test_pflip)[:, None] * rule_oh_test
    oof_comp = np.clip(oof_comp, 1e-6, 1.0)
    test_comp = np.clip(test_comp, 1e-6, 1.0)
    oof_comp /= oof_comp.sum(axis=1, keepdims=True)
    test_comp /= test_comp.sum(axis=1, keepdims=True)
    np.save(ART / "oof_flip_correction.npy", oof_comp)
    np.save(ART / "test_flip_correction.npy", test_comp)

    prior = np.bincount(y) / len(y)
    argmax_bal = balanced_accuracy_score(y, oof_comp.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(oof_comp, y, prior)
    log(f"composite standalone  argmax={argmax_bal:.5f}  "
        f"tuned={tuned_bal:.5f}  bias={bias.round(3).tolist()}")

    # ---------- Blend sweeps (fixed greedy bias) ----------
    log("blend sweeps (fixed greedy bias)")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nr = np.load(ART / "oof_xgb_nonrule.npy")
    test_nr = np.load(ART / "test_xgb_nonrule.npy")
    nonrule_res = json.loads((ART / "nonrule_results.json").read_text())
    bias_greedy = np.array(nonrule_res["greedy_bias"])
    greedy_oof = nonrule_res["greedy_tuned_oof"]
    oof_lbbest = log_blend2(oof_nr, oof_greedy, 0.15)
    test_lbbest = log_blend2(test_nr, test_greedy, 0.15)
    lbbest_oof = balanced_accuracy_score(
        y, (np.log(np.clip(oof_lbbest, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    )
    log(f"  greedy={greedy_oof:.5f}   LB-best (greedy+nonrule@0.15)={lbbest_oof:.5f}")

    grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40]
    results = {
        "flip_rate": float(flip_rate),
        "oof_pflip_auc": float(oof_auc),
        "standalone_argmax": float(argmax_bal),
        "standalone_tuned": float(tuned_bal),
        "greedy_oof": float(greedy_oof),
        "lbbest_oof": float(lbbest_oof),
    }
    for label, p_base, base_oof in [
        ("vs_greedy", oof_greedy, greedy_oof),
        ("vs_lbbest", oof_lbbest, lbbest_oof),
    ]:
        rows = []
        for a in grid:
            b = log_blend2(oof_comp, p_base, a)
            ba = balanced_accuracy_score(
                y, (np.log(np.clip(b, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
            )
            d = ba - base_oof
            rows.append({"alpha": a, "oof": float(ba), "delta": float(d)})
            log(f"  [{label}] alpha={a:.3f}  OOF={ba:.5f}  Δ={d:+.5f}")
        results[label] = {"sweep": rows,
                          "best": max(rows, key=lambda r: r["oof"])}

    # Submission decisions.
    results["submissions"] = []
    for label, p_test_base in [("vs_greedy", test_greedy),
                               ("vs_lbbest", test_lbbest)]:
        best = results[label]["best"]
        d = best["delta"]
        if best["alpha"] == 0.0 or d <= 0:
            log(f"  [{label}] no lift -> no submission")
            results["submissions"].append({"label": label, "action": "no_submission"})
            continue
        b_test = log_blend2(test_comp, p_test_base, best["alpha"])
        lp = np.log(np.clip(b_test, 1e-9, 1.0)) + bias_greedy
        preds = lp.argmax(axis=1)
        path = SUB / f"submission_flip_correction_{label}.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            path, index=False
        )
        # OOF confusion at the best alpha.
        cm = confusion_matrix(
            y, (np.log(np.clip(
                log_blend2(oof_comp,
                           oof_greedy if label == "vs_greedy" else oof_lbbest,
                           best["alpha"]), 1e-9, 1.0)) + bias_greedy
               ).argmax(axis=1)
        )
        log(f"  [{label}] alpha={best['alpha']:.3f}  OOF={best['oof']:.5f}  "
            f"Δ={d:+.5f}  ->  {path}")
        log(f"  OOF confusion (rows, Low/Medium/High):\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
        results["submissions"].append({
            "label": label, "path": str(path),
            "alpha": best["alpha"], "oof": best["oof"],
            "delta": d, "lb_probe_threshold_met": d >= 5e-4,
        })

    with open(ART / "flip_correction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote flip_correction_results.json   ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
