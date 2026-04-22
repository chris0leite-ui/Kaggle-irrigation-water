"""Self-distillation: train fresh XGB-dist to match greedy+nonrule soft probs.

Tier-2 follow-up. Forces a different inductive path: instead of fitting
one-hot labels, fit the teacher's 3-way class distribution. If the
student's smoothness pattern differs from direct-label XGB, the
blend might add orthogonal signal.

Protocol:
  - Teacher OOF: greedy+nonrule (LB-best, OOF 0.97421).
  - Student: XGB multi-class trained on 43-feat dist set with
    SOFT LABELS via xgb.DMatrix's weight + distribute approach:
    we fit one-hot argmax of teacher as pseudo-target but use
    teacher's max_prob as sample weight (confident rows weight 1,
    uncertain rows weight < 1). This is a cheap approximation to
    full soft-label distillation (xgboost doesn't natively support
    soft probs as targets without custom objective).
  - 5-fold stratified seed=42.
  - Fixed-greedy-bias blend sweep vs LB-best.

Low-risk experiment; expected +0.0000 to +0.0003.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
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


def add_distance_features(df):
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8); norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8); windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage, ACTIVE_STAGES), 2, 0).astype(np.int8)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry; out["norain"] = norain; out["hot"] = hot
    out["windy"] = windy; out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values)).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


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
    log("loading data + teacher OOFs")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    oof_teacher = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_teacher = log_blend2(test_nonrule, test_greedy, 0.15)
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])

    log("building dist features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y_true = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y_true) / len(y_true)

    # Distillation target: teacher's argmax, weighted by max_prob confidence.
    # This is the "hard distillation" approximation that XGB supports natively.
    y_teach = oof_teacher.argmax(axis=1).astype(np.int32)
    w_teach = oof_teacher.max(axis=1)  # (n,)
    log(f"teacher argmax == y_true rate: {(y_teach == y_true).mean():.5f}")
    log(f"teacher max_prob: min {w_teach.min():.3f}  med {np.median(w_teach):.3f}  "
        f"max {w_teach.max():.3f}")

    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
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

    log("5-fold XGB-distill (fit on teacher's argmax with conf-weighted samples)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stu = np.zeros((len(tr), 3), dtype=np.float64)
    test_stu = np.zeros((len(te), 3), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_true)):
        t0 = time.time()
        dtr = xgb.DMatrix(
            X.iloc[tr_idx], label=y_teach[tr_idx],
            weight=w_teach[tr_idx], enable_categorical=True,
        )
        # val against TRUE labels (we care about the real metric, not the teacher)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y_true[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof_stu[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_stu += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        bal = balanced_accuracy_score(y_true[va_idx], oof_stu[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y_true, oof_stu.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof_stu, y_true, prior)
    log(f"distill student standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")

    np.save(ART / "oof_xgb_distill.npy", oof_stu)
    np.save(ART / "test_xgb_distill.npy", test_stu)

    # ---- blend vs LB-best ----
    lbbest_ba = balanced_accuracy_score(y_true,
        (np.log(np.clip(oof_teacher, 1e-9, 1.0)) + bias_greedy).argmax(axis=1))
    log(f"LB-best ref: {lbbest_ba:.5f}")

    # Jaccard vs teacher
    s_p = oof_stu.argmax(axis=1)
    t_p = oof_teacher.argmax(axis=1)
    e_s = set(np.where(s_p != y_true)[0])
    e_t = set(np.where(t_p != y_true)[0])
    jac = len(e_s & e_t) / (len(e_s | e_t) or 1)
    log(f"error Jaccard student vs teacher: {jac:.4f}  "
        f"stu errs={len(e_s)}  teach errs={len(e_t)}")

    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    sweep = []
    best = {"alpha": 0.0, "oof": lbbest_ba, "delta": 0.0}
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_teacher
        else:
            blend_oof = log_blend2(oof_stu, oof_teacher, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y_true, (lp + bias_greedy).argmax(axis=1))
        delta = ba - lbbest_ba
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta": float(delta)}
            marker = "  <- best"
        sweep.append({"alpha": alpha, "oof": float(ba), "delta_vs_lbbest": float(delta)})
        log(f"  alpha_stu={alpha:.2f}  OOF={ba:.5f}  Δ={delta:+.5f}{marker}")

    results = {
        "student_argmax": float(argmax_bal),
        "student_tuned": float(tuned_bal),
        "lbbest_reference_oof": float(lbbest_ba),
        "error_jaccard_vs_teacher": float(jac),
        "sweep": sweep,
        "best": best,
    }
    if best["delta"] < 1e-5:
        log("no lift — null")
        results["action"] = "no_submission"
    else:
        log(f"lift Δ={best['delta']:+.5f}")
        results["action"] = "check_lift"

    with open(ART / "self_distill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/self_distill_results.json")


if __name__ == "__main__":
    main()
