"""XGB-dist augmented with NN-on-original prediction columns.

Idea 1 proper: the NN-on-original ensemble (oof_nn_orig_ens.npy) predicts
a smooth approximation of the rule on synthetic features. Instead of
blending its probabilities with greedy (prob-blend null — NN too weak
standalone, error magnitude hurts blend despite low Jaccard), let XGB
decide WHERE the NN signal is useful by feeding its 3 probability
columns as new features alongside the 43-feature dist set.

Hypothesis: near-boundary rows will be well-served by the NN's smooth
prediction; clean-rule rows keep their rule-based prediction via the
existing dist features. XGB learns the gating.

Protocol:
  1. Load 43-feature dist set (benchmark_dist.py style) + 3 NN-orig
     prob columns (nn_orig_p_low, nn_orig_p_med, nn_orig_p_high).
  2. Same XGB HPs as benchmark_xgb_dist.py (max_depth=7, etc.),
     5-fold stratified seed=42.
  3. Save OOF + test preds, run fixed-greedy-bias blend sweep as usual.
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
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full 43-feature dist set — same as benchmark_dist.py."""
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


def tune_log_bias(oof, y, prior):
    lp = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
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


def main():
    log("loading data + NN-orig prediction features")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    nn_tr = np.load(ART / "oof_nn_orig_ens.npy")
    nn_te = np.load(ART / "test_nn_orig_ens.npy")
    assert nn_tr.shape == (len(tr), 3), f"nn train shape {nn_tr.shape}"
    assert nn_te.shape == (len(te), 3), f"nn test shape {nn_te.shape}"
    log(f"  synth: train {tr.shape}  test {te.shape}")
    log(f"  NN-orig features: train {nn_tr.shape}  test {nn_te.shape}")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    # Add 3 NN-orig probability columns as features
    for i, cls in enumerate(["Low", "Medium", "High"]):
        tr[f"nn_orig_p_{cls.lower()}"] = nn_tr[:, i].astype(np.float32)
        te[f"nn_orig_p_{cls.lower()}"] = nn_te[:, i].astype(np.float32)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32").astype("category")
        te[c] = te[c].map(mapping).astype("int32").astype("category")

    feat_cols = num_cols + cat_cols
    log(f"features: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"  NN-orig cols in feat set: "
        f"{[c for c in feat_cols if c.startswith('nn_orig_')]}")

    X = tr[feat_cols]
    X_test = te[feat_cols]
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

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

    log("5-fold XGB with nn_orig features")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"bal_acc(argmax)={bal:.5f}  ({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    _, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"XGB-dist + nn_orig feats standalone: argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")

    # Reference: vanilla xgb-dist OOF (no NN features)
    vanilla = np.load(ART / "oof_xgb_vanilla_dist.npy")
    _, vanilla_tuned = tune_log_bias(vanilla, y, prior)
    log(f"reference (vanilla xgb_dist tuned OOF): {vanilla_tuned:.5f}  "
        f"(Δ = {tuned_bal - vanilla_tuned:+.5f})")

    np.save(ART / "oof_xgb_dist_nn_orig.npy", oof)
    np.save(ART / "test_xgb_dist_nn_orig.npy", test_pred)

    # ---- fixed-greedy-bias blend sweep ----
    log("fixed-greedy-bias log-blend sweep")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    bias_greedy = np.array(greedy_res["greedy_bias"])
    tuned_greedy = float(greedy_res["greedy_tuned_oof"])
    log(f"greedy baseline tuned = {tuned_greedy:.5f}  bias={bias_greedy.round(4).tolist()}")

    # diagnostic
    pred_xnn = oof.argmax(axis=1)
    pred_gr = oof_greedy.argmax(axis=1)
    e_x = set(np.where(pred_xnn != y)[0])
    e_g = set(np.where(pred_gr != y)[0])
    jac = len(e_x & e_g) / (len(e_x | e_g) or 1)
    log(f"  error Jaccard vs greedy = {jac:.4f}  "
        f"(xgb+nn errs={len(e_x)}, greedy errs={len(e_g)})")

    sweep = []
    best = {"alpha": 0.0, "oof": tuned_greedy, "delta_vs_greedy": 0.0}
    grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0]
    for alpha in grid:
        if alpha == 0.0:
            blend_oof = oof_greedy
        elif alpha == 1.0:
            blend_oof = oof
        else:
            blend_oof = log_blend2(oof, oof_greedy, alpha)
        lp = np.log(np.clip(blend_oof, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + bias_greedy).argmax(axis=1))
        delta = ba - tuned_greedy
        marker = ""
        if ba > best["oof"]:
            best = {"alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)}
            marker = "  <- best"
        sweep.append({"alpha": alpha, "oof": float(ba), "delta_vs_greedy": float(delta)})
        log(f"  alpha_xnn={alpha:.2f}  OOF (fixed bias) = {ba:.5f}  Δ = {delta:+.5f}{marker}")

    results = {
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "xgb_params": xgb_params,
        "standalone_argmax": float(argmax_bal),
        "standalone_tuned": float(tuned_bal),
        "vanilla_dist_tuned": float(vanilla_tuned),
        "delta_vs_vanilla": float(tuned_bal - vanilla_tuned),
        "greedy_tuned_oof": tuned_greedy,
        "greedy_bias": bias_greedy.tolist(),
        "error_jaccard_vs_greedy": float(jac),
        "sweep_log_blend": sweep,
        "best": best,
    }

    # Decision
    if best["alpha"] == 0.0 or best["delta_vs_greedy"] < 1e-5:
        log("no OOF lift — null result, no submission")
        results["action"] = "no_submission"
    elif best["delta_vs_greedy"] < 5e-4:
        log(f"lift {best['delta_vs_greedy']:+.5f} below +0.0005 threshold — borderline")
        a = best["alpha"]
        bl = log_blend2(test_pred, test_greedy, a) if 0 < a < 1 else (
            test_greedy if a == 0 else test_pred)
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_xgb_dist_nn_orig_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"borderline submission written to {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        a = best["alpha"]
        bl = log_blend2(test_pred, test_greedy, a) if 0 < a < 1 else (
            test_greedy if a == 0 else test_pred)
        lp = np.log(np.clip(bl, 1e-9, 1.0))
        preds = (lp + bias_greedy).argmax(axis=1)
        sub = OUT / "submission_xgb_dist_nn_orig_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "xgb_dist_nn_orig_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/xgb_dist_nn_orig_results.json")


if __name__ == "__main__":
    main()
