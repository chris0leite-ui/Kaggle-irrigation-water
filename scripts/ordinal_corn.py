"""Frank-Hall / CORN ordinal decomposition (brainstorm suggestion #3).

Decompose 3-class ordinal y in {0=Low, 1=Medium, 2=High} into two
binary XGB heads, each trained on the 43-feature dist set:

  Task A: P(y >= Medium)   — target = (y >= 1). Attacks the Low<->Medium
                             boundary (74 % of greedy+nonrule errors per
                             error_analysis_greedy_nonrule.json).
  Task B: P(y >= High)     — target = (y == 2). Attacks the Medium<->High
                             boundary (the remaining error mass, score=6
                             rows driven by low Soil_Moisture).

Frank-Hall recomposition (independent heads, post-hoc monotonicity):
  p_a = P(y >= 1)
  p_b = min(P(y >= 2), p_a)       # enforce monotone
  P(Low)    = 1 - p_a
  P(Medium) = p_a - p_b
  P(High)   = p_b
  then renormalize (cheap safety).

Protocol (learned from binhigh + nonrule postmortems):
  1. Same 5-fold stratified split, seed=42, same 43-feature dist set
     as benchmark_dist.py — OOF-aligned with every other saved .npy.
  2. Standalone tuned OOF, then FIXED-bias log-blend sweeps versus
     both the greedy blend (OOF 0.97375) and LB-best greedy+nonrule
     (OOF 0.97421). A new component that needs bias retune to show
     lift is fake signal (greedy_binhigh_minimal rule).
  3. Artefacts: oof_xgb_corn.npy, test_xgb_corn.npy,
     ordinal_corn_results.json. Submission only written if
     fixed-bias OOF lifts >= +0.0002 over LB-best.

Why this should be orthogonal to the current stack:
  Every saved OOF in scripts/artifacts/ was trained on a 3-class
  softmax objective. XGB multi:softprob fits per-class margins
  jointly with a softmax coupling. Binary logistic heads on
  ordinal cuts produce a different decision surface because the
  Bernoulli gradient focuses capacity on ONE boundary at a time
  — exactly the boundaries where error analysis says the mass is.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
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
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to benchmark_dist.add_distance_features — keep in sync."""
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


def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def recompose(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Frank-Hall 3-class probs with post-hoc monotonicity enforcement."""
    p_a = np.clip(p_a, 1e-6, 1 - 1e-6)
    p_b = np.clip(p_b, 1e-6, 1 - 1e-6)
    p_b = np.minimum(p_b, p_a)            # enforce P(y>=2) <= P(y>=1)
    p_low = 1.0 - p_a
    p_med = p_a - p_b
    p_high = p_b
    probs = np.stack([p_low, p_med, p_high], axis=1)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    logs = la + lb
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def train_binary_head(X, y_bin, X_test, skf, y_strat, xgb_params, name):
    """5-fold OOF + test for a single binary head with identical fold split."""
    oof = np.zeros(len(X), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    dte = xgb.DMatrix(X_test, enable_categorical=True)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_strat)):
        t0 = time.time()
        dtr = xgb.DMatrix(
            X.iloc[tr_idx], label=y_bin[tr_idx], enable_categorical=True
        )
        dva = xgb.DMatrix(
            X.iloc[va_idx], label=y_bin[va_idx], enable_categorical=True
        )
        booster = xgb.train(
            xgb_params,
            dtr,
            num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        bi = booster.best_iteration
        oof[va_idx] = booster.predict(dva, iteration_range=(0, bi + 1))
        test_pred += booster.predict(dte, iteration_range=(0, bi + 1)) / N_FOLDS
        auc = roc_auc_score(y_bin[va_idx], oof[va_idx])
        aucs.append(auc)
        log(f"  [{name}] fold {fold+1}/{N_FOLDS}  best_iter={bi}  "
            f"AUC={auc:.5f}  pos_rate={y_bin[tr_idx].mean():.4f}  ({time.time()-t0:.1f}s)")
    return oof, test_pred, aucs


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building 43-feature dist set")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32").astype("category")
        te[c] = te[c].map(mapping).astype("int32").astype("category")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

    # Ordinal binary targets.
    y_a = (y >= 1).astype(np.int32)   # P(y >= Medium)
    y_b = (y >= 2).astype(np.int32)   # P(y >= High)
    log(f"task A pos_rate = P(y>=Medium) = {y_a.mean():.4f}")
    log(f"task B pos_rate = P(y>=High)   = {y_b.mean():.4f}")

    xgb_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    log("training head A: P(y >= Medium)")
    p_a_oof, p_a_test, aucs_a = train_binary_head(
        X, y_a, X_test, skf, y, xgb_params, name="A"
    )

    log("training head B: P(y >= High)")
    p_b_oof, p_b_test, aucs_b = train_binary_head(
        X, y_b, X_test, skf, y, xgb_params, name="B"
    )

    log("recomposing 3-class probs (Frank-Hall + monotone clip)")
    oof = recompose(p_a_oof, p_b_oof)
    test_pred = recompose(p_a_test, p_b_test)

    # Diagnostic: how often was the monotone clip needed?
    violations_oof = float((p_b_oof > p_a_oof).mean())
    violations_test = float((p_b_test > p_a_test).mean())
    log(f"monotone clip: OOF violations={violations_oof:.4f}  "
        f"test violations={violations_test:.4f}")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias_corn, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"CORN standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    log(f"  bias = {dict(zip(CLASSES, bias_corn.round(4)))}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias_corn).argmax(axis=1))
    log(f"standalone CM (rows=true, cols=pred):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_xgb_corn.npy", oof)
    np.save(ART / "test_xgb_corn.npy", test_pred)
    np.save(ART / "oof_xgb_corn_head_a.npy", p_a_oof)
    np.save(ART / "test_xgb_corn_head_a.npy", p_a_test)
    np.save(ART / "oof_xgb_corn_head_b.npy", p_b_oof)
    np.save(ART / "test_xgb_corn_head_b.npy", p_b_test)

    # Blend sweeps with FIXED baseline bias.
    log("loading reference OOFs for blend sweeps")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")

    greedy_bias = np.array([
        0.13244116323609723, 0.568946691946548, 3.400768902044088
    ])
    greedy_tuned_oof = 0.973746084242468

    # Reconstruct LB-best (greedy + nonrule @ log-alpha=0.15), with same bias.
    oof_lbbest = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_lbbest = log_blend2(test_nonrule, test_greedy, 0.15)
    lbbest_bias = greedy_bias
    lp_lb = np.log(np.clip(oof_lbbest, 1e-9, 1.0))
    lbbest_tuned_oof = float(balanced_accuracy_score(
        y, (lp_lb + lbbest_bias).argmax(axis=1)
    ))
    log(f"baselines (fixed-bias, reproduced):")
    log(f"  greedy           OOF = {greedy_tuned_oof:.5f}")
    log(f"  greedy+nonrule   OOF = {lbbest_tuned_oof:.5f}")

    results = {
        "corn_standalone_argmax": float(argmax_bal),
        "corn_standalone_tuned": float(tuned_bal),
        "corn_bias": bias_corn.tolist(),
        "head_A_fold_aucs": [float(a) for a in aucs_a],
        "head_B_fold_aucs": [float(a) for a in aucs_b],
        "head_A_auc_mean": float(np.mean(aucs_a)),
        "head_B_auc_mean": float(np.mean(aucs_b)),
        "monotone_clip_oof_rate": violations_oof,
        "monotone_clip_test_rate": violations_test,
        "greedy_bias": greedy_bias.tolist(),
        "greedy_tuned_oof": greedy_tuned_oof,
        "lbbest_tuned_oof": lbbest_tuned_oof,
        "sweep_vs_greedy": [],
        "sweep_vs_lbbest": [],
    }

    log("sweep: log-blend alpha on CORN, vs greedy (fixed bias)")
    grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for alpha in grid:
        blend = log_blend2(oof, oof_greedy, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + greedy_bias).argmax(axis=1))
        delta = ba - greedy_tuned_oof
        results["sweep_vs_greedy"].append({
            "alpha": alpha, "oof": float(ba), "delta": float(delta)
        })
        log(f"  vs greedy   alpha={alpha:.3f}  OOF={ba:.5f}  Δ={delta:+.5f}")

    log("sweep: log-blend alpha on CORN, vs LB-best greedy+nonrule (fixed bias)")
    for alpha in grid:
        blend = log_blend2(oof, oof_lbbest, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + lbbest_bias).argmax(axis=1))
        delta = ba - lbbest_tuned_oof
        results["sweep_vs_lbbest"].append({
            "alpha": alpha, "oof": float(ba), "delta": float(delta)
        })
        log(f"  vs lbbest   alpha={alpha:.3f}  OOF={ba:.5f}  Δ={delta:+.5f}")

    best_g = max(results["sweep_vs_greedy"], key=lambda d: d["oof"])
    best_lb = max(results["sweep_vs_lbbest"], key=lambda d: d["oof"])
    results["best_vs_greedy"] = best_g
    results["best_vs_lbbest"] = best_lb
    log(f"best vs greedy:  alpha={best_g['alpha']}  OOF={best_g['oof']:.5f}  "
        f"Δ={best_g['delta']:+.5f}")
    log(f"best vs lbbest:  alpha={best_lb['alpha']}  OOF={best_lb['oof']:.5f}  "
        f"Δ={best_lb['delta']:+.5f}")

    # Submission emission: only if fixed-bias sweep vs LB-best shows >=+0.0002 lift.
    if best_lb["alpha"] > 0 and best_lb["delta"] >= 2e-4:
        blend_test = log_blend2(test_pred, test_lbbest, best_lb["alpha"])
        lp_test = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp_test + lbbest_bias).argmax(axis=1)
        sub_path = OUT / "submission_lbbest_corn_blend.csv"
        pd.DataFrame(
            {ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}
        ).to_csv(sub_path, index=False)
        log(f"wrote {sub_path}  (OOF lift +{best_lb['delta']:.5f} — candidate for LB probe)")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub_path)

        blend_oof = log_blend2(oof, oof_lbbest, best_lb["alpha"])
        cm_blend = confusion_matrix(
            y, (np.log(np.clip(blend_oof, 1e-9, 1.0)) + lbbest_bias).argmax(axis=1)
        )
        log(f"blend CM:\n"
            f"{pd.DataFrame(cm_blend, index=CLASSES, columns=CLASSES)}")
    elif best_lb["alpha"] > 0 and best_lb["delta"] > 0:
        log(f"OOF lift +{best_lb['delta']:.5f} is below +0.0002 threshold — "
            f"null; no submission.")
        results["action"] = "below_threshold_no_submit"
    else:
        log("fixed-bias sweep strictly <= baseline — null result; no submission.")
        results["action"] = "no_submission"

    with open(ART / "ordinal_corn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/ordinal_corn_results.json")


if __name__ == "__main__":
    main()
