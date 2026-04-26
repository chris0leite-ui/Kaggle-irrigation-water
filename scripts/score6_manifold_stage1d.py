"""Stage 1d: depth sweep + ensemble + row-level blend test.

Stage 1c showed top-K precision is non-monotonic in AUC: v2 (depth=6)
beats LR oracle (AUC 0.85) on top-K despite lower AUC. Stage 1d tests:
  (1) depth ∈ {3,4,5,6,7} XGB sweep at fixed feature set — find depth
      with optimal macro-delta-at-best-n, not optimal AUC.
  (2) ensemble of v2 + best depth-tuned XGB + best LR — does aggregation
      lift top-K precision?
  (3) row-level blend test: apply v2's P_H as additive log-prob to LB-best
      primary at score=6 only, sweep alpha. This uses ALL of v2's
      ranking, not just top-K, and may extract more signal than hard
      override.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, add_distance_features, fast_bal_acc, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
OUT = ART / "score6_manifold_stage1d_results.json"

N_H_TOTAL = 21009
N_M_TOTAL = 239074


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro_delta_local(y_3: np.ndarray, override_idx: np.ndarray) -> float:
    c = int((y_3[override_idx] == 2).sum())
    w_m = int((y_3[override_idx] == 1).sum())
    return (c / N_H_TOTAL - w_m / N_M_TOTAL) / 3.0


def topk_eval(name: str, score: np.ndarray, y_3: np.ndarray, y_bin: np.ndarray,
              ns: list[int]) -> dict:
    auc = float(roc_auc_score(y_bin, score))
    order = np.argsort(-score)
    out = {"auc": auc, "topn": {}}
    for n in ns:
        if n > len(y_bin):
            continue
        idx = order[:n]
        c = int((y_3[idx] == 2).sum())
        w = int((y_3[idx] == 1).sum())
        out["topn"][f"n_{n}"] = {"correct": c, "wrong_m": w,
                                  "prec": c / n,
                                  "macro_delta": macro_delta_local(y_3, idx)}
    bn = max(out["topn"].items(), key=lambda kv: kv[1]["macro_delta"])
    out["best_n"] = bn[0]
    out["best_macro_delta"] = bn[1]["macro_delta"]
    log(f"  {name:24s} AUC={auc:.4f}  best_n={bn[0]}  c={bn[1]['correct']:>3d}  "
        f"prec={bn[1]['prec']:.3f}  macro_Δ={bn[1]['macro_delta']:+.6f}")
    return out


def main() -> None:
    log("loading train")
    tr = pd.read_csv("data/train.csv")
    tr = add_distance_features(tr)

    y_full = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)
    score = tr["dgp_score"].to_numpy().astype(np.int8)

    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    teacher = log_blend([oof_r, oof_s1, oof_s7], np.array([0.25, 0.35, 0.40]))
    bias = np.array([1.4324, 1.4689, 3.4008])
    teacher_pred = np.argmax(np.log(np.clip(teacher, 1e-9, 1.0)) + bias, axis=1)

    domain = (score == 6) & (teacher_pred == 1)
    n_dom = int(domain.sum())
    log(f"domain: {n_dom:,}")

    y_3 = y_full[domain]
    y_bin = (y_3 == 2).astype(np.int8)

    raw_feats = [
        "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
        "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "sm_x_rf", "tc_x_ws",
    ]
    cat_feats = ["Crop_Type", "Soil_Type", "Region", "Season",
                  "Mulching_Used", "Irrigation_Type", "Water_Source",
                  "Crop_Growth_Stage"]
    df = tr.loc[domain, raw_feats + cat_feats].reset_index(drop=True)
    for c in cat_feats:
        m = {v: i for i, v in enumerate(sorted(df[c].dropna().unique()))}
        df[c] = df[c].map(m).fillna(-1).astype(np.int32)
    teacher_dom = teacher[domain]
    df["teacher_PL"] = teacher_dom[:, 0].astype(np.float32)
    df["teacher_PM"] = teacher_dom[:, 1].astype(np.float32)
    df["teacher_PH"] = teacher_dom[:, 2].astype(np.float32)
    df["teacher_mh_margin"] = (df["teacher_PM"] - df["teacher_PH"]).astype(np.float32)
    df["teacher_mh_ratio"] = (np.log(np.clip(df["teacher_PH"], 1e-9, 1)) -
                                np.log(np.clip(df["teacher_PM"], 1e-9, 1))).astype(np.float32)

    ns_eval = [5, 10, 25, 50, 100, 200, 500, 1000]
    results = {}

    # Baseline v2
    log("=== baseline v2 ===")
    oof_v2 = np.load(ART / "oof_spec6_mh_v2.npy")[domain]
    results["v2"] = topk_eval("v2", oof_v2, y_3, y_bin, ns_eval)

    # Depth sweep
    log("=== XGB depth sweep ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oofs_by_depth = {}
    for depth in [2, 3, 4, 5, 6]:
        oof = np.zeros(len(y_bin), dtype=np.float32)
        for tr_idx, va_idx in skf.split(df.index, y_bin):
            dtr = xgb.DMatrix(df.iloc[tr_idx], label=y_bin[tr_idx])
            dva = xgb.DMatrix(df.iloc[va_idx], label=y_bin[va_idx])
            params = dict(objective="binary:logistic", eval_metric="auc",
                           max_depth=depth, learning_rate=0.05,
                           min_child_weight=10, subsample=0.9,
                           colsample_bytree=0.9,
                           reg_alpha=1.0, reg_lambda=1.0,
                           tree_method="hist", verbosity=0, seed=42)
            booster = xgb.train(params, dtr, num_boost_round=2000,
                                  evals=[(dva, "v")], early_stopping_rounds=100,
                                  verbose_eval=0)
            oof[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        oofs_by_depth[depth] = oof
        results[f"xgb_d{depth}"] = topk_eval(f"xgb_d{depth}", oof, y_3, y_bin, ns_eval)

    # Ensemble of v2 + best XGB + univariate teacher_PH
    log("=== ensembles ===")
    best_depth = max([d for d in oofs_by_depth.keys()],
                       key=lambda d: results[f"xgb_d{d}"]["best_macro_delta"])
    log(f"best XGB depth = {best_depth}")

    # Plain mean ensemble
    ens_avg = (oof_v2 + oofs_by_depth[best_depth] + df["teacher_PH"].to_numpy()) / 3
    results["ens_mean"] = topk_eval("ens_mean(v2+xgb+PH)", ens_avg, y_3, y_bin, ns_eval)

    # Rank ensemble (each model contributes argranks averaged)
    def rank01(x):
        return (np.argsort(np.argsort(x))).astype(np.float32) / max(len(x) - 1, 1)
    ens_rank = (rank01(oof_v2) + rank01(oofs_by_depth[best_depth]) +
                 rank01(df["teacher_PH"].to_numpy())) / 3
    results["ens_rank"] = topk_eval("ens_rank(v2+xgb+PH)", ens_rank, y_3, y_bin, ns_eval)

    # Row-level prob blend test: take primary's per-row probs and add α × spec_PH at score=6
    log("=== row-level blend onto LB-best primary (alpha sweep) ===")
    # Reconstruct LB-best primary OOF
    p3 = teacher  # the 3-way log-blend
    # The actual LB-best primary is p3 + xgb_metastack_iso α=0.30
    # For simplicity, use teacher alone as the anchor here — tests whether
    # adding spec_v2 P_H at score=6 helps on top of the well-calibrated 3-way.
    log_p3 = np.log(np.clip(p3, 1e-9, 1.0))

    # v2 P_H is already a probability; we need to convert it into a log-prob shift
    # for the H column at score=6 rows. Use logit(P_H) as the shift.
    p_h_v2 = oof_v2  # in domain
    logit_v2 = np.log(np.clip(p_h_v2, 1e-9, 1.0)) - np.log(np.clip(1 - p_h_v2, 1e-9, 1.0))

    blend_results = {}
    y_full_3 = y_full
    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]:
        log_p_blend = log_p3.copy()
        # On in-domain rows, add alpha * logit_v2 to the H column
        log_p_blend[domain, 2] += alpha * logit_v2
        # Apply LB-best fixed bias and argmax
        pred = np.argmax(log_p_blend + bias, axis=1).astype(np.int8)
        bal = fast_bal_acc(y_full_3, pred)
        blend_results[f"alpha_{alpha}"] = float(bal)
        log(f"  alpha={alpha:.3f}  global macro={bal:.6f}")

    results["row_level_blend"] = blend_results

    # Best alpha and delta vs alpha=0
    best_alpha = max(blend_results.items(), key=lambda kv: kv[1])
    delta = best_alpha[1] - blend_results["alpha_0.0"]
    log(f"best blend alpha = {best_alpha[0]} → macro={best_alpha[1]:.6f}  Δ={delta:+.6f}")

    # Final summary
    log("\n=== SUMMARY ===")
    rs = sorted([(k, v) for k, v in results.items() if "macro_delta" in v.get("topn", {}).get(v.get("best_n", "n_5"), {}) or "auc" in v],
                  key=lambda kv: -(kv[1].get("best_macro_delta", -1)))
    for k, v in rs:
        if "best_macro_delta" not in v:
            continue
        log(f"  {k:20s} AUC={v['auc']:.4f}  best_Δ={v['best_macro_delta']:+.6f}  "
            f"@{v['best_n']}")
    log(f"  row_level_blend best_alpha={best_alpha[0]} Δ={delta:+.6f}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
