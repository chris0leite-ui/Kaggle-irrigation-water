"""Stage 1e: redo override analysis against the actual LB-best PRIMARY (4-stack).

v2 specialist was built against the LB-best 3-way teacher. The actual primary
is 4-stack: 3-way + xgb_metastack_iso α=0.30. Override decisions in deployment
operate on the primary, not on the 3-way. If the meta-stacker already corrects
some of v2's top-K candidates, our usable override count and macro-delta drop
further.

Two scenarios:
  (A) v2 vs primary on the OOF — "would v2 disagree with primary?" Pre-screen
      to find rows where they disagree, then estimate macro-delta.
  (B) build a fresh specialist (depth=2 XGB which is the stage1d top non-v2
      candidate by AUC) targeting score=6 ∩ primary_pred=Medium directly.
      Compare ceiling to v2-against-3way.
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
from common import CLS2IDX, add_distance_features, log_blend, fast_bal_acc  # noqa: E402

ART = Path("scripts/artifacts")
OUT = ART / "score6_manifold_stage1e_results.json"

N_H_TOTAL = 21009
N_M_TOTAL = 239074


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro_delta_local(y_3, override_idx):
    c = int((y_3[override_idx] == 2).sum())
    w_m = int((y_3[override_idx] == 1).sum())
    return (c / N_H_TOTAL - w_m / N_M_TOTAL) / 3.0


def topk_eval(name, score, y_3, y_bin, ns):
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
    log(f"  {name:30s} AUC={auc:.4f}  best_n={bn[0]}  c={bn[1]['correct']:>3d}  "
        f"prec={bn[1]['prec']:.3f}  macro_Δ={bn[1]['macro_delta']:+.6f}")
    return out


def main() -> None:
    log("loading train")
    tr = pd.read_csv("data/train.csv")
    tr = add_distance_features(tr)

    y_full = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)
    score = tr["dgp_score"].to_numpy().astype(np.int8)

    log("rebuilding LB-best 4-stack PRIMARY")
    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    teacher_3w = log_blend([oof_r, oof_s1, oof_s7], np.array([0.25, 0.35, 0.40]))

    # 4-stack = teacher_3w blended with xgb_metastack_iso at α=0.30
    # Need iso-cal'd metastack. Look for it on disk.
    meta_path = ART / "oof_xgb_metastack.npy"
    if not meta_path.exists():
        log(f"  WARN: {meta_path} not found, using 3-way only as proxy")
        primary_oof = teacher_3w
    else:
        oof_meta = np.load(meta_path)
        # Apply iso-cal per-fold using saved iso (or recompute)
        # We don't have iso fits saved. Approximate by full-OOF iso.
        from sklearn.isotonic import IsotonicRegression
        oof_meta_iso = np.zeros_like(oof_meta)
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(oof_meta[:, k], (y_full == k).astype(np.float32))
            oof_meta_iso[:, k] = ir.transform(oof_meta[:, k])
        oof_meta_iso /= np.maximum(oof_meta_iso.sum(axis=1, keepdims=True), 1e-9)
        primary_oof = log_blend([teacher_3w, oof_meta_iso], np.array([0.70, 0.30]))

    bias = np.array([1.4324, 1.4689, 3.4008])
    primary_pred = np.argmax(np.log(np.clip(primary_oof, 1e-9, 1.0)) + bias, axis=1).astype(np.int8)
    primary_macro = fast_bal_acc(y_full, primary_pred)
    log(f"primary OOF macro: {primary_macro:.6f} (target: ~0.98084)")

    # Domain against PRIMARY
    domain_p = (score == 6) & (primary_pred == 1)
    n_dp = int(domain_p.sum())
    h_dp = int(((y_full == 2) & domain_p).sum())
    log(f"primary-residual override domain: n={n_dp:,}  H={h_dp}  "
        f"prevalence={h_dp/n_dp:.4f}")

    # Domain against 3-way (what v2 was trained for)
    bias_3w = bias  # same bias
    teacher_3w_pred = np.argmax(np.log(np.clip(teacher_3w, 1e-9, 1.0)) + bias_3w, axis=1)
    domain_3w = (score == 6) & (teacher_3w_pred == 1)
    n_d3 = int(domain_3w.sum())
    h_d3 = int(((y_full == 2) & domain_3w).sum())
    log(f"3-way-residual override domain: n={n_d3:,}  H={h_d3}")

    # Diff: rows in 3-way domain but NOT in primary domain (= caught by meta-stacker)
    caught = domain_3w & ~domain_p
    caught_h = int(((y_full == 2) & caught).sum())
    log(f"  primary catches {int(caught.sum())} rows the 3-way left as M  "
        f"({caught_h} truly-H)")

    # New rows in primary domain but NOT in 3-way (meta-stacker introduced new errors?)
    new_in_p = domain_p & ~domain_3w
    new_in_p_h = int(((y_full == 2) & new_in_p).sum())
    log(f"  primary introduced {int(new_in_p.sum())} new M-pred at score=6  "
        f"({new_in_p_h} truly-H)")

    # Evaluate v2 against PRIMARY domain
    log("=== v2 specialist evaluated on PRIMARY-residual domain ===")
    y_3p = y_full[domain_p]
    y_bin_p = (y_3p == 2).astype(np.int8)
    oof_v2_p = np.load(ART / "oof_spec6_mh_v2.npy")[domain_p]
    ns_eval = [5, 10, 25, 50, 100, 200]
    r_v2_against_p = topk_eval("v2_vs_primary", oof_v2_p, y_3p, y_bin_p, ns_eval)

    # Build a fresh primary-aligned specialist
    log("=== fresh specialist trained against PRIMARY ===")
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
    df = tr.loc[domain_p, raw_feats + cat_feats].reset_index(drop=True)
    for c in cat_feats:
        m = {v: i for i, v in enumerate(sorted(df[c].dropna().unique()))}
        df[c] = df[c].map(m).fillna(-1).astype(np.int32)
    primary_dom = primary_oof[domain_p]
    df["primary_PL"] = primary_dom[:, 0].astype(np.float32)
    df["primary_PM"] = primary_dom[:, 1].astype(np.float32)
    df["primary_PH"] = primary_dom[:, 2].astype(np.float32)
    df["primary_mh_margin"] = (df["primary_PM"] - df["primary_PH"]).astype(np.float32)
    df["primary_mh_ratio"] = (np.log(np.clip(df["primary_PH"], 1e-9, 1)) -
                                np.log(np.clip(df["primary_PM"], 1e-9, 1))).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oofs = {}
    for depth, mcw, lr_, n_round in [(2, 10, 0.05, 2000), (4, 20, 0.05, 1500),
                                        (6, 5, 0.05, 3000)]:
        oof = np.zeros(len(y_bin_p), dtype=np.float32)
        for tr_idx, va_idx in skf.split(df.index, y_bin_p):
            dtr = xgb.DMatrix(df.iloc[tr_idx], label=y_bin_p[tr_idx])
            dva = xgb.DMatrix(df.iloc[va_idx], label=y_bin_p[va_idx])
            params = dict(objective="binary:logistic", eval_metric="auc",
                           max_depth=depth, learning_rate=lr_,
                           min_child_weight=mcw, subsample=0.9,
                           colsample_bytree=0.9,
                           reg_alpha=1.0, reg_lambda=1.0,
                           tree_method="hist", verbosity=0, seed=42)
            booster = xgb.train(params, dtr, num_boost_round=n_round,
                                  evals=[(dva, "v")], early_stopping_rounds=100,
                                  verbose_eval=0)
            oof[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        oofs[f"d{depth}"] = oof
        topk_eval(f"primary_spec_d{depth}", oof, y_3p, y_bin_p, ns_eval)

    # Save
    out = {
        "primary_macro_oof": float(primary_macro),
        "domain_primary": {"n": n_dp, "H": h_dp},
        "domain_3way": {"n": n_d3, "H": h_d3},
        "primary_caught": int(caught.sum()),
        "primary_caught_h": caught_h,
        "primary_introduced": int(new_in_p.sum()),
        "primary_introduced_h": new_in_p_h,
        "v2_vs_primary": r_v2_against_p,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
