"""Stage 1: feature importance diagnostic on N1-style meta-stacker.

Trains a SINGLE booster on full data with same HPs as N1 (lam_ce=0.3,
170-component bank, depth=4 + heavy-reg). Extracts gain importance per
feature and maps back to component names.

NOT for prediction — diagnostic only. ~5-10 min wall.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX
from recipe_macrorecall import make_macrorec_obj, macrorec_eval_metric
from tier1b_xgb_metastack import build_lbbest_stack, load_pool

ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
TARGET = "Irrigation_Need"

t0 = time.time()
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

print(f"[{time.strftime('%H:%M:%S')}] building meta features")
lb_oof, lb_test = build_lbbest_stack(y)
pool = load_pool(y)
component_names = sorted(pool.keys())
print(f"  pool: {len(component_names)} components")

tr_d = add_distance_features(train)
meta_cols = ["dgp_score", "rule_pred",
             "sm_dist", "rf_dist", "tc_dist", "ws_dist",
             "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]
meta_tr = tr_d[meta_cols].to_numpy(dtype=np.float32)
comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
X = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float32)
print(f"  X shape: {X.shape}")

# Build feature names so XGB importance can map back
feat_names = (
    [f"lb3_logp_{c}" for c in ('L', 'M', 'H')] +
    meta_cols +
    [f"{n}_logp_{c}" for n in component_names for c in ('L', 'M', 'H')]
)
assert len(feat_names) == X.shape[1], (len(feat_names), X.shape[1])

print(f"[{time.strftime('%H:%M:%S')}] training single booster (full data, no CV) with macrorec obj lam_ce=0.3")
xgb_params = dict(
    num_class=3, tree_method="hist",
    learning_rate=0.05, max_depth=4,
    min_child_weight=5, subsample=0.9, colsample_bytree=0.9,
    reg_alpha=5.0, reg_lambda=5.0,
    verbosity=0, seed=SEED, nthread=-1,
    disable_default_eval_metric=1,
)
dtr = xgb.DMatrix(X, label=y, feature_names=feat_names)
obj = make_macrorec_obj(y, n_classes=3, temperature=1.0, lam_ce=0.3)
booster = xgb.train(
    xgb_params, dtr, num_boost_round=300,  # cap for diagnostic (don't need full training)
    obj=obj,
    evals=[(dtr, "train")], verbose_eval=100,
)
print(f"  trained in {time.time() - t0:.0f}s")

# Extract gain importance per feature
gain = booster.get_score(importance_type="gain")
print(f"\nFeatures with non-zero gain: {len(gain)} / {len(feat_names)}")

# Aggregate by component (sum gain across L/M/H)
comp_gain: dict[str, float] = {}
for fname, g in gain.items():
    if fname in ("dgp_score", "rule_pred") or fname in meta_cols:
        comp_gain[f"_meta_{fname}"] = comp_gain.get(f"_meta_{fname}", 0.0) + g
        continue
    if fname.startswith("lb3_logp_"):
        comp_gain["_lb3_anchor"] = comp_gain.get("_lb3_anchor", 0.0) + g
        continue
    if fname.endswith("_logp_L") or fname.endswith("_logp_M") or fname.endswith("_logp_H"):
        comp = fname.rsplit("_logp_", 1)[0]
        comp_gain[comp] = comp_gain.get(comp, 0.0) + g

# Print ranked
sorted_comp = sorted(comp_gain.items(), key=lambda x: -x[1])
print(f"\nTop 50 components by aggregated gain (across 3 class outputs):")
print(f"  {'rank':>4}  {'component':50s}  {'gain':>10}  {'cumulative %':>12}")
total_g = sum(comp_gain.values())
cum = 0.0
for rank, (name, g) in enumerate(sorted_comp[:50], 1):
    cum += g
    print(f"  {rank:>4}  {name:50s}  {g:>10.2f}  {100*cum/total_g:>11.2f}%")

print(f"\nTotal gain: {total_g:.0f}")
print(f"\nComponents accounting for top 80% of gain:")
cum = 0.0
top80 = []
for name, g in sorted_comp:
    cum += g
    top80.append(name)
    if cum / total_g >= 0.80:
        break
print(f"  {len(top80)} components account for 80% of gain")
print(f"  → top-{len(top80)} curated pool would capture ~80% of macrorec meta's signal")

# Save the ranked list
out = {
    "n_features_total": len(feat_names),
    "n_features_used": len(gain),
    "total_gain": float(total_g),
    "top_50_components": [
        {"rank": i+1, "name": n, "gain": float(g), "share_pct": float(100*g/total_g)}
        for i, (n, g) in enumerate(sorted_comp[:50])
    ],
    "top_80pct_components": top80,
    "n_top_80pct": len(top80),
}
with open(ART / "feature_importance_macrorec.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote scripts/artifacts/feature_importance_macrorec.json")
