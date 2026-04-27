"""Build cuML meta-stacker input matrix and save as compressed npz.

Curated 15-component bank (matches LB-best v1 composition — pre-saturation
era). Avoids bank-extension overfit territory where 162-component pools
inflated OOF without LB transfer (see N5b family + R2 saturation log).

Output is a single npz the Kaggle GPU kernel loads to train cuML LR +
cuML RF on top, no per-component artefact upload needed.

Outputs scripts/artifacts/cuml_meta_input.npz with:
  X_tr  float16  (630_000, 17 + 15*3)
  X_te  float16  (270_000, ...)
  y     int8     (630_000,)
  fold_idx int8  (630_000,)   StratifiedKFold(5, seed=42)
  feature_names list[str]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402
from tier1b_xgb_metastack import build_lbbest_stack, log  # noqa: E402

ART = Path("scripts/artifacts")
SEED = 42
META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
             "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_boundary_dist", "min_axis_abs",
             "score_dist_low_mid", "score_dist_mid_high"]

# Curated 15-component bank — pre-saturation, LB-validated era.
# Each was either a foundation of LB-best primary or in the original v1 pool.
CURATED = [
    "recipe_full_te",
    "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler",
    "recipe_pseudolabel_seed123labeler",
    "realmlp",
    "xgb_nonrule",
    "xgb_corn",
    "xgb_dist_digits",
    "xgb_dist_routed_v3",
    "xgb_dist_digits_ote",
    "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_light",
    "recipe_full_te_catboost",
    "lgbm_te_orig",
    "lgbm_dist_digits_ote",
    "xgb_spec_678",
    "hybrid_lgbmxgb_blend",
]


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy().astype(np.int8)
    n_tr, n_te = len(train), len(test)
    log(f"train={n_tr}  test={n_te}")

    log(f"loading curated {len(CURATED)} components")
    pool = {}
    for name in CURATED:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            log(f"  SKIP {name}: missing")
            continue
        o = np.load(oof_p).astype(np.float32)
        t = np.load(test_p).astype(np.float32)
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != n_tr:
            log(f"  SKIP {name}: shape {o.shape}")
            continue
        if (o.sum(1) < 1e-3).any():
            log(f"  SKIP {name}: partial-fold zeros")
            continue
        pool[name] = (_normed(o), _normed(t))
        log(f"  + {name}")
    log(f"  loaded {len(pool)}/{len(CURATED)} components")

    log("building LB-best 3-stack anchor")
    lb_oof, lb_test = build_lbbest_stack(y.astype(np.int32))

    log("constructing distance / rule meta features")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = te_d[META_COLS].to_numpy(dtype=np.float32)

    component_names = sorted(pool.keys())
    feature_names = ["lb_logL", "lb_logM", "lb_logH"] + META_COLS
    for n in component_names:
        feature_names += [f"{n}_logL", f"{n}_logM", f"{n}_logH"]

    log(f"meta-feature shape: ({n_tr}, {len(feature_names)})")
    lb_log_tr = np.log(np.clip(lb_oof, 1e-9, 1.0))
    lb_log_te = np.log(np.clip(lb_test, 1e-9, 1.0))
    comp_tr = [np.log(np.clip(pool[n][0], 1e-9, 1.0)) for n in component_names]
    comp_te = [np.log(np.clip(pool[n][1], 1e-9, 1.0)) for n in component_names]

    X_tr = np.concatenate([lb_log_tr, meta_tr] + comp_tr, axis=1).astype(np.float16)
    X_te = np.concatenate([lb_log_te, meta_te] + comp_te, axis=1).astype(np.float16)
    log(f"X_tr {X_tr.shape} ({X_tr.nbytes/1e6:.1f} MB), X_te {X_te.shape} ({X_te.nbytes/1e6:.1f} MB)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_idx = np.zeros(n_tr, dtype=np.int8)
    for f, (_, va) in enumerate(skf.split(np.zeros(n_tr), y), 1):
        fold_idx[va] = f

    out_path = ART / "cuml_meta_input.npz"
    np.savez_compressed(
        out_path,
        X_tr=X_tr, X_te=X_te, y=y, fold_idx=fold_idx,
        feature_names=np.array(feature_names),
    )
    sz = out_path.stat().st_size / 1e6
    log(f"wrote {out_path}  ({sz:.1f} MB compressed)")


if __name__ == "__main__":
    main()
