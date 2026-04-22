"""Step 1/3 of the TE-continuous-regression OOF variant.

Same target shape as te_targets.py (per-class continuous TE keyed
by 5 cats x dgp_score, m=30 shrinkage to per-score prior) but
sourced from SYNTHETIC train labels instead of the 10k original.

Critically, the original variant was structurally rule-equivalent
because the 10k original is rule-perfect by construction. The
synthetic train carries the host's NN-flip signal -- so a TE
computed from synthetic labels can in principle deviate from the
rule on flip-rich cells, providing a non-rule-equivalent target.

Leak prevention (leave-one-fold-out):
  - For each row in fold k, target = TE built from folds {j != k}.
    Each row's target never depends on its own label.
  - For test rows, target = mean over the 5 fold-restricted lookups
    (each lookup excludes a different 5th of train).
  - The 5-fold split is the SAME StratifiedKFold(seed=42) used in
    every downstream OOF, so the XGB regression in step 2 sees a
    leak-free TE-target column for both its train and val rows.

Outputs:
  scripts/artifacts/te_targets_train_oof.npy  (630_000, 3)
  scripts/artifacts/te_targets_test_oof.npy   (270_000, 3)
  scripts/artifacts/te_targets_oof_meta.json  diagnostic info
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
ACTIVE_STAGES = ("Flowering", "Vegetative")

KEY_COLS = [
    "Crop_Type", "Soil_Type", "Season", "Region",
    "Crop_Growth_Stage", "dgp_score",
]
SHRINKAGE_M = 30.0

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_dgp_score(df: pd.DataFrame) -> pd.DataFrame:
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
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    return out


def build_te_from(
    df: pd.DataFrame, y: np.ndarray, m: float,
) -> tuple[dict, dict, np.ndarray]:
    """Return per-key shrunk per-class probs, per-score priors, global prior."""
    global_prior = np.bincount(y, minlength=3) / len(y)
    score_prior: dict[int, np.ndarray] = {}
    for s, sub in df.groupby("dgp_score"):
        ys = y[sub.index.values]
        if len(ys) == 0:
            score_prior[int(s)] = global_prior.copy()
        else:
            score_prior[int(s)] = (np.bincount(ys, minlength=3) / len(ys))

    lookup: dict[tuple, np.ndarray] = {}
    grouped = df.groupby(KEY_COLS, observed=True)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        ys = y[sub.index.values]
        counts = np.bincount(ys, minlength=3).astype(np.float64)
        score_int = int(key[KEY_COLS.index("dgp_score")])
        prior = score_prior.get(score_int, global_prior)
        shrunk = (counts + m * prior) / (counts.sum() + m)
        lookup[key] = shrunk
    return lookup, score_prior, global_prior


def apply_te(
    df: pd.DataFrame, lookup: dict, score_prior: dict, global_prior: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    n = len(df)
    out = np.zeros((n, 3), dtype=np.float32)
    cols = [df[c].values for c in KEY_COLS]
    keys = list(zip(*cols))
    hits = 0
    score_fallbacks = 0
    for i, key in enumerate(keys):
        v = lookup.get(key)
        if v is not None:
            out[i] = v
            hits += 1
            continue
        score_int = int(key[KEY_COLS.index("dgp_score")])
        prior = score_prior.get(score_int)
        if prior is not None:
            out[i] = prior
            score_fallbacks += 1
        else:
            out[i] = global_prior
    return out, hits, score_fallbacks


def main() -> None:
    t0 = time.time()
    log("loading synthetic train + test")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log(f"  train={len(tr)}  test={len(te)}")

    log("computing dgp_score")
    tr = add_dgp_score(tr)
    te = add_dgp_score(te)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(tr, y))

    log(f"building leave-one-fold-out TE (key={KEY_COLS}, m={SHRINKAGE_M})")
    target_train = np.zeros((len(tr), 3), dtype=np.float32)
    target_test = np.zeros((len(te), 3), dtype=np.float64)
    fold_diag = []
    for fold, (other_idx, this_idx) in enumerate(folds):
        t_f = time.time()
        # other_idx == "trainable" rows for this fold; we exclude this_idx from
        # the TE build so this_idx rows get a leak-free target.
        sub = tr.iloc[other_idx].reset_index(drop=True)
        y_sub = y[other_idx]
        lookup, score_prior, global_prior = build_te_from(sub, y_sub, SHRINKAGE_M)
        vals_this, hits_this, sf_this = apply_te(
            tr.iloc[this_idx].reset_index(drop=True),
            lookup, score_prior, global_prior,
        )
        target_train[this_idx] = vals_this
        vals_te, hits_te, sf_te = apply_te(
            te, lookup, score_prior, global_prior,
        )
        target_test += vals_te / N_FOLDS
        fold_diag.append({
            "fold": fold + 1,
            "n_other": int(len(other_idx)),
            "n_this": int(len(this_idx)),
            "n_cells_in_lookup": int(len(lookup)),
            "this_hits": int(hits_this),
            "this_score_fallbacks": int(sf_this),
            "test_hits": int(hits_te),
            "test_score_fallbacks": int(sf_te),
            "wall_s": time.time() - t_f,
        })
        log(f"  fold {fold+1}/{N_FOLDS}  cells={len(lookup):>5d}  "
            f"this_hits={hits_this:>6d}/{len(this_idx)}  "
            f"test_hits={hits_te:>6d}/{len(te)}  "
            f"({time.time()-t_f:.1f}s)")

    s_tr = target_train.sum(axis=1)
    s_te = target_test.sum(axis=1)
    log(f"target row-sum  train [{s_tr.min():.5f}, {s_tr.max():.5f}]   "
        f"test [{s_te.min():.5f}, {s_te.max():.5f}]")

    np.save(ART / "te_targets_train_oof.npy", target_train.astype(np.float32))
    np.save(ART / "te_targets_test_oof.npy", target_test.astype(np.float32))

    meta = {
        "key_cols": KEY_COLS,
        "shrinkage_m": SHRINKAGE_M,
        "source": "synthetic train, leave-one-fold-out (5-fold seed=42)",
        "fold_diag": fold_diag,
    }
    with open(ART / "te_targets_oof_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote te_targets_train_oof.npy, te_targets_test_oof.npy, "
        f"te_targets_oof_meta.json   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
