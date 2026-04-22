"""Sub-cell TE target variant (follow-up B of the TE-regression theorem).

Orig- and oof-variant TE both bucketed labels at rule-cell granularity
(5 cats x dgp_score) -- where the rule's own decision is invariant by
construction. That makes the soft-prob target argmax-equivalent to
the rule REGARDLESS of source (CLAUDE.md theorem).

Sub-cell key subdivides each rule-cell by TWO additional axes chosen
to maximise within-cell flip-direction signal (per 2026-04-21 DGP
residuals EDA):
  - Humidity bin (5 quantile bins; Cohen's d +0.076 on flips)
  - Crop_Type          (6 values; proxy for NN input axis)

The rule-cell axis itself is encoded as the 6-dim binary tuple
(dry, norain, hot, windy, nomulch, kc_active) -> 2^5 x 2 = 64 unique
rule-cells. Combined:
  64 (rule-cell) x 5 (humidity_bin) x 6 (crop_type) = 1920 sub-cells
On LOFO 504k rows per fold that's ~262 rows/sub-cell, dense enough
for m=15 Bayesian shrinkage to per-rule-cell prior.

If ANY sub-cell has a synthetic majority different from its parent
rule-cell's majority, the soft-prob target will deviate from the rule
at argmax -> a non-rule-equivalent regression target.

Outputs:
  scripts/artifacts/te_targets_train_subcell.npy  (630_000, 3)
  scripts/artifacts/te_targets_test_subcell.npy   (270_000, 3)
  scripts/artifacts/te_targets_subcell_meta.json  diagnostic info
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

N_HUMIDITY_BINS = 5
SHRINKAGE_M = 15.0

ART = Path("scripts/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_cell_features(df: pd.DataFrame) -> pd.DataFrame:
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
    kc_active = np.isin(stage_str, ACTIVE_STAGES).astype(np.int8)
    # 6-bit rule-cell id in [0, 64).
    rule_cell = (
        (dry.astype(np.int32) << 5)
        | (norain.astype(np.int32) << 4)
        | (hot.astype(np.int32) << 3)
        | (windy.astype(np.int32) << 2)
        | (nomulch.astype(np.int32) << 1)
        | (kc_active.astype(np.int32))
    ).astype(np.int32)
    out["rule_cell"] = rule_cell
    kc = (2 * kc_active).astype(np.int8)
    out["dgp_score"] = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    return out


def add_humidity_bin(
    df_all: list[pd.DataFrame], n_bins: int,
) -> tuple[list[pd.DataFrame], np.ndarray]:
    """Fit quantile bin edges on the concatenation and apply to all."""
    concat = pd.concat([d["Humidity"] for d in df_all], ignore_index=True)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(concat.values, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    # Collapse duplicate edges.
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        log(f"  humidity bin edges degenerate ({len(edges)-1}<{n_bins}) - keeping "
            f"{len(edges)-1} bins")
    out = []
    for d in df_all:
        b = np.digitize(d["Humidity"].values, edges[1:-1], right=False).astype(np.int8)
        d2 = d.copy()
        d2["humidity_bin"] = b
        out.append(d2)
    return out, edges


def build_te_from(
    df: pd.DataFrame, y: np.ndarray, key_cols: list[str], m: float,
) -> tuple[dict, dict, np.ndarray]:
    """Return shrunk per-class probs per key tuple; shrinkage toward rule_cell prior."""
    global_prior = np.bincount(y, minlength=3) / len(y)
    # Per-rule-cell prior (coarser than the full key).
    rc_prior: dict[int, np.ndarray] = {}
    for rc, sub in df.groupby("rule_cell"):
        ys = y[sub.index.values]
        rc_prior[int(rc)] = (np.bincount(ys, minlength=3) / len(ys)
                             if len(ys) else global_prior.copy())

    lookup: dict[tuple, np.ndarray] = {}
    grouped = df.groupby(key_cols, observed=True)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        ys = y[sub.index.values]
        counts = np.bincount(ys, minlength=3).astype(np.float64)
        rc = int(key[key_cols.index("rule_cell")])
        prior = rc_prior.get(rc, global_prior)
        shrunk = (counts + m * prior) / (counts.sum() + m)
        lookup[key] = shrunk
    return lookup, rc_prior, global_prior


def apply_te(
    df: pd.DataFrame, key_cols: list[str], lookup: dict,
    rc_prior: dict, global_prior: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    n = len(df)
    out = np.zeros((n, 3), dtype=np.float32)
    keys = list(zip(*[df[c].values for c in key_cols]))
    hits = 0
    rc_fallbacks = 0
    rc_idx = key_cols.index("rule_cell")
    for i, key in enumerate(keys):
        v = lookup.get(key)
        if v is not None:
            out[i] = v
            hits += 1
            continue
        rc = int(key[rc_idx])
        prior = rc_prior.get(rc)
        if prior is not None:
            out[i] = prior
            rc_fallbacks += 1
        else:
            out[i] = global_prior
    return out, hits, rc_fallbacks


def main() -> None:
    t0 = time.time()
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    log(f"  train={len(tr)}  test={len(te)}")

    log("adding rule_cell + dgp_score")
    tr = add_cell_features(tr)
    te = add_cell_features(te)

    log(f"fitting {N_HUMIDITY_BINS}-quantile humidity bin on union of train+test")
    [tr, te], edges = add_humidity_bin([tr, te], N_HUMIDITY_BINS)
    log(f"  humidity bin edges = {edges.round(3).tolist()}")

    # Report sub-cell diagnostic on full train (before LOFO).
    key_cols = ["rule_cell", "humidity_bin", "Crop_Type"]
    y_all = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    # Check: how many sub-cells have a majority class different from their rule-cell's?
    rc_major: dict[int, int] = {}
    for rc, sub in tr.groupby("rule_cell"):
        y_rc = y_all[sub.index.values]
        rc_major[int(rc)] = int(np.bincount(y_rc, minlength=3).argmax())
    n_sub_differ = 0
    n_sub_total = 0
    n_rows_in_differ = 0
    for key, sub in tr.groupby(key_cols, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        y_k = y_all[sub.index.values]
        sub_major = int(np.bincount(y_k, minlength=3).argmax())
        n_sub_total += 1
        if sub_major != rc_major.get(int(key[0]), -1):
            n_sub_differ += 1
            n_rows_in_differ += len(sub)
    log(f"  sub-cell majority vs rule-cell majority (on full 630k): "
        f"{n_sub_differ}/{n_sub_total} differ  ({n_rows_in_differ:,} rows, "
        f"{100*n_rows_in_differ/len(tr):.3f}% of train)")

    log("building leave-one-fold-out TE (5-fold seed=42)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(tr, y_all))

    target_train = np.zeros((len(tr), 3), dtype=np.float32)
    target_test = np.zeros((len(te), 3), dtype=np.float64)
    fold_diag = []
    for fold, (other_idx, this_idx) in enumerate(folds):
        t_f = time.time()
        sub = tr.iloc[other_idx].reset_index(drop=True)
        y_sub = y_all[other_idx]
        lookup, rc_prior, global_prior = build_te_from(
            sub, y_sub, key_cols, SHRINKAGE_M,
        )
        vals_this, hits_this, fb_this = apply_te(
            tr.iloc[this_idx].reset_index(drop=True),
            key_cols, lookup, rc_prior, global_prior,
        )
        target_train[this_idx] = vals_this
        vals_te, hits_te, fb_te = apply_te(
            te, key_cols, lookup, rc_prior, global_prior,
        )
        target_test += vals_te / N_FOLDS
        fold_diag.append({
            "fold": fold + 1,
            "n_cells_in_lookup": int(len(lookup)),
            "this_hits": int(hits_this),
            "this_rc_fallbacks": int(fb_this),
            "test_hits": int(hits_te),
            "test_rc_fallbacks": int(fb_te),
            "wall_s": time.time() - t_f,
        })
        log(f"  fold {fold+1}/{N_FOLDS}  cells={len(lookup)}  "
            f"this_hits={hits_this}/{len(this_idx)}  "
            f"test_hits={hits_te}/{len(te)}  ({time.time()-t_f:.1f}s)")

    np.save(ART / "te_targets_train_subcell.npy", target_train.astype(np.float32))
    np.save(ART / "te_targets_test_subcell.npy", target_test.astype(np.float32))

    meta = {
        "key_cols": key_cols,
        "n_humidity_bins": N_HUMIDITY_BINS,
        "humidity_edges": edges.tolist(),
        "shrinkage_m": SHRINKAGE_M,
        "source": "synthetic train, leave-one-fold-out (5-fold seed=42)",
        "sub_cell_majority_differs_from_rule_cell": {
            "count": n_sub_differ,
            "total_sub_cells": n_sub_total,
            "rows_in_differing_sub_cells": n_rows_in_differ,
            "frac_of_train": n_rows_in_differ / len(tr),
        },
        "fold_diag": fold_diag,
    }
    with open(ART / "te_targets_subcell_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote te_targets_{{train,test}}_subcell.npy, te_targets_subcell_meta.json   "
        f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
