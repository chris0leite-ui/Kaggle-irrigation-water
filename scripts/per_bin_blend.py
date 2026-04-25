"""Per-score-bin log-blend on the LB-best 3-way components.

Hypothesis: errors concentrate at dgp_score ∈ {3, 6} (74% of error mass).
The global log-blend weights (0.25 recipe + 0.35 pseudo_s1 + 0.40 pseudo_s7)
may be a cross-bin compromise; per-bin weights could improve locally without
hurting globally.

Honest nested CV:
  outer 5-fold (StratifiedKFold seed=42, aligned with saved OOFs)
    for each outer val fold k:
      fit_idx  = rows NOT in fold k
      eval_idx = rows in fold k
      for each score bin b:
        search weights (w1, w2, w3) on simplex (step=0.05, log-loss objective
        on fit_idx ∩ (score == b)) → pick best
        apply chosen weights to eval_idx ∩ (score == b)
  concat eval-fold predictions → honest out-of-fold OOF
  tune log-bias on concat OOF (diagnostic: both fixed and tuned)

Fixed-bias report is the LB-transfer proxy (per 2026-04-21 binhigh rule:
never retune bias when adding a component to a tuned stack).

Outputs:
  scripts/artifacts/per_bin_blend_results.json
  scripts/artifacts/oof_per_bin_blend.npy
  scripts/artifacts/test_per_bin_blend.npy
  submissions/submission_per_bin_blend.csv  (only if gate passes)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import (
    CLS2IDX,
    IDX2CLS,
    N_FOLDS,
    SEED,
    add_distance_features,
    fast_bal_acc,
    load_oof_pair,
    log_blend,
    tune_log_bias,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
LB_BEST_BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float64)
LB_BEST_WEIGHTS = np.array([0.25, 0.35, 0.40])
COMPONENTS = ("recipe_full_te", "recipe_pseudolabel", "recipe_pseudolabel_seed7labeler")
# Bin definition: concentrate search where errors live.
# Score 0-2 → bin 0 (clean Low). 3 → bin 1 (Low/Med boundary).
# 4-5 → bin 2 (clean Med). 6 → bin 3 (Med/High boundary).
# 7-9 → bin 4 (mostly High + some flips).
BIN_MAP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4}
BIN_NAMES = {0: "score_0-2", 1: "score_3", 2: "score_4-5", 3: "score_6", 4: "score_7-9"}


def simplex_grid(step: float = 0.05) -> np.ndarray:
    """All (w1, w2, w3) on the simplex (w_i in [0,1], sum = 1) at given step."""
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return np.asarray(pts, dtype=np.float64)


def log_loss_rows(oofs: list[np.ndarray], weights: np.ndarray,
                  y: np.ndarray, idx: np.ndarray) -> float:
    """Negative log-likelihood on selected rows (proper scoring, bin-safe)."""
    blend = log_blend([o[idx] for o in oofs], weights)
    return -np.log(np.clip(blend[np.arange(len(idx)), y[idx]], 1e-12, 1.0)).mean()


def build_prediction(
    oofs: list[np.ndarray], score: np.ndarray, idx: np.ndarray,
    weights_per_bin: dict[int, np.ndarray],
) -> np.ndarray:
    """Blend oofs[idx] with per-bin weights → (len(idx), 3) prob array."""
    score_idx = score[idx]
    bin_idx = np.array([BIN_MAP[s] for s in score_idx])
    out = np.zeros((len(idx), 3), dtype=np.float64)
    for b in sorted(BIN_NAMES):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows = idx[mask]
        out[mask] = log_blend([o[rows] for o in oofs], weights_per_bin[b])
    return out


def fit_per_bin_weights(
    oofs: list[np.ndarray], y: np.ndarray, score: np.ndarray,
    fit_idx: np.ndarray, grid: np.ndarray, objective: str = "bal_acc_global",
) -> dict[int, np.ndarray]:
    """Pick per-bin weights maximising chosen objective on fit_idx.

    - `log_loss`: per-bin NLL minimisation (independent bin objective, proper
      scoring but misaligned with macro-recall).
    - `bal_acc_global`: coordinate-descent. Start from LB-best global weights
      for every bin; for each bin in turn, sweep its weights on the simplex
      holding others fixed and pick the grid point that maximises fixed-bias
      global bal_acc on fit_idx. Iterate until no bin improves.
    """
    score_fit_all = score[fit_idx]
    bin_fit = np.array([BIN_MAP[s] for s in score_fit_all])

    out: dict[int, np.ndarray] = {b: LB_BEST_WEIGHTS.copy() for b in sorted(BIN_NAMES)}

    if objective == "log_loss":
        for b in sorted(BIN_NAMES):
            in_bin = fit_idx[bin_fit == b]
            if len(in_bin) < 100:
                continue
            best_loss = np.inf
            for w in grid:
                loss = log_loss_rows(oofs, w, y, in_bin)
                if loss < best_loss:
                    best_loss = loss
                    out[b] = w
        return out

    # bal_acc_global: coord descent. Keep a live (len(fit_idx), 3) prob array
    # and ONLY recompute the slice for the bin under test each trial — ~5x
    # speedup vs recomputing all 5 bins per trial.
    fit_y = y[fit_idx]
    cc = np.bincount(fit_y, minlength=3)
    # Pre-slice oof data to fit rows once (avoids re-indexing in each trial).
    oofs_fit = [o[fit_idx] for o in oofs]  # list of (|fit|, 3)
    # Per-bin masks into the LOCAL fit_idx space (not global index).
    bin_local_masks = {b: (bin_fit == b) for b in sorted(BIN_NAMES)}

    # Initial full prediction.
    fit_pred = np.zeros((len(fit_idx), 3), dtype=np.float64)
    for b in sorted(BIN_NAMES):
        mask = bin_local_masks[b]
        if not mask.any():
            continue
        fit_pred[mask] = log_blend([o[mask] for o in oofs_fit], out[b])

    def bal_from_pred(pred: np.ndarray) -> float:
        argm = (np.log(np.clip(pred, 1e-9, 1)) + LB_BEST_BIAS).argmax(1)
        return fast_bal_acc(fit_y, argm, class_counts=cc)

    best = bal_from_pred(fit_pred)
    import time as _t
    tag = os.environ.get("FIT_TAG", "fit")
    for it in range(4):
        improved = False
        for b in sorted(BIN_NAMES):
            mask = bin_local_masks[b]
            if mask.sum() < 100:
                continue
            t0 = _t.time()
            bin_oofs = [o[mask] for o in oofs_fit]
            saved_slice = fit_pred[mask].copy()
            current_w = out[b].copy()
            local_best = best
            local_best_w = current_w
            for w in grid:
                if np.allclose(w, current_w):
                    continue
                fit_pred[mask] = log_blend(bin_oofs, w)
                s = bal_from_pred(fit_pred)
                if s > local_best + 1e-6:
                    local_best = s
                    local_best_w = w.copy()
            # Commit best found for this bin.
            if not np.allclose(local_best_w, current_w):
                fit_pred[mask] = log_blend(bin_oofs, local_best_w)
                out[b] = local_best_w
                best = local_best
                improved = True
            else:
                fit_pred[mask] = saved_slice
            print(f"    [{tag}] it={it} bin={BIN_NAMES[b]} best={best:.5f} "
                  f"w=({local_best_w[0]:.2f},{local_best_w[1]:.2f},{local_best_w[2]:.2f}) "
                  f"dt={_t.time()-t0:.1f}s",
                  flush=True)
        if not improved:
            break
    return out


# `apply_per_bin_weights` was renamed to `build_prediction` earlier in this file.
apply_per_bin_weights = build_prediction


def main() -> None:
    print("Loading OOFs...")
    oofs, tests = [], []
    for c in COMPONENTS:
        oof, test = load_oof_pair(c)
        oofs.append(oof)
        tests.append(test)
        print(f"  {c}: oof={oof.shape}, test={test.shape}")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    y = train_df["Irrigation_Need"].map(CLS2IDX).to_numpy()
    score_train = add_distance_features(train_df)["dgp_score"].to_numpy()
    score_test = add_distance_features(test_df)["dgp_score"].to_numpy()

    # Global baseline: LB-best 3-way at fixed weights + fixed bias.
    baseline_oof = log_blend(oofs, LB_BEST_WEIGHTS)
    baseline_pred_fixed = (np.log(np.clip(baseline_oof, 1e-9, 1)) + LB_BEST_BIAS).argmax(1)
    baseline_fixed_bal = fast_bal_acc(y, baseline_pred_fixed)
    b_bias, baseline_tuned_bal = tune_log_bias(
        baseline_oof, y, prior=np.bincount(y, minlength=3) / len(y)
    )
    print(f"\nBaseline 3-way (LB-best):")
    print(f"  fixed-bias OOF bal_acc = {baseline_fixed_bal:.5f}  (bias={LB_BEST_BIAS.tolist()})")
    print(f"  tuned OOF bal_acc      = {baseline_tuned_bal:.5f}  (bias={b_bias.tolist()})")

    objective = os.environ.get("OBJECTIVE", "bal_acc_global")
    step = float(os.environ.get("GRID_STEP", "0.05"))
    grid = simplex_grid(step)
    print(f"\nSearch grid: {len(grid)} points on simplex (step={step})")
    print(f"Objective: {objective}")

    # In-sample per-bin fit (optimistic upper bound).
    all_idx = np.arange(len(y))
    os.environ["FIT_TAG"] = "in-sample"
    in_sample_weights = fit_per_bin_weights(oofs, y, score_train, all_idx, grid, objective)
    in_sample_oof = apply_per_bin_weights(oofs, score_train, all_idx, in_sample_weights)
    in_sample_fixed = (np.log(np.clip(in_sample_oof, 1e-9, 1)) + LB_BEST_BIAS).argmax(1)
    in_sample_fixed_bal = fast_bal_acc(y, in_sample_fixed)
    _, in_sample_tuned_bal = tune_log_bias(
        in_sample_oof, y, prior=np.bincount(y, minlength=3) / len(y)
    )
    print(f"\nIn-sample (optimistic) per-bin:")
    print(f"  fixed-bias: {in_sample_fixed_bal:.5f}  Δ={in_sample_fixed_bal - baseline_fixed_bal:+.5f}")
    print(f"  tuned:      {in_sample_tuned_bal:.5f}  Δ={in_sample_tuned_bal - baseline_tuned_bal:+.5f}")
    for b in sorted(BIN_NAMES):
        w = in_sample_weights[b]
        print(f"    {BIN_NAMES[b]}: w=({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})")

    # Nested CV (honest).
    print(f"\nNested {N_FOLDS}-fold CV...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    nested_oof = np.zeros_like(baseline_oof)
    nested_weights_per_fold: list[dict[int, list[float]]] = []
    for k, (fit_idx, eval_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        os.environ["FIT_TAG"] = f"fold{k + 1}"
        w_per_bin = fit_per_bin_weights(oofs, y, score_train, fit_idx, grid, objective)
        nested_oof[eval_idx] = apply_per_bin_weights(
            oofs, score_train, eval_idx, w_per_bin
        )
        nested_weights_per_fold.append({b: w_per_bin[b].tolist() for b in sorted(BIN_NAMES)})
        print(f"  fold {k + 1}/{N_FOLDS}: fit={len(fit_idx)}, eval={len(eval_idx)}")

    nested_fixed = (np.log(np.clip(nested_oof, 1e-9, 1)) + LB_BEST_BIAS).argmax(1)
    nested_fixed_bal = fast_bal_acc(y, nested_fixed)
    n_bias, nested_tuned_bal = tune_log_bias(
        nested_oof, y, prior=np.bincount(y, minlength=3) / len(y)
    )
    print(f"\nNested CV (honest):")
    print(f"  fixed-bias: {nested_fixed_bal:.5f}  Δ={nested_fixed_bal - baseline_fixed_bal:+.5f}")
    print(f"  tuned:      {nested_tuned_bal:.5f}  Δ={nested_tuned_bal - baseline_tuned_bal:+.5f}")

    # Test-side prediction: fit weights on ALL train, apply to test.
    test_idx = np.arange(len(score_test))
    test_blend = apply_per_bin_weights(tests, score_test, test_idx, in_sample_weights)

    # Persist artefacts.
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "oof_per_bin_blend.npy", nested_oof.astype(np.float32))
    np.save(ART / "test_per_bin_blend.npy", test_blend.astype(np.float32))
    results = {
        "baseline_fixed_bal": float(baseline_fixed_bal),
        "baseline_tuned_bal": float(baseline_tuned_bal),
        "in_sample_fixed_bal": float(in_sample_fixed_bal),
        "in_sample_tuned_bal": float(in_sample_tuned_bal),
        "nested_fixed_bal": float(nested_fixed_bal),
        "nested_tuned_bal": float(nested_tuned_bal),
        "overfit_gap_fixed": float(in_sample_fixed_bal - nested_fixed_bal),
        "overfit_gap_tuned": float(in_sample_tuned_bal - nested_tuned_bal),
        "in_sample_weights": {BIN_NAMES[b]: in_sample_weights[b].tolist() for b in sorted(BIN_NAMES)},
        "nested_weights_per_fold": [
            {BIN_NAMES[b]: w[b] for b in sorted(BIN_NAMES)} for w in nested_weights_per_fold
        ],
        "baseline_weights": LB_BEST_WEIGHTS.tolist(),
        "baseline_bias": LB_BEST_BIAS.tolist(),
        "components": list(COMPONENTS),
    }
    with open(ART / "per_bin_blend_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {ART}/per_bin_blend_results.json")

    # Emit submission if the HONEST fixed-bias gate passes +0.0002.
    gate = nested_fixed_bal - baseline_fixed_bal
    if gate >= 2e-4:
        test_ids = test_df["id"].to_numpy()
        test_pred = (np.log(np.clip(test_blend, 1e-9, 1)) + LB_BEST_BIAS).argmax(1)
        sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": [IDX2CLS[i] for i in test_pred]})
        SUB.mkdir(parents=True, exist_ok=True)
        path = SUB / "submission_per_bin_blend.csv"
        sub.to_csv(path, index=False)
        print(f"\n✅ Gate PASS (Δ={gate:+.5f}) — emitted {path}")
    else:
        print(f"\n⚠️  Gate MISS (Δ={gate:+.5f} < +0.0002) — no submission emitted")


if __name__ == "__main__":
    main()
