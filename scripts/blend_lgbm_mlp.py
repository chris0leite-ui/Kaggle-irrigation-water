"""Probability-level blend of LGBM+DGP and MLP+BalSoft.

Hypothesis (2026-04-21): the MLP standalone plateaus below LGBM+DGP
(0.96596 vs 0.97271 tuned OOF on the same 5-fold folds), but its
errors are *structurally different* — LGBM decides via axis-aligned
splits on rule thresholds, while the MLP uses a smooth manifold in
the full feature space. Blending at the softmax-probability level
should reduce the per-row error correlation and lift the tuned OOF.

Blend forms:
  arithmetic:  P = (1-w) * P_lgbm + w * P_mlp
  geometric:   P ∝ P_lgbm^(1-w) * P_mlp^w
Coord-ascent on w in [0, 0.5] under tuned-bias bal_acc.

Inputs:
  scripts/artifacts/oof_lgbm_dgp.npy, test_lgbm_dgp.npy
  scripts/artifacts/oof_mlp_balsoft.npy, test_mlp_balsoft.npy
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART_DIR = Path("scripts/artifacts")
OUT_DIR = Path("submissions")
OUT_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fast_bal_acc(y_true: np.ndarray, y_pred: np.ndarray, counts: np.ndarray) -> float:
    # Faster than sklearn for 3-class: per-class recall mean.
    correct = np.zeros(len(counts), dtype=np.int64)
    for c in range(len(counts)):
        mask = y_true == c
        correct[c] = int((y_pred[mask] == c).sum())
    return float((correct / counts).mean())


def coord_ascent_bias(oof: np.ndarray, y: np.ndarray,
                      start: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.int64)
    prior = counts / counts.sum()
    bias = -np.log(prior) if start is None else start.copy()
    grid = np.linspace(-2.5, 2.5, 26)

    def score(b: np.ndarray) -> float:
        return _fast_bal_acc(y, (log_oof + b).argmax(axis=1), counts)

    best = score(bias)
    for _ in range(10):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


log("loading OOF arrays")
oof_lgbm = np.load(ART_DIR / "oof_lgbm_dgp.npy")
test_lgbm = np.load(ART_DIR / "test_lgbm_dgp.npy")
oof_mlp = np.load(ART_DIR / "oof_mlp_balsoft.npy")
test_mlp = np.load(ART_DIR / "test_mlp_balsoft.npy")

log("loading labels")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
y = tr[TARGET].map(CLS2IDX).to_numpy().astype(np.int64)

log("tuning log-bias on each model individually (baselines)")
bias_l, score_l = coord_ascent_bias(oof_lgbm, y)
log(f"  LGBM+DGP tuned OOF = {score_l:.5f}  bias={dict(zip(CLASSES, bias_l.round(4)))}")
bias_m, score_m = coord_ascent_bias(oof_mlp, y)
log(f"  MLP+BalSoft tuned OOF = {score_m:.5f}  bias={dict(zip(CLASSES, bias_m.round(4)))}")

log("sweeping arithmetic blend w in [0, 0.5]")
weights = np.linspace(0.0, 0.5, 11)
arith_results = []
for w in weights:
    blend = (1 - w) * oof_lgbm + w * oof_mlp
    bias_b, score_b = coord_ascent_bias(blend, y)
    arith_results.append({"w": float(w), "tuned": score_b,
                          "bias": bias_b.tolist()})

best_arith = max(arith_results, key=lambda r: r["tuned"])
log(f"  best arithmetic w = {best_arith['w']:.2f}  tuned = {best_arith['tuned']:.5f}  "
    f"bias = {dict(zip(CLASSES, np.round(best_arith['bias'], 4).tolist()))}")

log("sweeping geometric blend w in [0, 0.5]")
log_lgbm = np.log(np.clip(oof_lgbm, 1e-9, 1.0))
log_mlp = np.log(np.clip(oof_mlp, 1e-9, 1.0))
geom_results = []
for w in weights:
    log_blend = (1 - w) * log_lgbm + w * log_mlp
    blend = np.exp(log_blend - log_blend.max(axis=1, keepdims=True))
    blend /= blend.sum(axis=1, keepdims=True)
    bias_b, score_b = coord_ascent_bias(blend, y)
    geom_results.append({"w": float(w), "tuned": score_b,
                         "bias": bias_b.tolist()})

best_geom = max(geom_results, key=lambda r: r["tuned"])
log(f"  best geometric w = {best_geom['w']:.2f}  tuned = {best_geom['tuned']:.5f}  "
    f"bias = {dict(zip(CLASSES, np.round(best_geom['bias'], 4).tolist()))}")

best = best_arith if best_arith["tuned"] >= best_geom["tuned"] else best_geom
blend_kind = "arithmetic" if best is best_arith else "geometric"
log(f"  overall best = {blend_kind} w={best['w']:.2f} -> {best['tuned']:.5f}")

if blend_kind == "arithmetic":
    oof_blend = (1 - best["w"]) * oof_lgbm + best["w"] * oof_mlp
    test_blend = (1 - best["w"]) * test_lgbm + best["w"] * test_mlp
else:
    log_test_lgbm = np.log(np.clip(test_lgbm, 1e-9, 1.0))
    log_test_mlp = np.log(np.clip(test_mlp, 1e-9, 1.0))
    lb = (1 - best["w"]) * log_lgbm + best["w"] * log_mlp
    oof_blend = np.exp(lb - lb.max(axis=1, keepdims=True))
    oof_blend /= oof_blend.sum(axis=1, keepdims=True)
    tb = (1 - best["w"]) * log_test_lgbm + best["w"] * log_test_mlp
    test_blend = np.exp(tb - tb.max(axis=1, keepdims=True))
    test_blend /= test_blend.sum(axis=1, keepdims=True)

bias_best = np.array(best["bias"])
log_oof_b = np.log(np.clip(oof_blend, 1e-9, 1.0))
tuned_cm = confusion_matrix(y, (log_oof_b + bias_best).argmax(axis=1)).tolist()

print("\n=== blend summary (OOF balanced accuracy) ===")
print(f"  LGBM+DGP tuned          {score_l:.5f}")
print(f"  MLP+BalSoft tuned       {score_m:.5f}")
print(f"  best blend ({blend_kind}, w={best['w']:.2f})  {best['tuned']:.5f}")
print(f"  Δ vs LGBM              {best['tuned'] - score_l:+.5f}")
print("\nconfusion matrix (best blend):")
print(pd.DataFrame(tuned_cm, index=CLASSES, columns=CLASSES))

log_test_b = np.log(np.clip(test_blend, 1e-9, 1.0))
tuned_test_idx = (log_test_b + bias_best).argmax(axis=1)
pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
    OUT_DIR / "submission_blend_lgbm_mlp_tuned.csv", index=False
)

np.save(ART_DIR / "oof_blend_lgbm_mlp.npy", oof_blend)
np.save(ART_DIR / "test_blend_lgbm_mlp.npy", test_blend)
with open(ART_DIR / "blend_lgbm_mlp_results.json", "w") as f:
    json.dump(
        {
            "classes": CLASSES,
            "baselines": {
                "lgbm_dgp_tuned": float(score_l),
                "lgbm_dgp_bias": bias_l.tolist(),
                "mlp_balsoft_tuned": float(score_m),
                "mlp_balsoft_bias": bias_m.tolist(),
            },
            "arithmetic_sweep": arith_results,
            "geometric_sweep": geom_results,
            "best": {
                "kind": blend_kind,
                "w": float(best["w"]),
                "tuned": float(best["tuned"]),
                "bias": np.asarray(best["bias"]).tolist(),
                "delta_vs_lgbm": float(best["tuned"] - score_l),
            },
            "confusion_matrix": tuned_cm,
        },
        f,
        indent=2,
    )
log(f"artefacts saved: submission_blend_lgbm_mlp_tuned.csv, "
    f"oof_blend_lgbm_mlp.npy, blend_lgbm_mlp_results.json")
