"""Blend LGBM+DGP with each balanced-ensemble model (BRF / EasyEnsemble
/ RUSBoost).

Pure prob-space linear blend followed by coord-ascent log-bias tuning,
evaluated on the saved 5-fold OOFs from scripts/benchmark_dgp.py and
scripts/benchmark_balanced_ensembles.py.

Reference ceilings (OOF bal_acc, tuned log-bias):
  LGBM+DGP       0.97271
  EasyEnsemble   0.96932
  RUSBoost       0.96666
  BalancedRF     0.96535

Hypothesis: balanced-ensemble confusion matrices have a different
High <-> Medium error profile than LGBM+DGP; the blend's per-class
recall may be more balanced after log-bias tuning, even when each
standalone component is below LGBM+DGP.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

SEED = 42
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


log("loading labels + OOF / test probs")
y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].map(CLS2IDX).values.astype(np.int32)
te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")

oofs = {
    "lgbm_dgp": np.load(ART_DIR / "oof_lgbm_dgp.npy"),
    "brf":      np.load(ART_DIR / "oof_brf.npy"),
    "easy":     np.load(ART_DIR / "oof_easy.npy"),
    "rusb":     np.load(ART_DIR / "oof_rusb.npy"),
}
tests = {
    "lgbm_dgp": np.load(ART_DIR / "test_lgbm_dgp.npy"),
    "brf":      np.load(ART_DIR / "test_brf.npy"),
    "easy":     np.load(ART_DIR / "test_easy.npy"),
    "rusb":     np.load(ART_DIR / "test_rusb.npy"),
}
for k, v in oofs.items():
    # normalize row-wise to ensure prob simplex (AdaBoost outputs can drift)
    v = np.clip(v, 1e-9, 1.0)
    oofs[k] = v / v.sum(axis=1, keepdims=True)
    t = np.clip(tests[k], 1e-9, 1.0)
    tests[k] = t / t.sum(axis=1, keepdims=True)
    log(f"  {k:<10s} argmax bal_acc = {balanced_accuracy_score(y, oofs[k].argmax(axis=1)):.5f}")


def tune_log_bias(oof: np.ndarray) -> tuple[float, np.ndarray]:
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))

    def score_bias(bias):
        return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))

    bias = -np.log(prior)
    best = score_bias(bias)
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score_bias(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return best, bias


log("=== pairwise LGBM+DGP x balanced-ensemble blends ===")
blend_sweep = {}
ws = np.round(np.arange(0.0, 1.01, 0.05), 2)
for partner in ["easy", "brf", "rusb"]:
    results = []
    for w in ws:
        blend = w * oofs["lgbm_dgp"] + (1 - w) * oofs[partner]
        bal, bias = tune_log_bias(blend)
        results.append((float(w), bal, bias.tolist()))
    best = max(results, key=lambda r: r[1])
    blend_sweep[partner] = {"sweep": results, "best_w": best[0], "best_bal": best[1], "best_bias": best[2]}
    log(f"  lgbm_dgp x {partner:<4s}  best w={best[0]:.2f}  bal={best[1]:.5f}  bias={[round(b,3) for b in best[2]]}")

log("=== pairwise LGBM+DGP x balanced-ensemble (geometric blend) ===")
geo_sweep = {}
for partner in ["easy", "brf", "rusb"]:
    results = []
    for w in ws:
        lo = np.log(oofs["lgbm_dgp"])
        lp = np.log(oofs[partner])
        blend = np.exp(w * lo + (1 - w) * lp)
        blend /= blend.sum(axis=1, keepdims=True)
        bal, bias = tune_log_bias(blend)
        results.append((float(w), bal, bias.tolist()))
    best = max(results, key=lambda r: r[1])
    geo_sweep[partner] = {"sweep": results, "best_w": best[0], "best_bal": best[1], "best_bias": best[2]}
    log(f"  geo lgbm_dgp x {partner:<4s}  best w={best[0]:.2f}  bal={best[1]:.5f}")

log("=== three-way blend (LGBM+DGP, Easy, BRF) — coarse simplex ===")
simplex_results = []
for w1 in np.round(np.arange(0.0, 1.01, 0.1), 2):
    for w2 in np.round(np.arange(0.0, 1.01 - w1 + 1e-9, 0.1), 2):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < -1e-9:
            continue
        blend = w1 * oofs["lgbm_dgp"] + w2 * oofs["easy"] + w3 * oofs["brf"]
        bal, bias = tune_log_bias(blend)
        simplex_results.append((float(w1), float(w2), float(w3), bal, bias.tolist()))
best3 = max(simplex_results, key=lambda r: r[3])
log(f"  3-way best  w_lgbm={best3[0]}  w_easy={best3[1]}  w_brf={best3[2]}  bal={best3[3]:.5f}")

log("=== three-way blend (LGBM+DGP, Easy, RUSBoost) — coarse simplex ===")
simplex4 = []
for w1 in np.round(np.arange(0.0, 1.01, 0.1), 2):
    for w2 in np.round(np.arange(0.0, 1.01 - w1 + 1e-9, 0.1), 2):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < -1e-9:
            continue
        blend = w1 * oofs["lgbm_dgp"] + w2 * oofs["easy"] + w3 * oofs["rusb"]
        bal, bias = tune_log_bias(blend)
        simplex4.append((float(w1), float(w2), float(w3), bal, bias.tolist()))
best4 = max(simplex4, key=lambda r: r[3])
log(f"  3-way best  w_lgbm={best4[0]}  w_easy={best4[1]}  w_rusb={best4[2]}  bal={best4[3]:.5f}")

log("=== four-way blend (all models) — coarse simplex, step 0.1 ===")
simplex_all = []
for w1 in np.round(np.arange(0.0, 1.01, 0.1), 2):
    for w2 in np.round(np.arange(0.0, 1.01 - w1 + 1e-9, 0.1), 2):
        for w3 in np.round(np.arange(0.0, 1.01 - w1 - w2 + 1e-9, 0.1), 2):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 < -1e-9:
                continue
            blend = w1 * oofs["lgbm_dgp"] + w2 * oofs["easy"] + w3 * oofs["brf"] + w4 * oofs["rusb"]
            bal, bias = tune_log_bias(blend)
            simplex_all.append((float(w1), float(w2), float(w3), float(w4), bal, bias.tolist()))
bestA = max(simplex_all, key=lambda r: r[4])
log(f"  4-way best  w=({bestA[0]},{bestA[1]},{bestA[2]},{bestA[3]})  bal={bestA[4]:.5f}")


# === build test submission for the best overall blend ===
all_candidates = [
    ("lgbm_dgp only",                1.0, blend_sweep["easy"]["sweep"][-1][1] if blend_sweep["easy"]["sweep"][-1][0] == 1.0 else None),
]
# pick the true best-bal across everything
candidates = []
for partner, v in blend_sweep.items():
    candidates.append((f"linear lgbm_dgp x {partner}", v["best_bal"],
                       {"mode": "linear", "partners": [("lgbm_dgp", v["best_w"]), (partner, 1.0 - v["best_w"])],
                        "bias": v["best_bias"]}))
for partner, v in geo_sweep.items():
    candidates.append((f"geo lgbm_dgp x {partner}", v["best_bal"],
                       {"mode": "geo", "partners": [("lgbm_dgp", v["best_w"]), (partner, 1.0 - v["best_w"])],
                        "bias": v["best_bias"]}))
candidates.append(("3-way lgbm+easy+brf", best3[3],
                   {"mode": "linear",
                    "partners": [("lgbm_dgp", best3[0]), ("easy", best3[1]), ("brf", best3[2])],
                    "bias": best3[4]}))
candidates.append(("3-way lgbm+easy+rusb", best4[3],
                   {"mode": "linear",
                    "partners": [("lgbm_dgp", best4[0]), ("easy", best4[1]), ("rusb", best4[2])],
                    "bias": best4[4]}))
candidates.append(("4-way all", bestA[4],
                   {"mode": "linear",
                    "partners": [("lgbm_dgp", bestA[0]), ("easy", bestA[1]), ("brf", bestA[2]), ("rusb", bestA[3])],
                    "bias": bestA[5]}))

best_overall = max(candidates, key=lambda c: c[1])
log(f"\n=== BEST BLEND: {best_overall[0]}  bal={best_overall[1]:.5f} ===")
log(f"  config: {best_overall[2]}")

cfg = best_overall[2]
if cfg["mode"] == "linear":
    test_blend = sum(w * tests[k] for k, w in cfg["partners"])
else:  # geometric
    log_sum = sum(w * np.log(tests[k]) for k, w in cfg["partners"])
    test_blend = np.exp(log_sum)
    test_blend /= test_blend.sum(axis=1, keepdims=True)

bias = np.array(cfg["bias"])
log_test = np.log(np.clip(test_blend, 1e-9, 1.0))
tuned_idx = (log_test + bias).argmax(axis=1)
sub_path = OUT_DIR / "submission_blend_best.csv"
pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(sub_path, index=False)
log(f"wrote {sub_path}")

# diagnostic confusion matrix on OOF
if cfg["mode"] == "linear":
    oof_blend = sum(w * oofs[k] for k, w in cfg["partners"])
else:
    log_sum = sum(w * np.log(oofs[k]) for k, w in cfg["partners"])
    oof_blend = np.exp(log_sum)
    oof_blend /= oof_blend.sum(axis=1, keepdims=True)
log_oof = np.log(np.clip(oof_blend, 1e-9, 1.0))
pred = (log_oof + bias).argmax(axis=1)
cm = confusion_matrix(y, pred)
print("\nconfusion matrix (OOF, tuned):")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

with open(ART_DIR / "blend_lgbm_balanced_results.json", "w") as f:
    json.dump({
        "seed": SEED,
        "class_priors": prior.tolist(),
        "pairwise_linear": blend_sweep,
        "pairwise_geo": geo_sweep,
        "three_way_lgbm_easy_brf": {"best": best3},
        "three_way_lgbm_easy_rusb": {"best": best4},
        "four_way": {"best": [bestA[0], bestA[1], bestA[2], bestA[3], bestA[4]]},
        "best_overall": {"name": best_overall[0], "bal_acc": best_overall[1], "config": best_overall[2]},
    }, f, indent=2)
log("results written")
