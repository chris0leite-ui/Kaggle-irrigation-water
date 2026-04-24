"""4-way multi-seed pseudo-label blend: recipe + pseudo_s1 + pseudo_s7 + pseudo_s123.

Reads previously-saved OOFs, holds bias fixed at recipe_full_te's tuned value
[1.4324, 1.4689, 3.4008] (LB-calibrated on LB 0.97939 recipe submission), and
searches over 4 log-space weights to find the best 4-way blend.

Why fixed bias: the 2026-04-21 binhigh experiment showed that re-tuning
log-bias after each new component manufactures OOF lift that doesn't transfer
(OOF +0.00036 → LB −0.00084). Every prior multi-seed 2-way/3-way used the
recipe bias held fixed and landed honest LB results (gap ≤ +0.00043).

Outputs:
  - per-component standalone OOF + errors + Jaccard vs recipe
  - pairwise sanity (recipe × each pseudo) — should reproduce 2026-04-24 LB results
  - known 3-way point (recipe 0.25 + s1 0.35 + s7 0.40) — reproduces LB 0.98005
  - 4-way greedy forward from known 3-way
  - 4-way dense grid (step=0.05) around the greedy pick
  - emits submission CSVs for the top 2 candidates and writes a results JSON
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import fast_bal_acc, log_blend

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    oof = np.load(ART / f"oof_{name}.npy")
    test = np.load(ART / f"test_{name}.npy")
    return oof, test


def _eval(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> tuple[float, int]:
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    return fast_bal_acc(y, pred, class_counts=cc), int((pred != y).sum())


def main() -> None:
    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(res["log_bias"], dtype=np.float64)
    log(f"anchor bias (recipe_full_te) = {bias.round(4).tolist()}")

    components = {
        "recipe":       _load("recipe_full_te"),
        "pseudo_s1":    _load("recipe_pseudolabel"),
        "pseudo_s7":    _load("recipe_pseudolabel_seed7labeler"),
        "pseudo_s123":  _load("recipe_pseudolabel_seed123labeler"),
    }

    log("=== standalone @ recipe bias ===")
    standalone: dict[str, tuple[float, int]] = {}
    for name, (oof, _) in components.items():
        ba, errs = _eval(oof, y, bias)
        standalone[name] = (ba, errs)
        log(f"  {name:14s}  bal={ba:.5f}  errs={errs}")

    # Error Jaccards vs recipe
    recipe_wrong = (np.log(np.clip(components["recipe"][0], 1e-9, 1.0)) + bias).argmax(1) != y
    for name, (oof, _) in components.items():
        if name == "recipe":
            continue
        w = (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(1) != y
        inter = int((recipe_wrong & w).sum())
        union = int((recipe_wrong | w).sum())
        log(f"  jaccard(recipe, {name}) = {inter/union:.4f}")

    def blend_eval(weights: dict[str, float]) -> tuple[float, np.ndarray, np.ndarray]:
        ws = np.array([weights[n] for n in weights], dtype=np.float64)
        oofs = [components[n][0] for n in weights]
        tests = [components[n][1] for n in weights]
        oof_blend = log_blend(oofs, ws)
        test_blend = log_blend(tests, ws)
        ba, _ = _eval(oof_blend, y, bias)
        return ba, oof_blend, test_blend

    log("=== pairwise sanity: recipe × each pseudo ===")
    pairwise_results: dict[str, dict] = {}
    for name in ["pseudo_s1", "pseudo_s7", "pseudo_s123"]:
        best = (0, bias, None)
        grid = np.linspace(0.0, 1.0, 41)
        rows = []
        for a in grid:
            ba, _, _ = blend_eval({"recipe": 1 - a, name: a})
            rows.append((float(a), float(ba)))
            if ba > best[0]:
                best = (ba, None, a)
        peak_a = best[2]
        log(f"  recipe × {name}: peak α={peak_a:.3f} bal={best[0]:.5f}")
        pairwise_results[name] = {"peak_alpha": float(peak_a), "peak_bal": float(best[0]),
                                   "curve": rows}

    log("=== known 3-way point (reproduces LB 0.98005 if artifacts align) ===")
    ba_3way, _, _ = blend_eval({"recipe": 0.25, "pseudo_s1": 0.35, "pseudo_s7": 0.40})
    log(f"  3-way (0.25, 0.35, 0.40) bal={ba_3way:.5f}  (prior run: 0.98029)")

    log("=== 4-way greedy forward from known 3-way + s123 scan ===")
    best_4 = (0.0, None, None, None)
    for beta in np.linspace(0.0, 0.6, 31):
        # shrink the 3 others proportionally
        shrink = 1.0 - beta
        w = {"recipe": 0.25 * shrink, "pseudo_s1": 0.35 * shrink,
             "pseudo_s7": 0.40 * shrink, "pseudo_s123": float(beta)}
        ba, _, _ = blend_eval(w)
        if ba > best_4[0]:
            best_4 = (ba, w, beta, ba - ba_3way)
    log(f"  best 4-way along s123-axis: β={best_4[2]:.3f} "
        f"bal={best_4[0]:.5f} Δ vs 3-way={best_4[3]:+.5f}")

    log("=== 4-way dense grid (step 0.05) ===")
    best_grid = (0.0, None)
    for wr in np.arange(0.10, 0.55, 0.05):
        for ws1 in np.arange(0.10, 0.55, 0.05):
            for ws7 in np.arange(0.10, 0.55, 0.05):
                ws123 = 1.0 - wr - ws1 - ws7
                if ws123 < 0.05 or ws123 > 0.55:
                    continue
                w = {"recipe": wr, "pseudo_s1": ws1,
                     "pseudo_s7": ws7, "pseudo_s123": ws123}
                ba, _, _ = blend_eval(w)
                if ba > best_grid[0]:
                    best_grid = (ba, w)
    log(f"  best grid: {best_grid[1]}  bal={best_grid[0]:.5f}")

    log("=== emitting 2 submissions ===")
    sample = pd.read_csv("data/sample_submission.csv")
    for tag, w in [("4way_axis", best_4[1]), ("4way_grid", best_grid[1])]:
        _, _, test_blend = blend_eval(w)
        pred = (np.log(np.clip(test_blend, 1e-9, 1.0)) + bias).argmax(1)
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred]
        path = SUB / f"submission_multiseed_{tag}.csv"
        sub.to_csv(path, index=False)
        np.save(ART / f"oof_multiseed_{tag}.npy", np.zeros((1,)))  # placeholder
        log(f"  wrote {path} weights={ {k: round(v,3) for k,v in w.items()} }")

    out = {
        "bias": bias.tolist(),
        "standalone": {k: {"bal": v[0], "errs": v[1]} for k, v in standalone.items()},
        "pairwise": pairwise_results,
        "three_way_fixed_point": float(ba_3way),
        "four_way_axis": {"weights": best_4[1], "bal": float(best_4[0])},
        "four_way_grid": {"weights": best_grid[1], "bal": float(best_grid[0])},
    }
    (ART / "blend_4way_multiseed_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {ART / 'blend_4way_multiseed_results.json'}")


if __name__ == "__main__":
    main()
