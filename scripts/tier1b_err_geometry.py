"""Tier 1b #2: Error-geometry analysis of the CURRENT LB-best 3-stack.

For the 9,572 OOF errors, break down by:
  - (dgp_score) × (true class) × (predicted class)
  - rule_pred × direction of confusion
  - score-band aggregates to see where mass concentrates

Goal: identify the dominant error bucket so we can train a targeted
specialist (e.g., spec3_lm_v3: binary Low↔Medium at score=3).
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend, CLS2IDX, fast_bal_acc  # noqa: E402

ART = Path("scripts/artifacts")
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def build_lbbest_oof(y):
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_t = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    nr_iso, _ = iso_cal(nr, nr_t, y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3 = log_blend([r, s1, s7], w3)
    s1 = log_blend([lb3, rm], np.array([0.8, 0.2]))
    s2 = log_blend([s1, nr_iso], np.array([0.925, 0.075]))
    return s2


def main():
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    oof = build_lbbest_oof(y)
    pred = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    err_mask = pred != y
    print(f"LB-best stack OOF bal_acc = {fast_bal_acc(y, pred):.5f}")
    print(f"total errors = {err_mask.sum():,}")

    # Per-class confusion
    print("\n=== confusion matrix (rows=true, cols=predicted) ===")
    conf = np.zeros((3, 3), dtype=np.int64)
    for t in range(3):
        for p in range(3):
            conf[t, p] = ((y == t) & (pred == p)).sum()
    labels = ["Low", "Medium", "High"]
    print(f"{'':8s} " + " ".join(f"{l:>8s}" for l in labels) + f" {'recall':>7s}")
    for t in range(3):
        row = " ".join(f"{conf[t, p]:>8d}" for p in range(3))
        rec = conf[t, t] / max(conf[t].sum(), 1)
        print(f"{labels[t]:8s} {row} {rec:>7.4f}")

    # Error directions (true, pred) excluding diagonal
    print("\n=== error buckets (true -> predicted), total 9,572 ===")
    total_errs = err_mask.sum()
    for t in range(3):
        for p in range(3):
            if t == p:
                continue
            n = conf[t, p]
            frac = n / max(total_errs, 1)
            print(f"  {labels[t]:8s} -> {labels[p]:8s}  n={n:>6d}  ({frac:6.2%})")

    # Per-score × direction
    tr_d = add_distance_features(train)
    score = tr_d["dgp_score"].to_numpy()
    print("\n=== score × error direction (top 15 mass-carrying buckets) ===")
    print(f"{'score':>5} {'true':>7s} {'pred':>7s} {'n':>6} {'%total_errs':>12s}")
    buckets = []
    for s in range(10):
        for t in range(3):
            for p in range(3):
                if t == p:
                    continue
                n = int((err_mask & (score == s) & (y == t) & (pred == p)).sum())
                if n > 0:
                    buckets.append((s, labels[t], labels[p], n))
    buckets.sort(key=lambda x: -x[3])
    cum = 0
    for s, tl, pl, n in buckets[:15]:
        cum += n
        print(f"{s:>5d} {tl:>7s} {pl:>7s} {n:>6d} {n/total_errs:>11.2%} "
              f"(cum {cum/total_errs:.1%})")

    # Summary: dominant buckets
    top_n = 3
    top_buckets = buckets[:top_n]
    top_mass = sum(b[3] for b in top_buckets) / total_errs
    print(f"\n=== TOP {top_n} BUCKETS = {top_mass:.1%} of total errors ===")
    for s, tl, pl, n in top_buckets:
        print(f"  score={s}  {tl} -> {pl}  n={n}  recoverable-mass={n} rows")

    # Per-class recall (reproduce for reference)
    cc = np.bincount(y, minlength=3)
    print(f"\nclass counts in train: L={cc[0]} M={cc[1]} H={cc[2]}")
    print(f"error mass in TRULY-H rows: {conf[2, :2].sum()} / {cc[2]} = "
          f"{conf[2, :2].sum()/cc[2]:.4f}")
    print(f"error mass in TRULY-M rows: {conf[1, [0,2]].sum()} / {cc[1]} = "
          f"{conf[1, [0,2]].sum()/cc[1]:.4f}")
    print(f"error mass in TRULY-L rows: {conf[0, 1:].sum()} / {cc[0]} = "
          f"{conf[0, 1:].sum()/cc[0]:.4f}")

    out = {
        "total_errors": int(total_errs),
        "confusion": conf.tolist(),
        "top_buckets": [dict(score=s, true=tl, pred=pl, n=n)
                         for s, tl, pl, n in buckets[:20]],
    }
    (ART / "tier1b_err_geometry_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote scripts/artifacts/tier1b_err_geometry_results.json")


if __name__ == "__main__":
    main()
