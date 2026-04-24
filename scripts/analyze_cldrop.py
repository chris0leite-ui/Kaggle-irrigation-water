"""Diagnose the cleanlab DROP OOF vs LB-best baseline.

Key questions:
1. Error Jaccard vs LB-best and vs recipe — is DROP orthogonal?
2. Error count delta — does DROP have FEWER errors? (necessary for blend)
3. Per-class recall — does DROP trade Medium for High or vice versa?
4. Blend sweep vs LB-best (fixed bias) — does any α > 0 lift?
5. Would a fresh greedy over the enlarged bank pick DROP?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
DATA = Path("data")


def log_blend(a, b, w=0.5):
    eps = 1e-12
    la = np.log(np.clip(a, eps, 1))
    lb = np.log(np.clip(b, eps, 1))
    z = w * la + (1 - w) * lb
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z); return ez / ez.sum(axis=1, keepdims=True)


def jaccard_errs(pred_a, pred_b, y):
    ea = pred_a != y
    eb = pred_b != y
    inter = int(np.logical_and(ea, eb).sum())
    union = int(np.logical_or(ea, eb).sum())
    return inter / max(union, 1), int(ea.sum()), int(eb.sum())


def tune_bias(oof, y, grid=None):
    if grid is None:
        grid = np.linspace(-1, 5, 121)
    eps = 1e-12
    lg = np.log(np.clip(oof, eps, 1))
    bias = np.zeros(3); best = balanced_accuracy_score(y, lg.argmax(1))
    for _ in range(5):
        imp = False
        for c in range(3):
            for v in grid:
                b = bias.copy(); b[c] = v
                sc = balanced_accuracy_score(y, (lg + b).argmax(1))
                if sc > best + 1e-7:
                    best = sc; bias = b; imp = True
        if not imp:
            break
    return bias, best


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map({"Low":0,"Medium":1,"High":2}).to_numpy()
    pi_src = np.bincount(y, minlength=3) / len(y)

    # Load all three
    recipe = np.load(ART / "oof_recipe_full_te.npy")
    pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    drop = np.load(ART / "oof_recipe_full_te_cldrop.npy")
    lb_best = log_blend(recipe, pseudo, 0.5)
    recipe_bias = np.array([1.4324, 1.4689, 3.4008])

    print("=" * 70)
    print("STANDALONE OOF bal_acc")
    print("=" * 70)
    for name, oof in [("recipe", recipe), ("pseudo", pseudo),
                      ("lb-best blend", lb_best), ("drop", drop)]:
        arg = balanced_accuracy_score(y, oof.argmax(1))
        lg = np.log(np.clip(oof, 1e-12, 1))
        b_rec = balanced_accuracy_score(y, (lg + recipe_bias).argmax(1))
        bias, b_tuned = tune_bias(oof, y)
        print(f"  {name:18s}  argmax={arg:.5f}  @recipe_bias={b_rec:.5f}  "
              f"tuned={b_tuned:.5f}  bias={bias.round(3).tolist()}")

    # DROP vs lb-best comparisons
    print("\n" + "=" * 70)
    print("DROP vs LB-best at recipe bias")
    print("=" * 70)
    drop_lg = np.log(np.clip(drop, 1e-12, 1))
    drop_pred = (drop_lg + recipe_bias).argmax(1)
    lb_lg = np.log(np.clip(lb_best, 1e-12, 1))
    lb_pred = (lb_lg + recipe_bias).argmax(1)
    j, e_drop, e_lb = jaccard_errs(drop_pred, lb_pred, y)
    print(f"  Jaccard(errs)={j:.4f}  drop_errs={e_drop}  lb_errs={e_lb}  "
          f"drop_has_fewer_errs={e_drop < e_lb}")

    cm_drop = confusion_matrix(y, drop_pred)
    cm_lb = confusion_matrix(y, lb_pred)
    r_drop = cm_drop.diagonal() / cm_drop.sum(axis=1)
    r_lb = cm_lb.diagonal() / cm_lb.sum(axis=1)
    print(f"  per-class recall  drop: L={r_drop[0]:.4f} M={r_drop[1]:.4f} H={r_drop[2]:.4f}")
    print(f"                    lb:   L={r_lb[0]:.4f} M={r_lb[1]:.4f} H={r_lb[2]:.4f}")
    print(f"                    Δ:    L={r_drop[0]-r_lb[0]:+.4f} "
          f"M={r_drop[1]-r_lb[1]:+.4f} H={r_drop[2]-r_lb[2]:+.4f}")

    # Fixed-bias log-blend sweep
    print("\n" + "=" * 70)
    print("Log-blend DROP × LB-best at FIXED recipe bias")
    print("=" * 70)
    print(f"  baseline (α=0) = {balanced_accuracy_score(y, lb_pred):.5f}")
    for a in [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        b = log_blend(drop, lb_best, a)
        pred = (np.log(np.clip(b, 1e-12, 1)) + recipe_bias).argmax(1)
        sc = balanced_accuracy_score(y, pred)
        delta = sc - balanced_accuracy_score(y, lb_pred)
        print(f"  α_drop={a:4.2f}  OOF={sc:.5f}  Δ={delta:+.5f}")

    # Fixed-bias log-blend vs recipe alone (not pseudo)
    print("\n" + "=" * 70)
    print("Log-blend DROP × RECIPE at FIXED recipe bias")
    print("=" * 70)
    recipe_pred = (np.log(np.clip(recipe, 1e-12, 1)) + recipe_bias).argmax(1)
    print(f"  baseline (α=0) = {balanced_accuracy_score(y, recipe_pred):.5f}")
    for a in [0.1, 0.25, 0.5, 0.75, 0.9]:
        b = log_blend(drop, recipe, a)
        pred = (np.log(np.clip(b, 1e-12, 1)) + recipe_bias).argmax(1)
        sc = balanced_accuracy_score(y, pred)
        delta = sc - balanced_accuracy_score(y, recipe_pred)
        print(f"  α_drop={a:4.2f}  OOF={sc:.5f}  Δ={delta:+.5f}")


if __name__ == "__main__":
    main()
