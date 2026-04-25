"""Blend-gate: n_ens=4 RealMLP vs LB-best 3-way + n_ens=1 baseline.

Mirrors scripts/blend_realmlp.py + scripts/emit_realmlp_3stack.py to
answer one question: does n_ens=4 stack into the 3-stack better than
n_ens=1 did (LB 0.98008)?

Standalone OOF tuned was a wash (0.97638 vs 0.97636), but blend math
depends on err-count + Jaccard at FIXED recipe bias. The n_ens=1
3-stack had errs 9572 (-301 vs anchor), Jaccard 0.92 vs LB3, OOF
0.98061. If n_ens=4 produces a 3-stack with errs <= 9572 AND OOF >
0.98061, we have a real upgrade.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(arr):
    return arr / np.clip(arr.sum(1, keepdims=True), 1e-9, None)


def _pred(probs, bias=BIAS):
    return (np.log(np.clip(probs, 1e-12, 1.0)) + bias).argmax(1)


def _err(probs, y, bias=BIAS):
    return _pred(probs, bias) != y


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


def jaccard(a_err, b_err):
    inter = int((a_err & b_err).sum())
    union = int((a_err | b_err).sum())
    return inter / max(union, 1)


def _load(name):
    return _normed(np.load(ART / f"oof_{name}.npy")), \
           _normed(np.load(ART / f"test_{name}.npy"))


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    recipe_oof, recipe_test = _load("recipe_full_te")
    p_s1_oof, p_s1_test = _load("recipe_pseudolabel")
    p_s7_oof, p_s7_test = _load("recipe_pseudolabel_seed7labeler")
    nonrule_oof, nonrule_test = _load("xgb_nonrule")
    nonrule_iso_oof, nonrule_iso_test = iso_cal(nonrule_oof, nonrule_test, y)

    rm1_oof, rm1_test = _load("realmlp")
    rm4_oof, rm4_test = _load("realmlp_ens4")

    # LB-best 3-way anchor.
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_oof = log_blend([recipe_oof, p_s1_oof, p_s7_oof], w3)
    lb3_test = log_blend([recipe_test, p_s1_test, p_s7_test], w3)
    bal_lb3 = balanced_accuracy_score(y, _pred(lb3_oof))
    err_lb3 = _err(lb3_oof, y)
    print(f"LB-best 3-way anchor      OOF={bal_lb3:.5f}  errs={int(err_lb3.sum())}")

    print()
    print("=== STANDALONE comparison ===")
    for name, oof in [("realmlp (n_ens=1)", rm1_oof),
                      ("realmlp (n_ens=4)", rm4_oof)]:
        bal_argmax = balanced_accuracy_score(y, oof.argmax(1))
        bal_at_bias = balanced_accuracy_score(y, _pred(oof))
        err_at_bias = int(_err(oof, y).sum())
        j_lb3 = jaccard(_err(oof, y), err_lb3)
        print(f"  {name:25s}  argmax={bal_argmax:.5f}  "
              f"@bias={bal_at_bias:.5f}  errs={err_at_bias}  "
              f"Jaccard(LB3)={j_lb3:.4f}")

    print()
    print("=== ALPHA SWEEP — log-blend(LB3, realmlp), fixed BIAS ===")
    alphas = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
              0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5]
    for label, rm_oof in [("n_ens=1", rm1_oof), ("n_ens=4", rm4_oof)]:
        print(f"\n  RealMLP {label}:")
        best = None
        for a in alphas:
            blend = log_blend([lb3_oof, rm_oof], np.array([1 - a, a]))
            bal = balanced_accuracy_score(y, _pred(blend))
            errs = int(_err(blend, y).sum())
            mark = "  ←peak" if best is None or bal > best[0] else ""
            print(f"    α={a:.3f}  OOF={bal:.5f}  errs={errs}{mark}")
            if best is None or bal > best[0]:
                best = (bal, a, errs)
        print(f"  best α={best[1]:.3f}  OOF={best[0]:.5f}  errs={best[2]}")

    print()
    print("=== 3-STACK (LB3 + RealMLP @α + nonrule_iso @0.075) ===")
    for label, rm_oof, rm_test in [
        ("n_ens=1", rm1_oof, rm1_test),
        ("n_ens=4", rm4_oof, rm4_test),
    ]:
        print(f"\n  RealMLP {label}:")
        best = None
        for a in [0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35]:
            s1 = log_blend([lb3_oof, rm_oof], np.array([1 - a, a]))
            s2 = log_blend([s1, nonrule_iso_oof], np.array([0.925, 0.075]))
            bal = balanced_accuracy_score(y, _pred(s2))
            errs = int(_err(s2, y).sum())
            mark = "  ←peak" if best is None or bal > best[0] else ""
            print(f"    α_rm={a:.3f}  OOF={bal:.5f}  errs={errs}{mark}")
            if best is None or bal > best[0]:
                # Also build test-side prediction at this α.
                ts1 = log_blend([lb3_test, rm_test], np.array([1 - a, a]))
                ts2 = log_blend([ts1, nonrule_iso_test],
                                np.array([0.925, 0.075]))
                best = (bal, a, errs, ts2)
        print(f"  best α={best[1]:.3f}  OOF={best[0]:.5f}  errs={best[2]}")

        # If n_ens=4 wins, emit submission.
        if label == "n_ens=4" and best[0] > 0.98061:
            sample = pd.read_csv(DATA / "sample_submission.csv")
            pred_test = _pred(best[3])
            sub = sample.copy()
            sub[TARGET] = [CLASSES[i] for i in pred_test]
            path = SUB / "submission_lb3_realmlp_ens4_nonruleiso.csv"
            sub.to_csv(path, index=False)
            print(f"  emitted {path}  (OOF {best[0]:.5f} > "
                  f"prior 3-stack 0.98061)")


if __name__ == "__main__":
    main()
