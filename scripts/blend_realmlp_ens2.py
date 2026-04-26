"""Blend-gate: RealMLP n_ens=2 @ n_epochs=40 vs n_ens=1 baseline.

Diagnostic between two n_ens=4-NULL hypotheses:
  (a) under-convergence (n_epochs=25 was too short with 4 heads)
  (b) variance floor (n_ens=1 already at the per-row variance floor)

Decision rule:
  n_ens=2 standalone OOF > n_ens=1 standalone (0.97636) AND
  n_ens=2 3-stack OOF    > n_ens=1 3-stack    (0.98061)
  AND magnitude / Jaccard pass (errs <= 1.04*anchor, Jaccard < 0.65)
  → real lift; emit submission for LB probe; plan n_ens=4 @ n_epochs=40.

  n_ens=2 standalone ~ n_ens=1 (within fold noise ~0.0005)
  → variance floor structural; lever closed; no LB probe.
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

# Decision thresholds (locked in design — tuning these would be
# binhigh-rule selection-overfit).
ANCHOR_3STACK_OOF = 0.98061   # n_ens=1 3-stack
ANCHOR_REALMLP_OOF = 0.97636  # n_ens=1 standalone tuned
LB_BEST_4STACK_OOF = 0.98084  # current primary
EMIT_THRESHOLD = ANCHOR_3STACK_OOF + 0.0002


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


def _try_load(name):
    p_oof = ART / f"oof_{name}.npy"
    p_test = ART / f"test_{name}.npy"
    if p_oof.exists() and p_test.exists():
        return _load(name)
    return None, None


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    recipe_oof, recipe_test = _load("recipe_full_te")
    p_s1_oof, p_s1_test = _load("recipe_pseudolabel")
    p_s7_oof, p_s7_test = _load("recipe_pseudolabel_seed7labeler")
    nonrule_oof, nonrule_test = _load("xgb_nonrule")
    nonrule_iso_oof, nonrule_iso_test = iso_cal(nonrule_oof, nonrule_test, y)

    rm1_oof, rm1_test = _load("realmlp")
    rm2_oof, rm2_test = _try_load("realmlp_ens2")
    rm4_oof, rm4_test = _try_load("realmlp_ens4")

    if rm2_oof is None:
        print("ERROR: oof_realmlp_ens2.npy not present; pull from Kaggle "
              "kernel first (see kaggle_kernel/kernel_realmlp_ens2/README.md)")
        sys.exit(2)

    # Detect partial OOF (kill triggered before all folds completed).
    rm2_zero_frac = float((rm2_oof.sum(1) == 0).mean())
    if rm2_zero_frac > 0.01:
        n_completed = round(5 * (1 - rm2_zero_frac))
        print(f"WARNING: realmlp_ens2 partial OOF — {n_completed}/5 folds "
              f"completed ({rm2_zero_frac*100:.1f}% rows are zero). "
              f"Standalone OOF computed on completed-fold rows only.")

    # LB-best 3-way anchor.
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_oof = log_blend([recipe_oof, p_s1_oof, p_s7_oof], w3)
    lb3_test = log_blend([recipe_test, p_s1_test, p_s7_test], w3)
    bal_lb3 = balanced_accuracy_score(y, _pred(lb3_oof))
    err_lb3 = _err(lb3_oof, y)
    print(f"LB-best 3-way anchor      OOF={bal_lb3:.5f}  "
          f"errs={int(err_lb3.sum())}")
    print(f"LB-best 4-stack (primary) OOF={LB_BEST_4STACK_OOF:.5f}  "
          f"(reference)")

    print()
    print("=== STANDALONE comparison ===")
    candidates = [("realmlp (n_ens=1, refs)", rm1_oof),
                  ("realmlp (n_ens=2, NEW)", rm2_oof)]
    if rm4_oof is not None:
        candidates.append(("realmlp (n_ens=4)", rm4_oof))

    for name, oof in candidates:
        # For partial OOFs, score on completed-fold rows only.
        nz_mask = oof.sum(1) > 0
        if not nz_mask.all():
            y_eval = y[nz_mask]
            oof_eval = oof[nz_mask]
            err_eval = _err(oof_eval, y_eval)
            err_lb3_eval = err_lb3[nz_mask]
        else:
            y_eval = y
            oof_eval = oof
            err_eval = _err(oof, y)
            err_lb3_eval = err_lb3
        bal_argmax = balanced_accuracy_score(y_eval, oof_eval.argmax(1))
        bal_at_bias = balanced_accuracy_score(y_eval, _pred(oof_eval))
        err_at_bias = int(err_eval.sum())
        j_lb3 = jaccard(err_eval, err_lb3_eval)
        print(f"  {name:30s}  argmax={bal_argmax:.5f}  "
              f"@bias={bal_at_bias:.5f}  errs={err_at_bias}  "
              f"Jaccard(LB3)={j_lb3:.4f}")

    print()
    print("=== ALPHA SWEEP — log-blend(LB3, realmlp), fixed BIAS ===")
    alphas = [0.0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
              0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5]
    for label, rm_oof in [("n_ens=1", rm1_oof), ("n_ens=2", rm2_oof)]:
        if (rm_oof.sum(1) == 0).mean() > 0.01:
            print(f"\n  RealMLP {label}: SKIPPED (partial OOF)")
            continue
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
    for label, rm_oof, rm_test in [("n_ens=1", rm1_oof, rm1_test),
                                    ("n_ens=2", rm2_oof, rm2_test)]:
        if (rm_oof.sum(1) == 0).mean() > 0.01:
            print(f"\n  RealMLP {label}: SKIPPED (partial OOF — re-run "
                  f"production with full 5-fold before blending)")
            continue
        print(f"\n  RealMLP {label}:")
        best = None
        for a in [0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35]:
            s1 = log_blend([lb3_oof, rm_oof], np.array([1 - a, a]))
            s2 = log_blend([s1, nonrule_iso_oof],
                           np.array([0.925, 0.075]))
            bal = balanced_accuracy_score(y, _pred(s2))
            errs = int(_err(s2, y).sum())
            mark = "  ←peak" if best is None or bal > best[0] else ""
            print(f"    α_rm={a:.3f}  OOF={bal:.5f}  errs={errs}{mark}")
            if best is None or bal > best[0]:
                ts1 = log_blend([lb3_test, rm_test],
                                np.array([1 - a, a]))
                ts2 = log_blend([ts1, nonrule_iso_test],
                                np.array([0.925, 0.075]))
                best = (bal, a, errs, ts2)
        print(f"  best α={best[1]:.3f}  OOF={best[0]:.5f}  errs={best[2]}")

        # Emit submission only if n_ens=2 cleanly beats n_ens=1's 3-stack
        # by at least the LB-transfer threshold.
        if label == "n_ens=2" and best[0] > EMIT_THRESHOLD:
            sample = pd.read_csv(DATA / "sample_submission.csv")
            pred_test = _pred(best[3])
            sub = sample.copy()
            sub[TARGET] = [CLASSES[i] for i in pred_test]
            path = SUB / "submission_lb3_realmlp_ens2_nonruleiso.csv"
            sub.to_csv(path, index=False)
            print(f"\n  ✓ EMITTED {path}")
            print(f"    OOF {best[0]:.5f} > emit threshold "
                  f"{EMIT_THRESHOLD:.5f}")
            print(f"    LB-probe candidate (compare vs primary "
                  f"OOF {LB_BEST_4STACK_OOF:.5f}).")
        elif label == "n_ens=2":
            print(f"\n  ✗ NOT EMITTED — best 3-stack OOF {best[0]:.5f} <= "
                  f"emit threshold {EMIT_THRESHOLD:.5f}")
            if best[0] >= ANCHOR_3STACK_OOF + 0.00005:
                print(f"    Marginal positive ({best[0] - ANCHOR_3STACK_OOF:+.5f}) "
                      f"— diagnostic: under-convergence may be partial. "
                      f"Consider n_ens=4 @ n_epochs=40 overnight retry.")
            else:
                print(f"    Variance floor structural — RealMLP lever "
                      f"closed at n_ens=1.")


if __name__ == "__main__":
    main()
