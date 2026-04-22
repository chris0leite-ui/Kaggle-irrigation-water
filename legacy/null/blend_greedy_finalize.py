"""Reconstruct the greedy-forward winner and the meta-stack winner
from saved OOF/test npys, write submissions + per-class diagnostics.

Greedy path (from the successful blend_ensemble.py run):
  start: xgb_hybrid_v3 (OOF 0.97352)
  + xgb_routed_v3     w=0.4  -> OOF 0.97368
  + xgb_spec_678      w=0.1  -> OOF 0.97375  (current best)
  stop (adding xgb_dist hurt)

Also:
  * Two alternative-weight variants around the greedy solution, to
    test sensitivity (w_routed +/- 0.1, w_spec +/- 0.05)
  * A logistic-regression meta-stacker with class_weight=balanced on
    concat(P_hybrid, P_routed, P_dgp, P_xgbdist, P_spec) as a final
    check. Not always optimal — trees/hybrids already capture much.

Every submission is gated on beating xgb_hybrid_v3 OOF (0.97352) by
>= 3e-4. Submissions NOT auto-uploaded (see CLAUDE.md "LB SUBMISSION
RULE — ALWAYS ASK FIRST").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def fast_bal_acc(y, pred, cc):
    m = pred == y
    hit = np.array([m[y == k].sum() for k in range(3)])
    return float((hit / np.maximum(cc, 1)).mean())


def tune_log_bias(oof, y, prior, cc):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = fast_bal_acc(y, (log_oof + bias).argmax(axis=1), cc)
    grid_def = np.linspace(-3.0, 3.0, 61)
    grid_hi = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = grid_hi if k == 2 else grid_def
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(axis=1), cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def per_class_recall(y, pred, cc):
    m = pred == y
    return {CLASSES[k]: float(m[y == k].sum() / cc[k]) for k in range(3)}


def log_blend_weighted(oofs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, 1e-9, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def write_submission(test: np.ndarray, bias: np.ndarray, te_ids: np.ndarray,
                     fname: str) -> None:
    pred = (np.log(np.clip(test, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred]}).to_csv(
        SUB / fname, index=False
    )
    print(f"  wrote {fname}")


def evaluate(oof_blend: np.ndarray, y: np.ndarray, prior: np.ndarray,
             cc: np.ndarray, label: str):
    bias, tuned = tune_log_bias(oof_blend, y, prior, cc)
    pred = (np.log(np.clip(oof_blend, 1e-9, 1.0)) + bias).argmax(axis=1)
    pcr = per_class_recall(y, pred, cc)
    print(f"  {label:40s}  bal={tuned:.5f}  "
          f"rec_L={pcr['Low']:.4f} rec_M={pcr['Medium']:.4f} rec_H={pcr['High']:.4f}  "
          f"bias={np.round(bias, 3).tolist()}")
    return bias, tuned, pcr


def main():
    tr = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    cc = np.bincount(y, minlength=3)

    oofs = {}
    tests = {}
    for name in ["lgbm_dgp", "xgb_dist", "xgb_dist_routed_v3",
                 "xgb_spec_678", "xgb_hybrid_v3"]:
        oofs[name] = np.load(ART / f"oof_{name}.npy")
        tests[name] = np.load(ART / f"test_{name}.npy")
        print(f"loaded {name}")

    # reference
    print("\n--- reference ---")
    evaluate(oofs["xgb_hybrid_v3"], y, prior, cc, "xgb_hybrid_v3 standalone")

    # greedy path
    print("\n--- greedy forward (reproduced) ---")
    # step 1: hybrid_v3 + routed_v3, log-blend
    for w_routed in [0.3, 0.4, 0.5]:
        oof_blend = log_blend_weighted(
            [oofs["xgb_hybrid_v3"], oofs["xgb_dist_routed_v3"]],
            [1.0 - w_routed, w_routed])
        evaluate(oof_blend, y, prior, cc,
                 f"greedy step1: hybrid_v3 + routed_v3(w={w_routed:.1f})")
    # step 2: + spec_678
    print("\n--- greedy final: hybrid_v3 + routed_v3 + spec_678 ---")
    best = {"tuned": -1.0}
    for w_r, w_s in [(0.35, 0.05), (0.40, 0.10), (0.45, 0.10),
                     (0.40, 0.15), (0.35, 0.10), (0.30, 0.10),
                     (0.30, 0.15), (0.50, 0.10)]:
        w_h = 1.0 - w_r - w_s
        if w_h <= 0:
            continue
        oof_b = log_blend_weighted(
            [oofs["xgb_hybrid_v3"], oofs["xgb_dist_routed_v3"], oofs["xgb_spec_678"]],
            [w_h, w_r, w_s])
        test_b = log_blend_weighted(
            [tests["xgb_hybrid_v3"], tests["xgb_dist_routed_v3"], tests["xgb_spec_678"]],
            [w_h, w_r, w_s])
        bias, tuned = tune_log_bias(oof_b, y, prior, cc)
        pred = (np.log(np.clip(oof_b, 1e-9, 1.0)) + bias).argmax(axis=1)
        pcr = per_class_recall(y, pred, cc)
        print(f"  greedy w=({w_h:.2f},{w_r:.2f},{w_s:.2f})  bal={tuned:.5f}  "
              f"rec_H={pcr['High']:.4f}  bias={np.round(bias, 3).tolist()}")
        if tuned > best["tuned"]:
            best = {"tuned": tuned, "w": (w_h, w_r, w_s),
                    "oof": oof_b, "test": test_b, "bias": bias,
                    "recall": pcr}

    # also the pairwise best (routed_v3 × hybrid_v3 α=0.5)
    print("\n--- best pair: xgb_routed_v3 × xgb_hybrid_v3 ---")
    oof_pair = log_blend_weighted(
        [oofs["xgb_hybrid_v3"], oofs["xgb_dist_routed_v3"]], [0.5, 0.5])
    test_pair = log_blend_weighted(
        [tests["xgb_hybrid_v3"], tests["xgb_dist_routed_v3"]], [0.5, 0.5])
    bias_pair, tuned_pair, pcr_pair = evaluate(oof_pair, y, prior, cc,
                                               "pair: hybrid_v3 + routed_v3 (50/50, log)")

    # meta-stack
    print("\n--- logistic meta-stacker (class_weight=balanced) ---")
    stack_names = ["lgbm_dgp", "xgb_dist", "xgb_dist_routed_v3", "xgb_hybrid_v3"]
    X_oof = np.concatenate([oofs[n] for n in stack_names], axis=1)
    X_te = np.concatenate([tests[n] for n in stack_names], axis=1)
    meta_oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for f, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y)):
        lr = LogisticRegression(solver="lbfgs", C=1.0,
                                class_weight="balanced", max_iter=1000, n_jobs=1)
        lr.fit(X_oof[tr_idx], y[tr_idx])
        meta_oof[va_idx] = lr.predict_proba(X_oof[va_idx])
        print(f"  meta fold {f+1}/5 done")
    lr_full = LogisticRegression(solver="lbfgs", C=1.0,
                                 class_weight="balanced", max_iter=1000, n_jobs=1)
    lr_full.fit(X_oof, y)
    meta_te = lr_full.predict_proba(X_te)
    bias_m, tuned_m, pcr_m = evaluate(meta_oof, y, prior, cc, "meta-stack (LR class-balanced)")

    # final summary
    print("\n\n=== CANDIDATES SUMMARY (gated by >= 3e-4 above hybrid_v3 0.97352) ===")
    threshold = 0.97352 + 3e-4
    cands = [
        ("greedy_w_best",    best["tuned"],  best["oof"],  best["test"],  best["bias"],  best["recall"]),
        ("pair_hybrid_routed", tuned_pair, oof_pair, test_pair, bias_pair, pcr_pair),
        ("meta_stack",       tuned_m,    meta_oof,    meta_te,     bias_m,     pcr_m),
    ]
    cands.sort(key=lambda r: r[1], reverse=True)
    for name, sc, o, t, b, rec in cands:
        flag = "✓" if sc >= threshold else " "
        print(f"  {flag}  {name:25s}  bal={sc:.5f}  Δ={sc-0.97352:+.5f}  "
              f"rec_H={rec['High']:.4f}")

    # write submission for candidates that clear the threshold
    for name, sc, o, t, b, _ in cands:
        if sc < threshold:
            continue
        fname = f"submission_blend_{name}.csv"
        write_submission(t, b, te_ids, fname)

    # also dump greedy test probs (needed for further blending)
    np.save(ART / "oof_blend_greedy_final.npy", best["oof"])
    np.save(ART / "test_blend_greedy_final.npy", best["test"])
    print(f"\n  saved oof_blend_greedy_final.npy, test_blend_greedy_final.npy")

    with open(ART / "blend_greedy_finalize_results.json", "w") as f:
        json.dump({
            "hybrid_v3_standalone": 0.97352,
            "greedy_best_w":  {"w_hybrid": best["w"][0], "w_routed": best["w"][1],
                               "w_spec": best["w"][2], "tuned": best["tuned"],
                               "bias": best["bias"].tolist(),
                               "rec_H": best["recall"]["High"]},
            "pair_hybrid_routed_50_50": {"tuned": tuned_pair,
                                          "bias": bias_pair.tolist(),
                                          "rec_H": pcr_pair["High"]},
            "meta_stack": {"tuned": tuned_m,
                           "components": stack_names,
                           "bias": bias_m.tolist(),
                           "rec_H": pcr_m["High"]},
        }, f, indent=2)
    print("\n--- done; NO auto-submission to LB per CLAUDE.md rule ---")


if __name__ == "__main__":
    main()
