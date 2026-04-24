"""C0 + P5 combined: per-class isotonic calibration + greedy forward-blend.

Step 1: for each component OOF, fit per-class IsotonicRegression on
OOF → y (out-of-fold on the saved OOF probs), apply to test probs,
renormalize rows to sum to 1. This gives each component a calibrated
twin.

Step 2: greedy forward-selection anchored on recipe_full_te, using
FIXED recipe bias [1.43, 1.47, 3.40] throughout. Candidates include
both raw and isotonic-calibrated variants. Stop when no candidate
improves OOF by at least 1e-4.

Uses log-blend (exponentially weighted average in log-prob space)
because it's the method verified to transfer to LB in the 2026-04-23
greedy_full_bank 6-way (OOF 0.97552 → LB 0.97581).

Output:
  scripts/artifacts/c0_isotonic_greedy_results.json
  scripts/artifacts/oof_c0_greedy.npy       (final blend OOF)
  scripts/artifacts/test_c0_greedy.npy      (final blend test)
  submissions/submission_c0_greedy.csv      (if OOF > 0.98013 + 1e-4)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
LB_BEST_OOF = 0.98013  # 50/50 log-blend of recipe + pseudolabel, tuned bias

# Candidates: non-composite component OOFs only (skip blends-of-blends).
# Names must have corresponding oof_<name>.npy + test_<name>.npy on disk.
CANDIDATES = [
    "recipe_full_te",
    "recipe_pseudolabel",
    "recipe_pseudolabel_stage2",
    "recipe_allpairs",
    "recipe_catboost",
    "recipe_lgbm",
    "recipe_171pair",
    "recipe_full_te_a01",
    "recipe_full_te_a10",
    "recipe_full_te_catboost",
    "recipe_full_te_lgbm",
    "recipe_full_te_cldrop",
    "recipe_no_ote",
    "recipe_no_digits",
    "recipe_no_combos",
    "recipe_no_orig",
    "em_uniform",
    "xgb_corn",
    "xgb_nonrule",
    "xgb_dist_digits",
    "lgbm_dist_digits",
    "xgb_dist_digits_ote",
    "xgb_dist_digits_ote_digits",
    "xgb_dist_digits_ote_digits_pairs",
    "xgb_dist_digits_ote_digits_light",
    "xgb_dist_digits_ote_light",
    "lgbm_dist_digits_ote",
    "xgb_dist_routed_v3",
    "xgb_spec_678",
    "xgb_vanilla_dist",
    "catboost_optuna",
    "catboost_recipe_gpu",
    "extratrees_dist_digits",
    "extratrees_dist_digits_v2",
    "lgbm_competitor",
    "lgbm_te_orig",
    "soft_distill",
    "tabpfn",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_blend(probs_list, weights):
    """Weighted log-blend of a list of (N,3) prob matrices.
    Weights must be non-negative; will be normalized to sum to 1."""
    eps = 1e-12
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    log_p = np.zeros_like(probs_list[0], dtype=np.float64)
    for w, p in zip(weights, probs_list):
        log_p += w * np.log(np.clip(p, eps, 1.0))
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    ez = np.exp(log_p)
    return (ez / ez.sum(axis=1, keepdims=True)).astype(np.float32)


def bal_acc_at_recipe_bias(probs, y):
    eps = 1e-12
    log_p = np.log(np.clip(probs, eps, 1.0)) + RECIPE_BIAS
    return balanced_accuracy_score(y, log_p.argmax(1))


def isotonic_calibrate(oof, test, y):
    """Per-class isotonic, then renormalize rows."""
    oof_cal = np.zeros_like(oof, dtype=np.float32)
    test_cal = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oof_cal[:, c] = ir.predict(oof[:, c])
        test_cal[:, c] = ir.predict(test[:, c])
    # Renormalize rows.
    oof_cal = oof_cal / np.clip(oof_cal.sum(axis=1, keepdims=True), 1e-9, None)
    test_cal = test_cal / np.clip(test_cal.sum(axis=1, keepdims=True), 1e-9, None)
    return oof_cal.astype(np.float32), test_cal.astype(np.float32)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    N = len(y)

    # Load + validate + isotonic-calibrate every candidate.
    components = {}
    skipped = []
    log("Loading + validating + calibrating components...")
    for name in CANDIDATES:
        oof_p = ART / f"oof_{name}.npy"
        test_p = ART / f"test_{name}.npy"
        if not oof_p.exists() or not test_p.exists():
            skipped.append((name, "missing files"))
            continue
        oof = np.load(oof_p)
        test = np.load(test_p)
        if oof.shape != (N, 3):
            skipped.append((name, f"oof shape {oof.shape}"))
            continue
        # Ensure probs (some might be logits)
        if oof.min() < -1e-6 or oof.max() > 1 + 1e-6:
            skipped.append((name, f"oof out of [0,1]: min={oof.min()} max={oof.max()}"))
            continue
        row_sums = oof.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            # Renormalize and continue
            oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-9, None)
            test = test / np.clip(test.sum(axis=1, keepdims=True), 1e-9, None)

        arg_raw = balanced_accuracy_score(y, oof.argmax(1))
        bal_raw = bal_acc_at_recipe_bias(oof, y)
        oof_cal, test_cal = isotonic_calibrate(oof, test, y)
        arg_cal = balanced_accuracy_score(y, oof_cal.argmax(1))
        bal_cal = bal_acc_at_recipe_bias(oof_cal, y)

        components[name] = dict(oof=oof.astype(np.float32),
                                test=test.astype(np.float32),
                                oof_iso=oof_cal,
                                test_iso=test_cal,
                                arg_raw=arg_raw, bal_raw=bal_raw,
                                arg_cal=arg_cal, bal_cal=bal_cal)
        print(f"  {name:35s}  raw: arg={arg_raw:.4f} bias={bal_raw:.5f}  "
              f"iso: arg={arg_cal:.4f} bias={bal_cal:.5f}  "
              f"Δiso={bal_cal - bal_raw:+.5f}")
    log(f"Loaded {len(components)} components; skipped {len(skipped)}:")
    for name, reason in skipped:
        print(f"  SKIP {name}: {reason}")

    # ------------------ Step 2: greedy forward-selection -----------------
    # Candidate pool: both raw and iso versions as separate keys.
    pool = {}
    for name, c in components.items():
        pool[f"{name}"] = (c["oof"], c["test"], c["bal_raw"])
        pool[f"{name}__iso"] = (c["oof_iso"], c["test_iso"], c["bal_cal"])

    # Anchor: recipe_full_te. Log-bias is FIXED at recipe bias.
    anchor_name = "recipe_full_te"
    oof_cur = components[anchor_name]["oof"].copy()
    test_cur = components[anchor_name]["test"].copy()
    chosen = [(anchor_name, 1.0)]
    bal_cur = bal_acc_at_recipe_bias(oof_cur, y)
    log(f"Anchor = {anchor_name}  bal@recipe_bias = {bal_cur:.5f}")

    alphas = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    picked_names = {anchor_name}
    for step in range(1, 9):
        best = None
        for key, (oof_k, test_k, _) in pool.items():
            base_name = key.replace("__iso", "")
            if base_name in picked_names:
                continue
            # Find best alpha for blending this candidate with current blend.
            for a in alphas:
                oof_try = log_blend([oof_cur, oof_k], [1.0 - a, a])
                sc = bal_acc_at_recipe_bias(oof_try, y)
                if best is None or sc > best[0]:
                    best = (sc, key, base_name, a, oof_try, test_k)
        if best is None:
            break
        sc, key, base, a, oof_try, test_k = best
        delta = sc - bal_cur
        print(f"  step {step}: + {key:45s}  α={a:4.3f}  OOF={sc:.5f}  Δ={delta:+.5f}")
        if delta < 1e-4:
            log("  no improvement ≥ 1e-4 — stopping")
            break
        chosen.append((key, a))
        picked_names.add(base)
        # Update current: rebuild from all chosen with their effective weights.
        # Effective log-space weight for step i with weight a_i:
        #   w_0 = prod(1-a_i)
        #   w_k = a_k * prod_{j>k}(1-a_j)
        # This matches the sequential application of log_blend.
        oof_cur = log_blend([oof_cur, oof_k], [1.0 - a, a])
        test_cur = log_blend([test_cur, test_k], [1.0 - a, a])
        bal_cur = sc

    log(f"Final OOF bal@recipe_bias = {bal_cur:.5f}  "
        f"(Δ vs LB-best {bal_cur - LB_BEST_OOF:+.5f})")
    # Per-class recall
    from sklearn.metrics import confusion_matrix
    pred = (np.log(np.clip(oof_cur, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
    cm = confusion_matrix(y, pred)
    recall = cm.diagonal() / cm.sum(axis=1)
    print(f"  per-class recall: Low={recall[0]:.4f}  "
          f"Medium={recall[1]:.4f}  High={recall[2]:.4f}")

    # Save
    np.save(ART / "oof_c0_greedy.npy", oof_cur.astype(np.float32))
    np.save(ART / "test_c0_greedy.npy", test_cur.astype(np.float32))
    results = dict(
        anchor=anchor_name,
        chosen=[(n, float(a)) for n, a in chosen],
        final_oof_bal_acc=float(bal_cur),
        delta_vs_lb_best=float(bal_cur - LB_BEST_OOF),
        per_class_recall=dict(Low=float(recall[0]),
                              Medium=float(recall[1]),
                              High=float(recall[2])),
        n_components_loaded=len(components),
        skipped=[list(s) for s in skipped],
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "c0_isotonic_greedy_results.json").write_text(json.dumps(results, indent=2))
    log(f"Wrote c0_isotonic_greedy_results.json in {time.time() - t0:.1f}s")

    # Build submission if we beat LB-best OOF by ≥ 1e-4
    if bal_cur > LB_BEST_OOF + 1e-4:
        eps = 1e-12
        pred_test = (np.log(np.clip(test_cur, eps, 1)) + RECIPE_BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = pd.DataFrame({
            "id": sample["id"].values,
            "Irrigation_Need": ["Low", "Medium", "High"],
        })
        sub["Irrigation_Need"] = [["Low", "Medium", "High"][i] for i in pred_test]
        sub_path = SUB / "submission_c0_greedy.csv"
        sub.to_csv(sub_path, index=False)
        log(f"Wrote {sub_path} dist={dict(sub['Irrigation_Need'].value_counts())}")


if __name__ == "__main__":
    main()
