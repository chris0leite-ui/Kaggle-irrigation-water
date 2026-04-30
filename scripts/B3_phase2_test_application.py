"""B3 phase 2: apply flip-specialist to test, build candidate, 4-gate.

Phase 1 finding: flip-specialist trained on 10,304 rule-violator rows
achieves 99.14% OOF accuracy. Per-direction recall: L->M 100%, H->M 99.8%,
M->H 98.0%, M->L 97.0%.

Phase 2 mechanism:
  1. Retrain specialist on FULL 10,304 violators (no holdout)
  2. Predict on FULL train (630k) AND test (270k)
  3. Calibration check: on TRAIN non-violators, what fraction does specialist
     INCORRECTLY flag as violator (i.e., specialist != rule prediction)?
  4. If FP rate is low (e.g., < 1%), build candidate: flip 4b where specialist
     disagrees with both 4b and rule with high confidence.
  5. 4-gate per direction on TRAIN OOF.

Distinct from W13 (binary "is wrong"): this is 3-class output, telling us
which class to flip to.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb


ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).to_numpy()


def dgp_rule(score):
    pred = np.full_like(score, 1, dtype=np.int8)
    pred[score <= 3] = 0
    pred[score >= 7] = 2
    return pred


def csv_to_argmax(path: Path) -> np.ndarray:
    s = pd.read_csv(path)["Irrigation_Need"]
    return s.map(LMH_REV).to_numpy(dtype=np.int8)


def main():
    t0 = time.time()
    print("=== B3 phase 2: apply flip-specialist to test ===\n")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].to_numpy()

    y_full = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    score_train = compute_dgp_score(train)
    rule_train = dgp_rule(score_train)
    score_test = compute_dgp_score(test)
    rule_test = dgp_rule(score_test)

    # Identify violators
    viol_mask_tr = rule_train != y_full
    n_viol = int(viol_mask_tr.sum())
    print(f"TRAIN violators: {n_viol} ({n_viol/len(train)*100:.2f}%)")
    print(f"TEST size: {len(test)}, expected violators ~{int(len(test)*0.0164)}")

    # Train specialist on FULL violators (no holdout, max signal)
    feature_cols = [c for c in train.columns if c not in ["id", "Irrigation_Need"]]
    X_tr_v = train.loc[viol_mask_tr, feature_cols].copy()
    y_tr_v = y_full[viol_mask_tr]
    X_train_full = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = X_tr_v.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_tr_v[c] = X_tr_v[c].astype("category")
        X_train_full[c] = X_train_full[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print(f"\nTraining specialist on {len(X_tr_v)} violators (full data)...")
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=15, subsample=0.85, colsample_bytree=0.85,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr_v, y_tr_v)
    print(f"  trained: {model.n_estimators_} trees")

    # Predict on full train + test
    p_train = model.predict_proba(X_train_full)
    p_test = model.predict_proba(X_test)
    am_train = p_train.argmax(1).astype(np.int8)
    am_test = p_test.argmax(1).astype(np.int8)
    conf_train = p_train.max(axis=1)
    conf_test = p_test.max(axis=1)

    # === Calibration check: behavior on TRAIN non-violators ===
    print(f"\n=== TRAIN: behavior on non-violators ===")
    nonv_mask = ~viol_mask_tr
    n_nonv = int(nonv_mask.sum())
    print(f"  TRAIN non-violators: {n_nonv}")
    # Specialist agrees with rule on non-violators?
    agree_with_rule = (am_train[nonv_mask] == rule_train[nonv_mask])
    p_agree = float(agree_with_rule.mean())
    print(f"  Specialist == rule on non-violators: {p_agree:.4f}")
    # Specialist agrees with TRUE label on non-violators?
    agree_with_truth = (am_train[nonv_mask] == y_full[nonv_mask])
    p_truth = float(agree_with_truth.mean())
    print(f"  Specialist == true label on non-violators: {p_truth:.4f}")

    # By confidence threshold: false positive rate (specialist != rule on non-violator)
    print(f"\n  False-positive (specialist != rule on non-violators) by confidence threshold:")
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        m = nonv_mask & (conf_train >= thresh)
        n = int(m.sum())
        if n == 0: continue
        fp_rate = float((am_train[m] != rule_train[m]).mean())
        # Of those flagged, how many actually have true_label == specialist_argmax?
        flagged = m & (am_train != rule_train)
        n_flagged = int(flagged.sum())
        if n_flagged > 0:
            true_pos = float((am_train[flagged] == y_full[flagged]).mean())
        else:
            true_pos = float("nan")
        print(f"    conf >= {thresh:.2f}: n_eligible={n:>7}, FP_rate={fp_rate:.4f}, "
              f"n_flagged_as_violator={n_flagged}, P(specialist correct | flagged)={true_pos:.4f}")

    # === Apply to TEST ===
    print(f"\n=== TEST: candidate flip rows ===")
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")

    # Specialist disagrees with rule on test
    specialist_disagree_rule = am_test != rule_test
    print(f"Specialist disagrees with rule on test: {int(specialist_disagree_rule.sum())} rows")
    print(f"  (expected ~{int(len(test)*0.0164)} = ~1.64% if calibrated)")

    # Sweep thresholds and report direction breakdown
    print(f"\nFlip candidates by confidence threshold (specialist != 4b AND conf >= τ):")
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        flip_mask = specialist_disagree_rule & (am_test != fb) & (conf_test >= thresh)
        n = int(flip_mask.sum())
        if n == 0: continue
        # Direction breakdown (4b -> specialist class)
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                d = int(((fb == fr) & (am_test == to) & flip_mask).sum())
                if d > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = d
        print(f"  conf >= {thresh:.2f}: n={n:>5}  dirs={dirs}")

    # Build candidate at conf >= 0.85
    THRESH = 0.85
    flip_mask = specialist_disagree_rule & (am_test != fb) & (conf_test >= THRESH)
    n_flips = int(flip_mask.sum())
    new_pred = fb.copy()
    new_pred[flip_mask] = am_test[flip_mask]

    # Direction projection per direction (use specialist OOF accuracy as precision estimate)
    # Specialist OOF per-direction accuracy on violators:
    SPECIALIST_PRECISION_ON_VIOLATORS = {
        "L->M": 1.000, "L->H": 0.95,    # only if flip exists (rare)
        "M->L": 0.970, "M->H": 0.980,
        "H->M": 0.998, "H->L": 0.95,
    }
    # But on test side, we ALSO need to discount by FP rate (specialist falsely flags non-violators)
    # which depends on conf threshold

    print(f"\n=== Candidate CSV ===")
    print(f"  Threshold: conf >= {THRESH}")
    print(f"  Flips: {n_flips}")
    if n_flips > 0:
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                d = int(((fb == fr) & (am_test == to) & flip_mask).sum())
                if d > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = d
        print(f"  Directions (4b -> specialist): {dirs}")

        # Save candidate
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
        })
        out_csv = SUB / f"submission_B3_flip_specialist_t{int(THRESH*100)}.csv"
        sub.to_csv(out_csv, index=False)
        print(f"  Emitted: {out_csv}")

    # Save specialist test predictions for downstream use
    np.save(ART / "B3_test_specialist.npy", p_test)
    np.save(ART / "B3_train_specialist.npy", p_train)
    print(f"\nSaved test/train specialist arrays")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
