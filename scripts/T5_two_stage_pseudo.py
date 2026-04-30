"""T5 — 2-stage test-time pseudo-labeling refinement.

Stage 1: Identify test rows where 14-bank components are high-consensus
         (>= 12/14 agree). Lock pseudo-label = bank-majority on those rows.
Stage 2: Train a small LightGBM on (real_train + stage1_pseudo). Predict
         ONLY the remaining ~50k low-consensus test rows.
Final:   Stage 1 frozen + Stage 2 prediction on the remainder.

Validation: 5-fold on TRAIN — fold-out OOF macro-recall must beat single-stage
4b proxy (B's 0.980816).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features  # noqa: E402
from T2_conformal_helpers import load_bank  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

CONSENSUS_THRESHOLD = 12  # of 14


def macro_recall(y_true, y_pred):
    rec = []
    for c in range(3):
        m = y_true == c
        if m.sum() == 0:
            continue
        rec.append(float((y_pred[m] == c).mean()))
    return sum(rec) / len(rec)


def main():
    import lightgbm as lgb

    print(f"=== T5 — 2-stage test-time pseudo-labeling (consensus >= {CONSENSUS_THRESHOLD}/14) ===\n")

    # Load test bank, compute consensus
    test_bank = load_bank("test")
    test_argmax = test_bank.argmax(axis=2)  # (14, 270000)
    test_majority = mode(test_argmax, axis=0, keepdims=False).mode
    test_agree_count = (test_argmax == test_majority).sum(axis=0)  # (270000,)

    consensus_mask_test = test_agree_count >= CONSENSUS_THRESHOLD
    n_consensus_test = int(consensus_mask_test.sum())
    n_remainder_test = int((~consensus_mask_test).sum())
    print(f"Test consensus rows (>={CONSENSUS_THRESHOLD}/14): {n_consensus_test}")
    print(f"Test remainder (boundary): {n_remainder_test}")
    print(f"Consensus class dist: {np.bincount(test_majority[consensus_mask_test], minlength=3).tolist()}")

    # Load test data + features for stage 2 model
    test_df = pd.read_csv(DATA / "test.csv")
    train_df = pd.read_csv(DATA / "train.csv")
    test_d = add_distance_features(test_df)
    train_d = add_distance_features(train_df)

    feat_cols = [
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "Soil_pH", "Humidity", "Sunlight_Hours", "Organic_Carbon",
        "Electrical_Conductivity", "Field_Area_hectare", "Previous_Irrigation_mm",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "min_boundary_dist", "min_axis_abs", "dgp_score",
        "score_dist_low_mid", "score_dist_mid_high",
    ]
    cat_cols = [
        "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Mulching_Used",
        "Irrigation_Type", "Water_Source", "Region", "Season",
    ]
    for c in cat_cols:
        train_d[c] = pd.Categorical(train_d[c]).codes
        test_d[c] = pd.Categorical(test_d[c]).codes

    Xtr_all = train_d[feat_cols + cat_cols].to_numpy(dtype=np.float32)
    Xte_all = test_d[feat_cols + cat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)

    # ---- TRAIN OOF VALIDATION ----
    # Mechanism: 5-fold on TRAIN. Within each fold:
    #   - val rows act as "test"
    #   - apply same stage-1 (consensus from bank OOF; consensus is >= 12/14 agree)
    #   - train stage-2 LGBM on tr_idx + stage1_pseudo (val rows where consensus
    #     applies); predict val_idx remainder
    # Final OOF prediction = consensus on val rows where applicable + stage-2
    # on remainder. Compare to B's OOF macro 0.980816.

    oof_bank = load_bank("oof")
    oof_argmax = oof_bank.argmax(axis=2)
    oof_majority = mode(oof_argmax, axis=0, keepdims=False).mode
    oof_agree_count = (oof_argmax == oof_majority).sum(axis=0)
    consensus_mask_oof = oof_agree_count >= CONSENSUS_THRESHOLD
    print(f"\nTRAIN OOF consensus rows: {int(consensus_mask_oof.sum())}/{len(oof_majority)}")

    # 4b OOF analog for baseline comparison
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    t1_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)

    def tune_bias(p, y, n_steps=21):
        best = np.zeros(3, dtype=np.float64)
        log_p = np.log(np.clip(p, 1e-9, None))
        best_score = macro_recall(y, (log_p + best).argmax(1))
        for _ in range(3):
            improved = False
            for c in range(3):
                grid = np.linspace(best[c] - 1.0, best[c] + 4.0, n_steps)
                for v in grid:
                    trial = best.copy(); trial[c] = v
                    s = macro_recall(y, (log_p + trial).argmax(1))
                    if s > best_score:
                        best_score = s; best = trial; improved = True
            if not improved: break
        return best, best_score

    bv1, _ = tune_bias(v1_oof / v1_oof.sum(1, keepdims=True), y_train)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1)

    bra, _ = tune_bias(raw_oof / raw_oof.sum(1, keepdims=True), y_train)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1)

    bt1, _ = tune_bias(t1_oof / t1_oof.sum(1, keepdims=True), y_train)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1)

    una = (a_ra == a_t1)
    fb_oof = a_v1.copy()
    om = una & (a_v1 != a_ra)
    fb_oof[om] = a_ra[om]
    base_macro = macro_recall(y_train, fb_oof)
    print(f"4b OOF analog macro: {base_macro:.6f}")

    # 5-fold validation of T5 mechanism
    oof_t5 = np.zeros(len(y_train), dtype=np.int8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nRunning 5-fold T5 validation...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_all, y_train)):
        # Stage 1: lock consensus on val_idx
        va_consensus = consensus_mask_oof[va_idx]
        n_va_cons = int(va_consensus.sum())
        n_va_rem = len(va_idx) - n_va_cons

        # Stage 2: train LGBM on tr_idx + (val_idx & consensus) with pseudo-labels
        # = bank-majority on those rows.
        pseudo_mask_va = va_idx[va_consensus]
        pseudo_y = oof_majority[pseudo_mask_va].astype(np.int8)

        X_stage2 = np.vstack([Xtr_all[tr_idx], Xtr_all[pseudo_mask_va]])
        y_stage2 = np.concatenate([y_train[tr_idx], pseudo_y])

        model = lgb.LGBMClassifier(
            num_leaves=15, n_estimators=200, learning_rate=0.05,
            objective="multiclass", num_class=3, n_jobs=-1, verbosity=-1,
            random_state=42, class_weight="balanced",
        )
        model.fit(X_stage2, y_stage2, categorical_feature=list(range(len(feat_cols), len(feat_cols)+len(cat_cols))))

        # Predict val_idx
        va_pred = model.predict(Xtr_all[va_idx])

        # Final OOF on val_idx: consensus where applies + LGBM elsewhere
        oof_t5[va_idx] = va_pred
        oof_t5[pseudo_mask_va] = pseudo_y  # frozen consensus

        print(f"  fold {fold+1}: {n_va_cons} consensus + {n_va_rem} remainder")

    new_macro = macro_recall(y_train, oof_t5)
    print(f"\nT5 OOF macro:        {new_macro:.6f}")
    print(f"4b OOF baseline:     {base_macro:.6f}")
    print(f"delta:               {new_macro - base_macro:+.6f}")

    if new_macro > base_macro + 1e-4:
        print("\n  T5 OOF lift > 1e-4: build test-side candidate")
        # Train final stage-2 on all TRAIN + test consensus pseudo-labels, predict test remainder
        pseudo_test_idx = np.where(consensus_mask_test)[0]
        pseudo_test_y = test_majority[pseudo_test_idx].astype(np.int8)
        X_full = np.vstack([Xtr_all, Xte_all[pseudo_test_idx]])
        y_full = np.concatenate([y_train, pseudo_test_y])
        model = lgb.LGBMClassifier(
            num_leaves=15, n_estimators=200, learning_rate=0.05,
            objective="multiclass", num_class=3, n_jobs=-1, verbosity=-1,
            random_state=42, class_weight="balanced",
        )
        model.fit(X_full, y_full, categorical_feature=list(range(len(feat_cols), len(feat_cols)+len(cat_cols))))
        test_pred_remainder = model.predict(Xte_all[~consensus_mask_test])
        final_test_pred = np.zeros(len(Xte_all), dtype=np.int8)
        final_test_pred[consensus_mask_test] = test_majority[consensus_mask_test]
        final_test_pred[~consensus_mask_test] = test_pred_remainder

        # Compare to 4b
        fb = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")
        fb_argmax = fb["Irrigation_Need"].map({"Low":0,"Medium":1,"High":2}).to_numpy()
        diff = int((final_test_pred != fb_argmax).sum())
        print(f"  Diff vs 4b: {diff}")

        out_csv = SUB / "submission_T5_two_stage_pseudo.csv"
        pd.DataFrame({
            "id": test_df["id"].to_numpy(),
            "Irrigation_Need": pd.Series(final_test_pred).map({0: "Low", 1: "Medium", 2: "High"}),
        }).to_csv(out_csv, index=False)
        print(f"  emitted: {out_csv}")
    else:
        print("\n  T5 OOF lift <= 1e-4: no test candidate emitted")

    out = ART / "T5_two_stage_pseudo_results.json"
    out.write_text(json.dumps({
        "consensus_threshold": CONSENSUS_THRESHOLD,
        "n_consensus_test": n_consensus_test,
        "n_consensus_oof": int(consensus_mask_oof.sum()),
        "oof_macro_4b_baseline": float(base_macro),
        "oof_macro_t5": float(new_macro),
        "oof_delta": float(new_macro - base_macro),
    }, indent=2))


if __name__ == "__main__":
    main()
