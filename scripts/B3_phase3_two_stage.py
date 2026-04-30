"""B3 phase 3: 2-stage flip detection (gate + specialist).

Phase 2 finding: B3 specialist alone is OOD on 98.4% non-violator rows
(99% FP rate). Need a stage-1 gate to identify likely violators before
applying specialist.

Stage 1: binary classifier P(rule != label) on FULL 630k train.
Stage 2: 3-class flip-specialist (already trained in B3 phase 2).
Decision: flip 4b at row r iff
  P(violator)(r) >= τ_v  AND
  specialist_argmax(r) != 4b_argmax(r) AND
  specialist_conf(r) >= τ_s

Tune τ_v and τ_s on TRAIN OOF using the 4-gate criteria per direction.

Distinct from W13 which used bank-majority as the override target. Here
the override target is the specialist's own 3-class output.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}

BREAK_EVEN = {
    "H->M": 100367 / (100367 + 10174),
    "H->L": 159459 / (159459 + 10174),
    "M->H": 10174 / (10174 + 100367),
    "M->L": 159459 / (159459 + 100367),
    "L->H": 10174 / (10174 + 159459),
    "L->M": 100367 / (100367 + 159459),
}


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
    print("=== B3 phase 3: 2-stage flip detection ===\n")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].to_numpy()

    y_full = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    score_train = compute_dgp_score(train)
    rule_train = dgp_rule(score_train)
    score_test = compute_dgp_score(test)
    rule_test = dgp_rule(score_test)
    is_violator_train = (rule_train != y_full).astype(np.int8)

    feature_cols = [c for c in train.columns if c not in ["id", "Irrigation_Need"]]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # === STAGE 1: binary P(violator) classifier, 5-fold OOF ===
    print("Stage 1: binary P(violator) classifier, 5-fold OOF...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_viol_oof = np.zeros(len(X_train), dtype=np.float32)
    p_viol_test_folds = np.zeros((5, len(X_test)), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X_train, is_violator_train)):
        Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
        ytr_v, yva_v = is_violator_train[tr], is_violator_train[va]
        # Use class weights to balance the 1.6% positive class
        n_neg = (ytr_v == 0).sum()
        n_pos = (ytr_v == 1).sum()
        spw = n_neg / max(n_pos, 1)
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=63,
            min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=spw, random_state=42 + fold, verbose=-1, n_jobs=-1,
        )
        model.fit(Xtr, ytr_v, eval_set=[(Xva, yva_v)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        p_viol_oof[va] = model.predict_proba(Xva)[:, 1]
        p_viol_test_folds[fold] = model.predict_proba(X_test)[:, 1]
        print(f"  fold {fold+1}: best_iter={model.best_iteration_}")
    p_viol_test = p_viol_test_folds.mean(axis=0)

    # AUC on TRAIN OOF
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(is_violator_train, p_viol_oof)
    print(f"\n  Stage-1 TRAIN OOF AUC: {auc:.4f}")

    # === STAGE 2: load B3 specialist test predictions (computed in phase 2) ===
    p_spec_test = np.load(ART / "B3_test_specialist.npy")
    p_spec_train = np.load(ART / "B3_train_specialist.npy")
    am_spec_test = p_spec_test.argmax(1).astype(np.int8)
    am_spec_train = p_spec_train.argmax(1).astype(np.int8)
    conf_spec_test = p_spec_test.max(axis=1)
    conf_spec_train = p_spec_train.max(axis=1)

    # === Gate sweep on TRAIN OOF ===
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")

    # B-on-OOF analog: tier1b argmax (since 4b is mostly B + 108 flips)
    tier1b_oof = np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)
    tier1b_oof = tier1b_oof / np.clip(tier1b_oof.sum(1, keepdims=True), 1e-9, None)
    tier1b_am = tier1b_oof.argmax(1).astype(np.int8)

    print(f"\n=== Gate sweep on TRAIN OOF (anchor=tier1b_am, target=specialist_argmax) ===")
    print(f"For each (τ_v, τ_s), compute n_flips on TRAIN OOF and per-direction precision.\n")

    best_config = None
    best_macro_delta = -np.inf
    N_L, N_M, N_H = 159459, 100367, 10174
    Ns = [N_L, N_M, N_H]

    for tv in [0.50, 0.70, 0.85, 0.90, 0.95]:
        for ts in [0.50, 0.70, 0.85, 0.90, 0.95]:
            mask = (p_viol_oof >= tv) & (am_spec_train != tier1b_am) & (conf_spec_train >= ts)
            n = int(mask.sum())
            if n == 0: continue

            # Per-direction precision and projection
            macro_delta = 0.0
            dirs_str_parts = []
            all_pass = True
            for fr in range(3):
                for to in range(3):
                    if fr == to: continue
                    d_mask = mask & (tier1b_am == fr) & (am_spec_train == to)
                    d_n = int(d_mask.sum())
                    if d_n == 0: continue
                    p_correct = (y_full[d_mask] == to).mean()
                    d_label = f"{LMH[fr]}->{LMH[to]}"
                    be = BREAK_EVEN[d_label]
                    pass_str = "✓" if p_correct >= be else "✗"
                    if p_correct < be: all_pass = False
                    macro_delta += d_n * (p_correct / Ns[to] - (1 - p_correct) / Ns[fr]) / 3
                    dirs_str_parts.append(f"{d_label}:n={d_n},P={p_correct:.2f}{pass_str}")
            n_test_flips_eq = int(((p_viol_test >= tv) & (am_spec_test != fb) & (conf_spec_test >= ts)).sum())
            print(f"τ_v={tv:.2f} τ_s={ts:.2f}  n_train={n:>5}  n_test={n_test_flips_eq:>5}  "
                  f"macro_delta_train={macro_delta:+.6f}  "
                  f"all_dir_pass={all_pass}")
            if all_pass and macro_delta > best_macro_delta and n >= 30:
                best_macro_delta = macro_delta
                best_config = (tv, ts, n, n_test_flips_eq)

    print(f"\nBest TRAIN-OOF config: {best_config}")
    if best_config is not None:
        tv, ts, n_train, n_test = best_config
        print(f"  Projected LB at TRAIN-OOF precision: {0.98150 + best_macro_delta:.5f}")
        # Apply to test, build candidate
        flip_mask_test = (p_viol_test >= tv) & (am_spec_test != fb) & (conf_spec_test >= ts)
        n_flips = int(flip_mask_test.sum())
        new_pred = fb.copy()
        new_pred[flip_mask_test] = am_spec_test[flip_mask_test]
        # Direction breakdown
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                d = int(((fb == fr) & (am_spec_test == to) & flip_mask_test).sum())
                if d > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = d
        print(f"  Test-side flips: {n_flips}")
        print(f"  Test-side dirs: {dirs}")

        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
        })
        out_csv = SUB / f"submission_B3_2stage_tv{int(tv*100)}_ts{int(ts*100)}.csv"
        sub.to_csv(out_csv, index=False)
        print(f"  Emitted: {out_csv}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
