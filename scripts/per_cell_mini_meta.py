"""Per-rule-cell mini-meta-stackers (#3 from the speculative menu).

Cells: (dry, norain, hot, windy, nomulch, stage_kc_bool) → up to 64 cells
(closer to ~128 once Crop_Growth_Stage is split). Within each cell, train
a tiny multinomial LR with strong L2 on a small feature set:
  - 3 LB-best 4-stack probs   (the strong anchor as baseline)
  - 3 xgb_metastack_iso probs (orthogonal stacker signal)
  - 3 realmlp probs           (NN orthogonality)
  - 7 non-rule numerics       (Humidity, Prev_Irrigation, EC, Soil_pH,
                                Field_Area, Sunlight, Organic_Carbon)
= 16 features per row.

Fallback: cells with < N_MIN training rows get LB-best 4-stack as-is.

Differs from prior per-bin / per-cell experiments because:
  - Per-bin blend (failed) optimised one global blend per score-bin (5 bins);
    overfit on 10 free weights.
  - 128-cell empirical-Bayes (failed) just averaged y per cell — no use of
    the strong anchor predictions.
  - This combines: anchor predictions (saturated info) + non-rule features
    (within-cell variation signal) + per-cell specialisation (different
    rows behave differently).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, log_blend  # noqa: E402
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal,  # noqa: E402
                            load_y, normed)


ART = Path("scripts/artifacts")
DATA = Path("data")
SEED = 42
N_FOLDS = 5
N_MIN = 100   # min train rows per cell to train; else fallback
EPS = 1e-12
SMOKE = bool(int(__import__("os").environ.get("SMOKE", "0")))


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def cell_id(df_eng: pd.DataFrame) -> np.ndarray:
    """6-bit cell id over (dry, norain, hot, windy, nomulch, kc_active)."""
    bits = (df_eng["dry"].astype(int).to_numpy() * 32
            + df_eng["norain"].astype(int).to_numpy() * 16
            + df_eng["hot"].astype(int).to_numpy() * 8
            + df_eng["windy"].astype(int).to_numpy() * 4
            + df_eng["nomulch"].astype(int).to_numpy() * 2
            + df_eng["kc_active"].astype(int).to_numpy())
    return bits.astype(np.int16)


def make_features(anchor: np.ndarray, meta_iso: np.ndarray,
                  realmlp: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    nums = ["Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
            "Soil_pH", "Field_Area_hectare", "Sunlight_Hours", "Organic_Carbon"]
    F_anchor = np.log(np.clip(anchor, EPS, 1.0))
    F_meta = np.log(np.clip(meta_iso, EPS, 1.0))
    F_rm = np.log(np.clip(realmlp, EPS, 1.0))
    F_num = df[nums].astype(np.float32).to_numpy()
    return np.hstack([F_anchor, F_meta, F_rm, F_num]).astype(np.float32)


def main():
    log("Loading data + experts")
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    y = load_y()

    train_eng = add_distance_features(train_df)
    test_eng = add_distance_features(test_df)
    train_cell = cell_id(train_eng)
    test_cell = cell_id(test_eng)
    log(f"Train cells: unique={len(np.unique(train_cell))} "
        f"min_count={pd.Series(train_cell).value_counts().min()} "
        f"max_count={pd.Series(train_cell).value_counts().max()}")

    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    lb4_o = log_blend([lb3_o, meta_iso_o], np.array([0.70, 0.30]))
    lb4_t = log_blend([lb3_t, meta_iso_t], np.array([0.70, 0.30]))
    realmlp_o, realmlp_t = L("realmlp")

    Xtr_full = make_features(lb4_o, meta_iso_o, realmlp_o, train_df)
    Xte_full = make_features(lb4_t, meta_iso_t, realmlp_t, test_df)
    log(f"Per-row feature dim={Xtr_full.shape[1]}")

    # Standardize (avoid LR scale issues — uses log-probs + raw nums)
    mu = Xtr_full.mean(0, keepdims=True)
    sd = Xtr_full.std(0, keepdims=True).clip(EPS, None)
    Xtr_full = (Xtr_full - mu) / sd
    Xte_full = (Xte_full - mu) / sd

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = lb4_o.copy()        # fall-back to LB-best 4-stack
    test_acc = lb4_t.copy()         # average across folds will keep it equivalent
    fold_test_predictions = []
    fold_records = []

    cell_train_counts = pd.Series(train_cell).value_counts()
    log(f"Cells with >= {N_MIN} train rows: "
        f"{(cell_train_counts >= N_MIN).sum()} / {len(cell_train_counts)}")

    for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        cells_tr_count = pd.Series(train_cell[tr]).value_counts()
        eligible_cells = set(cells_tr_count[cells_tr_count >= N_MIN].index)
        n_trained = 0
        n_overridden_va = 0
        n_overridden_test = 0
        # Test predictions for this fold start as the LB4 baseline
        test_pred_fold = lb4_t.copy()
        for cell in eligible_cells:
            tr_mask_global = (train_cell == cell)
            tr_idx_cell = np.intersect1d(np.where(tr_mask_global)[0], tr,
                                         assume_unique=True)
            va_idx_cell = np.intersect1d(np.where(tr_mask_global)[0], va,
                                         assume_unique=True)
            test_idx_cell = np.where(test_cell == cell)[0]
            if len(tr_idx_cell) < N_MIN:
                continue
            X_cell_tr = Xtr_full[tr_idx_cell]
            y_cell_tr = y[tr_idx_cell]
            uniq = np.unique(y_cell_tr)
            if len(uniq) < 2:
                continue   # LR needs 2+ classes; LB4 fallback is strict majority anyway
            try:
                lr = LogisticRegression(C=0.1, max_iter=400, solver="lbfgs")
                lr.fit(X_cell_tr, y_cell_tr)
            except Exception as e:
                if fi == 0 and n_trained < 3:
                    log(f"    cell {cell} skipped: {type(e).__name__}: {e}")
                continue
            # Project per-cell (may be 2-class even though target is 3-class)
            n_trained += 1
            if len(va_idx_cell):
                proba = np.zeros((len(va_idx_cell), 3), dtype=np.float32)
                p_lr = lr.predict_proba(Xtr_full[va_idx_cell])
                for j, cls in enumerate(lr.classes_):
                    proba[:, int(cls)] = p_lr[:, j]
                # Smooth: keep tiny mass on missing classes
                proba = (proba + 1e-6) / (proba + 1e-6).sum(1, keepdims=True)
                oof_blend[va_idx_cell] = proba
                n_overridden_va += len(va_idx_cell)
            if len(test_idx_cell):
                proba_te = np.zeros((len(test_idx_cell), 3), dtype=np.float32)
                p_lr_te = lr.predict_proba(Xte_full[test_idx_cell])
                for j, cls in enumerate(lr.classes_):
                    proba_te[:, int(cls)] = p_lr_te[:, j]
                proba_te = (proba_te + 1e-6) / (proba_te + 1e-6).sum(1, keepdims=True)
                test_pred_fold[test_idx_cell] = proba_te
                n_overridden_test += len(test_idx_cell)
        fold_test_predictions.append(test_pred_fold)
        fold_records.append({
            "fold": fi + 1,
            "n_cells_trained": n_trained,
            "n_overridden_va": int(n_overridden_va),
            "n_overridden_test_uniq": int(n_overridden_test),
            "wall_s": round(time.time() - t0, 2),
        })
        log(f"  fold {fi+1}: cells_trained={n_trained} "
            f"va_override={n_overridden_va} wall={time.time()-t0:.1f}s")

    test_acc = np.mean(fold_test_predictions, axis=0)

    bal = balanced_accuracy_score(
        y, (np.log(np.clip(oof_blend, EPS, 1)) + BIAS).argmax(1))
    bal_anchor = balanced_accuracy_score(
        y, (np.log(np.clip(lb4_o, EPS, 1)) + BIAS).argmax(1))
    errs = int((y != (np.log(np.clip(oof_blend, EPS, 1)) + BIAS).argmax(1)).sum())
    errs_anchor = int((y != (np.log(np.clip(lb4_o, EPS, 1)) + BIAS).argmax(1)).sum())
    log(f"PerCell OOF tuned bal_acc = {bal:.6f} (anchor LB4 = {bal_anchor:.6f}, Δ={bal-bal_anchor:+.5f})")
    log(f"PerCell errs = {errs} (anchor errs = {errs_anchor}, Δ={errs-errs_anchor:+d})")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_per_cell_meta{suffix}.npy", oof_blend)
    np.save(ART / f"test_per_cell_meta{suffix}.npy", test_acc)
    out = {
        "smoke": SMOKE,
        "oof_tuned_bal_acc": bal,
        "anchor_lb4_oof_bal_acc": bal_anchor,
        "delta": bal - bal_anchor,
        "errs": errs,
        "errs_anchor": errs_anchor,
        "fold_records": fold_records,
        "n_min": N_MIN,
    }
    (ART / f"per_cell_meta_results{suffix}.json").write_text(json.dumps(out, indent=2))
    log(f"Saved oof_per_cell_meta{suffix}.npy")


if __name__ == "__main__":
    main()
