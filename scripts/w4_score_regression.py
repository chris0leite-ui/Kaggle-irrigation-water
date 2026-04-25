"""W4 — XGB score-regression: predict integer dgp_score (0-9), convert
to 3-class probabilities via softmax over distances to class midpoints.

Architecturally distinct from every model in the meta-stacker bank
(all 70+ are CE-trained on 3-class y). RMSE on integer score gives
a different gradient surface that may produce orthogonal errors.

Feature set: same 89 cols as xgb_dist_digits (dist + raw + digits +
factorized cats). Fast pipeline (~5 min/fold on 504k → ~25 min total).

SMOKE first: 1 fold × 20k subsample × n_est=300 → ~30s.

Output convention:
  - oof_xgb_score_reg.npy: (N, 3) softmax probs from class-midpoint distances
  - test_xgb_score_reg.npy: (N_test, 3) same
Match the convention of every other OOF in the artifact bank.
"""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scripts.dgp_formula import dgp_score
from scripts.w8_fe_ideas import rule_features
from scripts.w8_grade import build_dist_features, factorize_cats, RAW_NUMS, RAW_CATS

ART = "scripts/artifacts/"
SMOKE = os.environ.get("SMOKE", "1") == "1"
SUBSAMPLE = 20_000 if SMOKE else None
N_FOLDS = 5  # full 5-fold even in smoke (subsample is small enough)
N_EST = 300 if SMOKE else 2000
SEED = 42

# Class midpoints in score space (Low: 0-3, Medium: 4-6, High: 7-9)
MIDPOINTS = np.array([1.5, 5.0, 8.0], dtype=np.float32)


def add_digit_cols(df, n_digits=4):
    """Per-numeric digit extraction, matching xgb_dist_digits features."""
    out = pd.DataFrame()
    EPS = 1e-6
    for c in RAW_NUMS:
        v = df[c].astype(float).values
        for d in range(-2, 2):
            mod = 10 ** (-d)
            digit = (np.floor(v * mod + EPS) % 10).astype(np.int8)
            if digit.std() > 0:
                out[f"{c}_d{d}"] = digit
    return out


def cont_to_probs(cont_score, sigma=1.5):
    """Convert continuous predicted score to 3-class probabilities via
    softmax over negative squared distance to class midpoints.

    sigma controls sharpness; sigma=1.5 spans two integer scores per std.
    """
    # cont_score shape (N,), midpoints shape (3,)
    sq = -((cont_score[:, None] - MIDPOINTS[None, :]) / sigma) ** 2
    sq = sq - sq.max(axis=1, keepdims=True)
    e = np.exp(sq)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def coord_ascent_bias(log_p, y):
    cur = np.array([0., 0., 0.])
    best = balanced_accuracy_score(y, (log_p + cur).argmax(1))
    rounds = 0; improved = True
    while improved and rounds < 8:
        improved = False
        for k in range(3):
            for db in np.linspace(-3.5, 3.5, 15):
                test_b = cur.copy(); test_b[k] = db
                ba_t = balanced_accuracy_score(y, (log_p + test_b).argmax(1))
                if ba_t > best + 1e-6:
                    best = ba_t; cur = test_b; improved = True
        rounds += 1
    return best, cur


def main():
    print(f"=== W4 score-regression XGB (SMOKE={SMOKE}, N_FOLDS={N_FOLDS}, N_EST={N_EST}) ===")
    t0 = time.time()
    train = pd.read_csv("data/train.csv", dtype_backend="numpy_nullable")
    test = pd.read_csv("data/test.csv", dtype_backend="numpy_nullable")
    y_full = train["Irrigation_Need"].astype(str).map({"Low":0,"Medium":1,"High":2}).values.astype(np.int64)
    # Target = class midpoint (not dgp_score). For flipped rows where rule
    # disagrees with truth, this differs from dgp_score → forces XGB to
    # learn flips from non-rule features.
    score_full = MIDPOINTS[y_full].astype(np.float32)
    print(f"loaded train {len(train)} test {len(test)} ({time.time()-t0:.1f}s)")

    if SUBSAMPLE is not None:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train), size=SUBSAMPLE, replace=False)
        train = train.iloc[idx].reset_index(drop=True)
        y = y_full[idx]; score_target = score_full[idx]
    else:
        y = y_full; score_target = score_full

    # Build features: raw nums + dist + digits + factorized cats
    cat_tr, cat_te = factorize_cats(train, test, RAW_CATS)
    nums_tr = train[RAW_NUMS].astype(np.float32).reset_index(drop=True)
    nums_te = test[RAW_NUMS].astype(np.float32).reset_index(drop=True)
    dist_tr = build_dist_features(train).reset_index(drop=True)
    dist_te = build_dist_features(test).reset_index(drop=True)
    digits_tr = add_digit_cols(train).reset_index(drop=True)
    digits_te = add_digit_cols(test).reset_index(drop=True)
    feats_tr = pd.concat([nums_tr, dist_tr, digits_tr, cat_tr.reset_index(drop=True)], axis=1)
    feats_te = pd.concat([nums_te, dist_te, digits_te, cat_te.reset_index(drop=True)], axis=1)
    print(f"features: {feats_tr.shape[1]} cols, train={len(feats_tr)}, test={len(feats_te)}")

    # XGB regressor params — heavy reg matching recipe philosophy
    params = dict(
        n_estimators=N_EST, max_depth=4, max_leaves=30,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="reg:squarederror", tree_method="hist",
        eval_metric="rmse",
        n_jobs=-1, random_state=SEED, verbosity=0,
        early_stopping_rounds=50 if SMOKE else 200,
    )

    skf = StratifiedKFold(n_splits=max(N_FOLDS, 2), shuffle=True, random_state=SEED)
    fold_iter = list(skf.split(feats_tr, y))[:N_FOLDS]

    oof_cont = np.zeros(len(y), dtype=np.float32)
    test_cont = np.zeros(len(feats_te), dtype=np.float32)
    fold_rmse = []
    for f, (tr, va) in enumerate(fold_iter):
        tf = time.time()
        model = xgb.XGBRegressor(**params)
        model.fit(feats_tr.iloc[tr].values, score_target[tr],
                  eval_set=[(feats_tr.iloc[va].values, score_target[va])],
                  verbose=500)
        pred_va = model.predict(feats_tr.iloc[va].values)
        oof_cont[va] = pred_va
        test_cont += model.predict(feats_te.values) / N_FOLDS
        rmse = float(np.sqrt(((pred_va - score_target[va]) ** 2).mean()))
        fold_rmse.append(rmse)
        print(f"  fold {f+1}: best_iter={model.best_iteration} val_rmse={rmse:.4f} ({time.time()-tf:.1f}s)")

    # Convert to 3-class probs (sweep sigma to find best macro-recall)
    print("\n--- sigma sweep on OOF ---")
    best_sigma = 1.5; best_ba_argmax = 0
    for sigma in [0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
        probs = cont_to_probs(oof_cont, sigma=sigma)
        ba = balanced_accuracy_score(y, probs.argmax(1))
        if ba > best_ba_argmax:
            best_ba_argmax = ba; best_sigma = sigma
        print(f"  sigma={sigma:.1f}  argmax_bal_acc={ba:.5f}")
    print(f"\nbest_sigma = {best_sigma}, argmax tuned next")
    oof_p = cont_to_probs(oof_cont, sigma=best_sigma)
    test_p = cont_to_probs(test_cont, sigma=best_sigma)

    # Tuned log-bias
    log_p = np.log(oof_p.clip(1e-12))
    tuned_ba, bias = coord_ascent_bias(log_p, y)
    print(f"tuned OOF: {tuned_ba:.5f}  bias={bias.tolist()}")
    pred = (log_p + bias).argmax(1)
    errs = int((pred != y).sum())
    print(f"errs: {errs}")

    # Jaccard vs LB-best 4-stack (only on full data)
    jacc = None
    if not SMOKE:
        from sklearn.isotonic import IsotonicRegression
        def L(n): return np.load(ART + n)
        def iso(p, y_):
            o = np.zeros_like(p)
            for k in range(3):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
                o[:, k] = ir.fit_transform(p[:, k], (y_ == k).astype(float))
            return o / o.sum(1, keepdims=True).clip(1e-9)
        def lb(*pw):
            out = sum(w * np.log(p.clip(1e-12)) for p, w in pw)
            out = np.exp(out - out.max(1, keepdims=True))
            return out / out.sum(1, keepdims=True)
        lb3 = lb((L("oof_recipe_full_te.npy"), .25),
                 (L("oof_recipe_pseudolabel.npy"), .35),
                 (L("oof_recipe_pseudolabel_seed7labeler.npy"), .40))
        s1 = lb((lb3, .80), (L("oof_realmlp.npy"), .20))
        s2 = lb((s1, .925), (iso(L("oof_xgb_nonrule.npy"), y), .075))
        final_lb = lb((s2, .70), (iso(L("oof_xgb_metastack.npy"), y), .30))
        BIAS = np.array([1.4324, 1.4689, 3.4008])
        pred_lb = (np.log(final_lb.clip(1e-12)) + BIAS).argmax(1)
        e_us = (pred != y); e_lb = (pred_lb != y)
        jacc = (e_us & e_lb).sum() / max((e_us | e_lb).sum(), 1)
        print(f"\nLB-best 4-stack errs: {e_lb.sum()}, Jaccard(W4, LB-best): {jacc:.4f}")

    # Save
    suffix = "_smoke" if SMOKE else ""
    np.save(ART + f"oof_xgb_score_reg{suffix}.npy", oof_p)
    np.save(ART + f"test_xgb_score_reg{suffix}.npy", test_p)
    out = {
        "smoke": SMOKE, "n_folds": N_FOLDS, "n_est": N_EST,
        "best_sigma": float(best_sigma),
        "fold_rmse": fold_rmse, "tuned_oof": float(tuned_ba),
        "bias": bias.tolist(), "errs": int(errs),
        "jaccard_vs_lbbest": float(jacc) if jacc is not None else None,
        "wall_s": time.time() - t0,
    }
    json_path = ART + f"w4_score_reg{suffix}_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults → {json_path}, total wall {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
