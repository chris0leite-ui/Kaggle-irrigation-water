"""W8 grader — score each W8 FE idea on a fast 1-fold smoke.

Approach: train a moderately-rich XGB (raw 19 + dist 28) baseline,
then test each idea as added cols. A passing idea has Δ >= +0.0005
tuned bal_acc OR Jaccard < 0.85 vs LB-best with errs ≤ baseline.

Smoke config: 250k stratified subsample, 1 fold, max_depth=4,
n_estimators=600, early_stopping=50. Each idea ~30s.
"""
from __future__ import annotations
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
from scripts.dgp_formula import dgp_score
from scripts.w8_fe_ideas import ALL_IDEAS, rule_features


ART = "scripts/artifacts/"
SUBSAMPLE = 250_000
SEED = 42

# ---------- features for the smoke base ----------
RAW_NUMS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
            "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
            "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare"]
RAW_CATS = ["Mulching_Used", "Crop_Growth_Stage", "Crop_Type", "Soil_Type",
            "Season", "Irrigation_Type", "Water_Source", "Region"]


def build_dist_features(df):
    r = rule_features(df)
    out = pd.DataFrame({
        "sm_dist": r["sm"] - 25, "rf_dist": r["rf"] - 300,
        "tc_dist": r["tc"] - 30, "ws_dist": r["ws"] - 10,
        "abs_sm": np.abs(r["sm"] - 25), "abs_rf": np.abs(r["rf"] - 300),
        "abs_tc": np.abs(r["tc"] - 30), "abs_ws": np.abs(r["ws"] - 10),
        "dgp_dry": r["dry"], "dgp_nor": r["nor"], "dgp_hot": r["hot"],
        "dgp_win": r["win"], "dgp_mu": r["mu"], "dgp_kc": r["kc"],
        "dgp_score": r["score"],
    })
    return out


def factorize_cats(train_df, test_df, cats):
    """Factorize categoricals using train vocab."""
    out_tr = pd.DataFrame()
    out_te = pd.DataFrame()
    for c in cats:
        tr = train_df[c].astype(str).values
        te = test_df[c].astype(str).values
        vocab = sorted(pd.unique(np.concatenate([tr, te]).astype(str)))
        m = {v: i for i, v in enumerate(vocab)}
        out_tr[c] = pd.Series(tr).map(m).astype(np.int32)
        out_te[c] = pd.Series(te).map(m).astype(np.int32)
    return out_tr, out_te


def make_base(df_tr_full):
    """Construct base feature matrix: raw nums + dist + factorized cats."""
    nums = df_tr_full[RAW_NUMS].astype(np.float32).reset_index(drop=True)
    dist = build_dist_features(df_tr_full).reset_index(drop=True)
    return pd.concat([nums, dist], axis=1)


def train_eval_one_fold(Xtr, ytr, Xva, yva, name="base"):
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    params = {"objective": "multi:softprob", "num_class": 3,
              "eta": 0.1, "max_depth": 4, "min_child_weight": 5,
              "subsample": 0.9, "colsample_bytree": 0.9,
              "reg_alpha": 5, "reg_lambda": 5, "tree_method": "hist",
              "verbosity": 0, "seed": SEED}
    bst = xgb.train(params, dtr, num_boost_round=600,
                    evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
    proba = bst.predict(dva)
    return proba, bst.best_iteration


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
    print(f"=== W8 grader (smoke: {SUBSAMPLE} stratified subsample × 1 fold) ===")
    t0 = time.time()
    train = pd.read_csv("data/train.csv", dtype_backend="numpy_nullable")
    test = pd.read_csv("data/test.csv", dtype_backend="numpy_nullable")
    y_full = train["Irrigation_Need"].astype(str).map({"Low":0,"Medium":1,"High":2}).values.astype(np.int64)

    rng = np.random.default_rng(SEED)
    idx = []
    for k in [0,1,2]:
        ki = np.where(y_full == k)[0]
        n_k = int(round(SUBSAMPLE * (y_full == k).mean()))
        idx.extend(rng.choice(ki, size=min(n_k, len(ki)), replace=False))
    idx = np.array(idx); rng.shuffle(idx)
    sub = train.iloc[idx].reset_index(drop=True)
    y_sub = y_full[idx]

    # Cat factorize using full train+test vocab
    cat_tr, cat_te = factorize_cats(sub, test, RAW_CATS)
    base = make_base(sub)
    base = pd.concat([base, cat_tr.reset_index(drop=True)], axis=1)
    print(f"base feats: {base.shape[1]} cols, sub n={len(sub)}")

    # 1-fold inner split (stratified)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(skf.split(np.zeros(len(y_sub)), y_sub))
    print(f"1-fold split: tr={len(tr)} va={len(va)}")

    # Baseline (no idea)
    print("\n--- baseline (raw+dist+cats) ---")
    t1 = time.time()
    proba_base, n_iter = train_eval_one_fold(base.iloc[tr].values, y_sub[tr],
                                             base.iloc[va].values, y_sub[va], "base")
    log_p_base = np.log(proba_base.clip(1e-12))
    base_argmax = balanced_accuracy_score(y_sub[va], proba_base.argmax(1))
    base_tuned, _ = coord_ascent_bias(log_p_base, y_sub[va])
    base_pred_tuned = (log_p_base + np.array(_)).argmax(1)
    base_errs = (base_pred_tuned != y_sub[va]).sum()
    print(f"  base: argmax={base_argmax:.5f} tuned={base_tuned:.5f} errs={base_errs} n_iter={n_iter} ({time.time()-t1:.1f}s)")

    # LB-best argmax on val rows for Jaccard reference
    def L(n): return np.load(ART + n)
    def iso(p, y_):
        o = np.zeros_like(p)
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
            o[:, k] = ir.fit_transform(p[:, k], (y_ == k).astype(float))
        return o / o.sum(1, keepdims=True).clip(1e-9)
    def lbf(*pw):
        out = sum(w * np.log(p.clip(1e-12)) for p, w in pw)
        out = np.exp(out - out.max(1, keepdims=True))
        return out / out.sum(1, keepdims=True)
    lb3 = lbf((L("oof_recipe_full_te.npy"), .25),
              (L("oof_recipe_pseudolabel.npy"), .35),
              (L("oof_recipe_pseudolabel_seed7labeler.npy"), .40))
    s1 = lbf((lb3, .80), (L("oof_realmlp.npy"), .20))
    s2 = lbf((s1, .925), (iso(L("oof_xgb_nonrule.npy"), y_full), .075))
    final = lbf((s2, .70), (iso(L("oof_xgb_metastack.npy"), y_full), .30))
    bias = np.array([1.4324, 1.4689, 3.4008])
    lbbest_pred = (np.log(final.clip(1e-12)) + bias).argmax(1)
    lbbest_va = lbbest_pred[idx[va]]
    lbbest_errs = (lbbest_va != y_sub[va]).sum()
    print(f"  LB-best on these val rows: errs={lbbest_errs}")

    results = []
    for iid, name, fn in ALL_IDEAS:
        t1 = time.time()
        try:
            extra = fn(sub)
        except Exception as e:
            print(f"  {iid} ERROR: {e}")
            continue
        # Append extra cols
        extra_df = pd.DataFrame({k: v for k, v in extra.items()})
        # Sanitize
        for c in extra_df.columns:
            extra_df[c] = pd.to_numeric(extra_df[c], errors="coerce").fillna(0).astype(np.float32)
        feats = pd.concat([base, extra_df.reset_index(drop=True)], axis=1)
        proba, ni = train_eval_one_fold(feats.iloc[tr].values, y_sub[tr],
                                        feats.iloc[va].values, y_sub[va], iid)
        log_p = np.log(proba.clip(1e-12))
        argmax = balanced_accuracy_score(y_sub[va], proba.argmax(1))
        tuned, b = coord_ascent_bias(log_p, y_sub[va])
        pred = (log_p + np.array(b)).argmax(1)
        errs = (pred != y_sub[va]).sum()
        delta = tuned - base_tuned
        # Jaccard vs LB-best
        e_us = (pred != y_sub[va]); e_lb = (lbbest_va != y_sub[va])
        inter = (e_us & e_lb).sum(); union = (e_us | e_lb).sum()
        jacc = inter / union if union > 0 else 1.0
        rec = {"id": iid, "name": name, "n_extra_cols": int(extra_df.shape[1]),
               "argmax": float(argmax), "tuned": float(tuned), "delta_vs_base": float(delta),
               "errs": int(errs), "lbbest_errs": int(lbbest_errs),
               "jaccard_vs_lbbest": float(jacc), "n_iter": int(ni),
               "wall_s": time.time() - t1}
        results.append(rec)
        flag = ""
        if delta >= 0.0005:
            flag = "  *** PASS (Δ≥+0.0005) ***"
        elif delta >= 0.0002:
            flag = "  (close: +0.0002≤Δ<+0.0005)"
        print(f"  {iid:>4} {name:<28} +{extra_df.shape[1]:>2}c  tuned={tuned:.5f}  Δ={delta:+.5f}  errs={errs}/{lbbest_errs} J={jacc:.3f}  ({rec['wall_s']:.1f}s){flag}")

    # Save
    out = ART + "w8_grade_results.json"
    with open(out, "w") as f:
        json.dump({"subsample": SUBSAMPLE, "n_ideas": len(results),
                   "base_tuned": base_tuned, "base_errs": int(base_errs),
                   "lbbest_errs_on_va": int(lbbest_errs), "results": results,
                   "total_wall_s": time.time() - t0}, f, indent=2)
    print(f"\nresults → {out}, total wall {(time.time()-t0)/60:.1f} min")

    # Surface
    sorted_r = sorted(results, key=lambda r: -r["delta_vs_base"])
    print("\n=== TOP 8 by Δ tuned vs base ===")
    for r in sorted_r[:8]:
        print(f"  {r['id']} {r['name']:<28} Δ={r['delta_vs_base']:+.5f}  J={r['jaccard_vs_lbbest']:.3f}  errs={r['errs']}/{r['lbbest_errs']}")

    passing = [r for r in results if r["delta_vs_base"] >= 0.0005]
    close = [r for r in results if 0.0002 <= r["delta_vs_base"] < 0.0005]
    print(f"\nPASSING (Δ≥+0.0005): {len(passing)}")
    for r in passing:
        print(f"  {r['id']} {r['name']} Δ={r['delta_vs_base']:+.5f}")
    print(f"CLOSE (0.0002≤Δ<0.0005): {len(close)}")
    for r in close:
        print(f"  {r['id']} {r['name']} Δ={r['delta_vs_base']:+.5f}")


if __name__ == "__main__":
    main()
