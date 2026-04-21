"""Blend current-best hybrid (OOF 0.97352) with LGBM x XGB blend
(OOF 0.97327). The two architectures make structurally different
predictions:

  hybrid = routed-{0,1,2} XGB + spec-{6,7,8} override
     - deterministic rule on rule-trivial scores
     - specialist overlay on ambiguous high-side scores
  lgbmxgb = vanilla LGBM-dist x XGB-dist log-blend
     - no routing, no specialists
     - pure model-family diversity

If error geometry is complementary, blending them could add diversity
that neither captured alone. Fast test — both OOFs on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
SPEC_SCORES = (6, 7, 8)
ACTIVE_STAGES = ("Flowering", "Vegetative")
ART = Path("scripts/artifacts")
SUB = Path("submissions")


def tune(p, y, prior):
    lp = np.log(np.clip(p, 1e-9, 1.0))
    b = -np.log(prior)
    best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
    grid = np.linspace(-3, 3, 61)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = b.copy()
            sc = []
            for g in grid:
                base[k] = b[k] + g
                sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
            j = int(np.argmax(sc))
            if sc[j] > best + 1e-6:
                b[k] = b[k] + grid[j]
                best = sc[j]
                imp = True
        if not imp:
            break
    return b, best


def compute_dgp_score(df):
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(ACTIVE_STAGES).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).values


def main():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    tr_scores = compute_dgp_score(tr)
    te_scores = compute_dgp_score(te)

    # Hybrid OOF+test
    oof_main = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_main = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")
    tr_m = np.isin(tr_scores, SPEC_SCORES)
    te_m = np.isin(te_scores, SPEC_SCORES)
    oof_hyb = oof_main.copy()
    oof_hyb[tr_m] = oof_spec[tr_m]
    test_hyb = test_main.copy()
    test_hyb[te_m] = test_spec[te_m]

    # LGBM x XGB blend OOF+test (using on-disk OOFs we have)
    oof_lgbm = np.load(ART / "oof_lgbm_te_orig.npy")   # LGBM-dist ≈ 0.97270
    test_lgbm = np.load(ART / "test_lgbm_te_orig.npy")
    oof_xgb = np.load(ART / "oof_xgb_vanilla_dist.npy")  # XGB-dist = 0.97304
    test_xgb = np.load(ART / "test_xgb_vanilla_dist.npy")

    # Best observed log-blend was w_lgbm=0.45 (log space)
    w_lx = 0.45
    oof_lx = np.exp(w_lx * np.log(np.clip(oof_lgbm, 1e-9, 1.0)) +
                    (1 - w_lx) * np.log(np.clip(oof_xgb, 1e-9, 1.0)))
    oof_lx /= oof_lx.sum(axis=1, keepdims=True)
    test_lx = np.exp(w_lx * np.log(np.clip(test_lgbm, 1e-9, 1.0)) +
                     (1 - w_lx) * np.log(np.clip(test_xgb, 1e-9, 1.0)))
    test_lx /= test_lx.sum(axis=1, keepdims=True)

    _, hyb_bal = tune(oof_hyb, y, prior)
    _, lx_bal = tune(oof_lx, y, prior)
    print(f"hybrid OOF tuned          : {hyb_bal:.5f}")
    print(f"LGBM*0.45+XGB*0.55 tuned  : {lx_bal:.5f}")

    # Jaccard first
    b_h, _ = tune(oof_hyb, y, prior)
    b_l, _ = tune(oof_lx, y, prior)
    pred_h = (np.log(np.clip(oof_hyb, 1e-9, 1.0)) + b_h).argmax(axis=1)
    pred_l = (np.log(np.clip(oof_lx, 1e-9, 1.0)) + b_l).argmax(axis=1)
    err_h = set(np.where(pred_h != y)[0].tolist())
    err_l = set(np.where(pred_l != y)[0].tolist())
    j = len(err_h & err_l) / len(err_h | err_l) if (err_h | err_l) else 0
    print(f"Jaccard hybrid vs lgbmxgb : {j:.4f}  "
          f"(inter {len(err_h & err_l)} / union {len(err_h | err_l)})")

    print("\n=== Hybrid x LGBMxXGB blend sweep ===")
    results = {"components": {"hybrid": float(hyb_bal),
                              "lgbmxgb": float(lx_bal)},
               "jaccard": float(j),
               "sweep": {}}
    best_bal = -1
    best_w = None
    best_blend_oof = None
    best_blend_test = None
    best_bias = None
    for w in np.arange(0.0, 1.01, 0.05):
        blend_oof = np.exp(w * np.log(np.clip(oof_hyb, 1e-9, 1.0)) +
                           (1 - w) * np.log(np.clip(oof_lx, 1e-9, 1.0)))
        blend_oof /= blend_oof.sum(axis=1, keepdims=True)
        blend_test = np.exp(w * np.log(np.clip(test_hyb, 1e-9, 1.0)) +
                            (1 - w) * np.log(np.clip(test_lx, 1e-9, 1.0)))
        blend_test /= blend_test.sum(axis=1, keepdims=True)
        bias, t = tune(blend_oof, y, prior)
        results["sweep"][f"w_hyb={w:.2f}"] = float(t)
        flag = ""
        if t > best_bal:
            best_bal = t
            best_w = float(w)
            best_blend_oof = blend_oof
            best_blend_test = blend_test
            best_bias = bias
            flag = "  <-- new best"
        print(f"  w_hyb={w:.2f}  tuned={t:.5f}{flag}")

    print(f"\nbest w_hyb={best_w}  tuned={best_bal:.5f}")
    print(f"  vs hybrid alone           : {best_bal - hyb_bal:+.5f}")
    print(f"  vs LGBM*0.45+XGB*0.55     : {best_bal - lx_bal:+.5f}")
    results["best"] = {"w_hyb": best_w, "tuned": float(best_bal)}

    cm = confusion_matrix(
        y, (np.log(np.clip(best_blend_oof, 1e-9, 1.0)) + best_bias).argmax(axis=1))
    print(f"  best-blend OOF confusion:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_hybrid_lgbmxgb_blend.npy", best_blend_oof)
    np.save(ART / "test_hybrid_lgbmxgb_blend.npy", best_blend_test)
    with open(ART / "blend_hybrid_lgbmxgb_results.json", "w") as f:
        json.dump(results, f, indent=2)

    tuned_idx = (np.log(np.clip(best_blend_test, 1e-9, 1.0)) + best_bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_hybrid_lgbmxgb_blend.csv", index=False)


if __name__ == "__main__":
    main()
