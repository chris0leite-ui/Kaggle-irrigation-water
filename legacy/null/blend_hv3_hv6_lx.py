"""Extend the hybrid x LGBMxXGB blend by also including hybrid-v6.

Hybrid-v6 = main_routed_v6 + spec-{6,7,8} override. V6 routes
{0,1,2,5} instead of {0,1,2}; its training distribution is different
from v3 (one extra score removed). Even though v6 alone is -0.00012
vs v3, its errors might be complementary.

Test a 3-way blend in log-space across (hybrid-v3, hybrid-v6,
LGBMxXGB) weights summing to 1. Coarse grid, tune log-bias per blend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


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
    tr_m = np.isin(tr_scores, SPEC_SCORES)
    te_m = np.isin(te_scores, SPEC_SCORES)

    # Hybrid-v3
    oof_v3 = np.load(ART / "oof_xgb_dist_routed_v3.npy")
    test_v3 = np.load(ART / "test_xgb_dist_routed_v3.npy")
    oof_spec = np.load(ART / "oof_xgb_spec_678.npy")
    test_spec = np.load(ART / "test_xgb_spec_678.npy")
    oof_hv3 = oof_v3.copy(); oof_hv3[tr_m] = oof_spec[tr_m]
    test_hv3 = test_v3.copy(); test_hv3[te_m] = test_spec[te_m]

    # Hybrid-v6
    oof_v6 = np.load(ART / "oof_xgb_dist_routed_v6.npy")
    test_v6 = np.load(ART / "test_xgb_dist_routed_v6.npy")
    oof_hv6 = oof_v6.copy(); oof_hv6[tr_m] = oof_spec[tr_m]
    test_hv6 = test_v6.copy(); test_hv6[te_m] = test_spec[te_m]

    # LGBM x XGB blend (log-blend at w=0.45)
    oof_lgbm = np.load(ART / "oof_lgbm_te_orig.npy")
    test_lgbm = np.load(ART / "test_lgbm_te_orig.npy")
    oof_xgb = np.load(ART / "oof_xgb_vanilla_dist.npy")
    test_xgb = np.load(ART / "test_xgb_vanilla_dist.npy")
    w_lx = 0.45
    oof_lx = np.exp(w_lx * np.log(np.clip(oof_lgbm, 1e-9, 1.0)) +
                    (1 - w_lx) * np.log(np.clip(oof_xgb, 1e-9, 1.0)))
    oof_lx /= oof_lx.sum(axis=1, keepdims=True)
    test_lx = np.exp(w_lx * np.log(np.clip(test_lgbm, 1e-9, 1.0)) +
                     (1 - w_lx) * np.log(np.clip(test_xgb, 1e-9, 1.0)))
    test_lx /= test_lx.sum(axis=1, keepdims=True)

    def scorit(oof):
        _, t = tune(oof, y, prior)
        return t
    print(f"hybrid-v3:          tuned={scorit(oof_hv3):.5f}")
    print(f"hybrid-v6:          tuned={scorit(oof_hv6):.5f}")
    print(f"LGBMxXGB blend:     tuned={scorit(oof_lx):.5f}")

    # Jaccard error sets
    def err_set(oof):
        b, _ = tune(oof, y, prior)
        pred = (np.log(np.clip(oof, 1e-9, 1.0)) + b).argmax(axis=1)
        return set(np.where(pred != y)[0].tolist())
    eh3 = err_set(oof_hv3); eh6 = err_set(oof_hv6); elx = err_set(oof_lx)
    def jac(a, b):
        return len(a & b) / len(a | b) if (a | b) else 0
    print(f"\nJaccards:")
    print(f"  hv3 vs hv6: {jac(eh3, eh6):.4f}")
    print(f"  hv3 vs lx : {jac(eh3, elx):.4f}")
    print(f"  hv6 vs lx : {jac(eh6, elx):.4f}")

    # 3-way blend sweep in log space
    print(f"\n=== 3-way blend sweep (w3 + w6 + wlx = 1) ===")
    best_bal = -1
    best_w = None
    best_oof = None
    best_test = None
    best_bias = None
    rows = []
    for w3 in np.arange(0.0, 1.01, 0.1):
        for w6 in np.arange(0.0, 1.01 - w3 + 1e-9, 0.1):
            wlx = 1 - w3 - w6
            if wlx < -1e-9: continue
            wlx = max(0.0, wlx)
            oof_b = np.exp(
                w3 * np.log(np.clip(oof_hv3, 1e-9, 1.0)) +
                w6 * np.log(np.clip(oof_hv6, 1e-9, 1.0)) +
                wlx * np.log(np.clip(oof_lx, 1e-9, 1.0)))
            oof_b /= oof_b.sum(axis=1, keepdims=True)
            test_b = np.exp(
                w3 * np.log(np.clip(test_hv3, 1e-9, 1.0)) +
                w6 * np.log(np.clip(test_hv6, 1e-9, 1.0)) +
                wlx * np.log(np.clip(test_lx, 1e-9, 1.0)))
            test_b /= test_b.sum(axis=1, keepdims=True)
            bias, t = tune(oof_b, y, prior)
            rows.append((w3, w6, wlx, t))
            if t > best_bal + 1e-7:
                best_bal = t
                best_w = (float(w3), float(w6), float(wlx))
                best_oof = oof_b
                best_test = test_b
                best_bias = bias

    # show top 10
    rows.sort(key=lambda x: -x[3])
    print(f"top 10 3-way blends:")
    for w3, w6, wlx, t in rows[:10]:
        print(f"  (hv3={w3:.1f}, hv6={w6:.1f}, lx={wlx:.1f})  tuned={t:.5f}")
    print(f"\nbest: {best_w}  tuned={best_bal:.5f}")
    print(f"  vs hybrid-v3 alone       : {best_bal - 0.97352:+.5f}")
    print(f"  vs hybrid x lgbmxgb      : {best_bal - 0.97362:+.5f}")

    np.save(ART / "oof_hybrid_v3v6_lgbmxgb_blend.npy", best_oof)
    np.save(ART / "test_hybrid_v3v6_lgbmxgb_blend.npy", best_test)
    with open(ART / "blend_hv3_hv6_lx_results.json", "w") as f:
        json.dump({"best": {"w": best_w, "tuned": float(best_bal)},
                   "top10": [[float(x) for x in r] for r in rows[:10]]}, f, indent=2)
    if best_bal > 0.97362:
        tuned_idx = (np.log(np.clip(best_test, 1e-9, 1.0)) + best_bias).argmax(axis=1)
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
            SUB / "submission_blend_hv3_hv6_lx.csv", index=False)


if __name__ == "__main__":
    main()
