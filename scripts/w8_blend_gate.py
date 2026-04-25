"""W8 EXTRA_FE='w8' recipe — blend-gate vs LB-best 4-stack.

Inputs (produced by `EXTRA_FE=w8 python scripts/recipe_full_te.py`):
  scripts/artifacts/oof_recipe_full_te_fexw8.npy
  scripts/artifacts/test_recipe_full_te_fexw8.npy
  scripts/artifacts/recipe_full_te_fexw8_results.json

Decision rule (per the magnitude-trap heuristics):
  - PASS gate: standalone OOF >= recipe baseline 0.97967 AND
               errs <= LB-best 4-stack 9415 AND
               Jaccard < 0.85 vs LB-best AND
               blend Δ on top of LB-best 4-stack at fixed bias >= +0.0002

If gate passes, emit submission CSV at the peak α (use the LB-best
4-stack's bias unchanged — no per-α retune to avoid binhigh-style
selection inflation).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def L(n): return np.load(ART / n)
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


def main():
    # Load truth
    train = pd.read_csv("data/train.csv")
    y = train["Irrigation_Need"].astype(str).map({"Low":0,"Medium":1,"High":2}).values.astype(np.int64)
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].values
    BIAS = np.array([1.4324, 1.4689, 3.4008])

    # Reconstruct LB-best 4-stack
    lb3 = lb((L("oof_recipe_full_te.npy"), .25),
             (L("oof_recipe_pseudolabel.npy"), .35),
             (L("oof_recipe_pseudolabel_seed7labeler.npy"), .40))
    s1 = lb((lb3, .80), (L("oof_realmlp.npy"), .20))
    s2 = lb((s1, .925), (iso(L("oof_xgb_nonrule.npy"), y), .075))
    final_lb = lb((s2, .70), (iso(L("oof_xgb_metastack.npy"), y), .30))
    pred_lb = (np.log(final_lb.clip(1e-12)) + BIAS).argmax(1)
    lbbest_ba = balanced_accuracy_score(y, pred_lb)
    lbbest_errs = (pred_lb != y).sum()
    print(f"LB-best 4-stack: tuned OOF {lbbest_ba:.5f}, errs {lbbest_errs}")
    # Test side
    test_lb3 = lb((L("test_recipe_full_te.npy"), .25),
                  (L("test_recipe_pseudolabel.npy"), .35),
                  (L("test_recipe_pseudolabel_seed7labeler.npy"), .40))
    # iso from train applied to test — fit on train, transform test
    def iso_apply(train_p, train_y, test_p):
        o = np.zeros_like(test_p)
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
            ir.fit(train_p[:, k], (train_y == k).astype(float))
            o[:, k] = ir.transform(test_p[:, k])
        return o / o.sum(1, keepdims=True).clip(1e-9)
    test_nr_iso = iso_apply(L("oof_xgb_nonrule.npy"), y, L("test_xgb_nonrule.npy"))
    test_ms_iso = iso_apply(L("oof_xgb_metastack.npy"), y, L("test_xgb_metastack.npy"))
    test_s1 = lb((test_lb3, .80), (L("test_realmlp.npy"), .20))
    test_s2 = lb((test_s1, .925), (test_nr_iso, .075))
    test_final_lb = lb((test_s2, .70), (test_ms_iso, .30))

    # Recipe baseline (for standalone Δ)
    recipe = L("oof_recipe_full_te.npy")
    recipe_pred = (np.log(recipe.clip(1e-12)) + BIAS).argmax(1)
    recipe_ba = balanced_accuracy_score(y, recipe_pred)
    recipe_errs = (recipe_pred != y).sum()
    print(f"Recipe baseline:  tuned OOF {recipe_ba:.5f}, errs {recipe_errs}")

    # Recipe + W8
    p = ART / "oof_recipe_full_te_fexw8.npy"
    if not p.exists():
        print(f"\nNOT FOUND: {p}")
        print("Run: EXTRA_FE=w8 python scripts/recipe_full_te.py first.")
        return
    w8 = L("oof_recipe_full_te_fexw8.npy")
    w8_pred = (np.log(w8.clip(1e-12)) + BIAS).argmax(1)
    w8_ba = balanced_accuracy_score(y, w8_pred)
    w8_errs = (w8_pred != y).sum()
    delta_vs_recipe = w8_ba - recipe_ba
    print(f"\nRecipe+W8:        tuned OOF {w8_ba:.5f}, errs {w8_errs}")
    print(f"  Δ vs recipe:    {delta_vs_recipe:+.5f}")

    # Use the recipe_full_te_fexw8's own tuned bias (from results JSON)
    fexw8_json = json.loads((ART / "recipe_full_te_fexw8_results.json").read_text())
    own_bias = np.array(fexw8_json.get("tuned_bias", BIAS))
    own_pred = (np.log(w8.clip(1e-12)) + own_bias).argmax(1)
    own_ba = balanced_accuracy_score(y, own_pred)
    print(f"  own tuned bias: {own_bias.tolist()}, own tuned OOF {own_ba:.5f}")

    # Jaccards
    e_w8 = (w8_pred != y); e_lb = (pred_lb != y); e_recipe = (recipe_pred != y)
    j_lb = (e_w8 & e_lb).sum() / max((e_w8 | e_lb).sum(), 1)
    j_recipe = (e_w8 & e_recipe).sum() / max((e_w8 | e_recipe).sum(), 1)
    print(f"  Jaccard vs recipe:    {j_recipe:.4f}")
    print(f"  Jaccard vs LB-best:   {j_lb:.4f}")

    # Blend gate vs LB-best 4-stack at FIXED recipe bias
    print("\n--- blend gate: log-blend (LB-best, w8) at fixed recipe bias ---")
    print(f"{'α':>6} {'tuned':>9} {'Δ':>9} {'errs':>6} {'recL':>7} {'recM':>7} {'recH':>7} {'flag'}")
    best_alpha = 0.0; best_delta = 0.0; best_pred = pred_lb
    for alpha in [0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500]:
        blend = lb((final_lb, 1 - alpha), (w8, alpha))
        bp = (np.log(blend.clip(1e-12)) + BIAS).argmax(1)
        ba = balanced_accuracy_score(y, bp)
        delta = ba - lbbest_ba
        errs = (bp != y).sum()
        cm_diag = [((y == k) & (bp == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
        cm_lb = [((y == k) & (pred_lb == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
        rec_drops = [(cm_diag[k] - cm_lb[k]) for k in (0,1,2)]
        flag = ""
        if delta >= 0.0002 and all(d >= -5e-4 for d in rec_drops):
            flag = "  *** PASS ***"
        elif delta >= 0.0001:
            flag = "  (close)"
        print(f"{alpha:>6.3f} {ba:>9.5f} {delta:>+9.5f} {errs:>6} {cm_diag[0]:>7.4f} {cm_diag[1]:>7.4f} {cm_diag[2]:>7.4f}{flag}")
        if delta > best_delta:
            best_delta = delta; best_alpha = alpha; best_pred = bp

    print(f"\nBest: α={best_alpha:.3f}, Δ={best_delta:+.5f}")

    # Emit if PASS
    if best_delta >= 0.0002:
        # Build test-side blend
        test_w8 = L("test_recipe_full_te_fexw8.npy")
        test_blend = lb((test_final_lb, 1 - best_alpha), (test_w8, best_alpha))
        test_bp = (np.log(test_blend.clip(1e-12)) + BIAS).argmax(1)
        labels = np.array(["Low", "Medium", "High"])[test_bp]
        sub_path = SUB / f"submission_lb4_w8_a{int(best_alpha*1000):03d}.csv"
        pd.DataFrame({"id": test_ids, "Irrigation_Need": labels}).to_csv(sub_path, index=False)
        print(f"\n*** EMITTED: {sub_path} (α={best_alpha:.3f}, Δ OOF {best_delta:+.5f}) ***")
        from collections import Counter
        c = Counter(labels.tolist())
        print(f"  pred dist: {dict(c)}")
        # Test-row diff vs LB-best primary
        primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
        n_diff = (primary["Irrigation_Need"].values != labels).sum()
        print(f"  rows differing from LB-best primary: {n_diff} ({100*n_diff/len(labels):.3f}%)")
    else:
        print(f"\nNO PASS (best Δ={best_delta:+.5f} < +0.0002 LB-transfer threshold)")

    # Save analysis
    out = {
        "lbbest_ba": float(lbbest_ba), "lbbest_errs": int(lbbest_errs),
        "recipe_ba": float(recipe_ba), "recipe_errs": int(recipe_errs),
        "w8_ba": float(w8_ba), "w8_errs": int(w8_errs),
        "delta_vs_recipe": float(delta_vs_recipe),
        "jaccard_vs_recipe": float(j_recipe),
        "jaccard_vs_lbbest": float(j_lb),
        "best_blend_alpha": float(best_alpha),
        "best_blend_delta": float(best_delta),
    }
    out_path = ART / "w8_blend_gate_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nresults → {out_path}")


if __name__ == "__main__":
    main()
