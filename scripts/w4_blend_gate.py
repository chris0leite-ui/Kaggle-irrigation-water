"""W4 blend-gate vs LB-best 4-stack — fixed-bias log-blend sweep."""
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

    # W4 OOF + test
    w4 = L("oof_xgb_score_reg.npy")
    w4_test = L("test_xgb_score_reg.npy")

    # W4 standalone with WIDER bias grid (initial coord ascent hit ±3.5 edges)
    print("\n--- W4 standalone, wider bias grid ---")
    log_p = np.log(w4.clip(1e-12))
    cur = np.array([0., 0., 0.])
    best = balanced_accuracy_score(y, log_p.argmax(1))
    rounds = 0; improved = True
    while improved and rounds < 12:
        improved = False
        for k in range(3):
            for db in np.linspace(-7.0, 7.0, 29):
                test_b = cur.copy(); test_b[k] = db
                ba_t = balanced_accuracy_score(y, (log_p + test_b).argmax(1))
                if ba_t > best + 1e-6:
                    best = ba_t; cur = test_b; improved = True
        rounds += 1
    print(f"  argmax {balanced_accuracy_score(y, log_p.argmax(1)):.5f}")
    print(f"  tuned (wide grid) {best:.5f}  bias {cur.tolist()}")
    pred_w4 = (log_p + cur).argmax(1)
    w4_errs = (pred_w4 != y).sum()
    print(f"  errs {w4_errs}")

    # Jaccard
    e_w4 = (pred_w4 != y); e_lb = (pred_lb != y)
    jacc = (e_w4 & e_lb).sum() / max((e_w4 | e_lb).sum(), 1)
    print(f"  Jaccard vs LB-best: {jacc:.4f}")

    # Per-class recall
    rec_w4 = [((y == k) & (pred_w4 == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
    rec_lb = [((y == k) & (pred_lb == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
    print(f"  per-class recall (W4):     L {rec_w4[0]:.4f}  M {rec_w4[1]:.4f}  H {rec_w4[2]:.4f}")
    print(f"  per-class recall (LB):     L {rec_lb[0]:.4f}  M {rec_lb[1]:.4f}  H {rec_lb[2]:.4f}")

    # Blend gate vs LB-best at FIXED LB bias
    print("\n--- blend gate: log-blend (LB-best, w4) at fixed LB bias ---")
    print(f"{'α':>6} {'tuned':>9} {'Δ':>9} {'errs':>6} {'recL':>7} {'recM':>7} {'recH':>7} {'flag'}")
    best_alpha = 0.0; best_delta = 0.0
    for alpha in [0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500]:
        blend = lb((final_lb, 1 - alpha), (w4, alpha))
        bp = (np.log(blend.clip(1e-12)) + BIAS).argmax(1)
        ba = balanced_accuracy_score(y, bp)
        delta = ba - lbbest_ba
        errs = (bp != y).sum()
        cm = [((y == k) & (bp == k)).sum() / max((y == k).sum(), 1) for k in (0,1,2)]
        rec_drops = [(cm[k] - rec_lb[k]) for k in (0,1,2)]
        flag = ""
        if delta >= 0.0002 and all(d >= -5e-4 for d in rec_drops):
            flag = "  *** PASS ***"
        elif delta >= 0.0001:
            flag = "  (close)"
        print(f"{alpha:>6.3f} {ba:>9.5f} {delta:>+9.5f} {errs:>6} {cm[0]:>7.4f} {cm[1]:>7.4f} {cm[2]:>7.4f}{flag}")
        if delta > best_delta:
            best_delta = delta; best_alpha = alpha

    # Try iso-cal'd W4
    print("\n--- W4 iso-calibrated, blend gate ---")
    w4_iso = iso(w4, y)
    print(f"{'α':>6} {'tuned':>9} {'Δ':>9} {'errs':>6}")
    best_alpha_iso = 0.0; best_delta_iso = 0.0
    for alpha in [0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500]:
        blend = lb((final_lb, 1 - alpha), (w4_iso, alpha))
        bp = (np.log(blend.clip(1e-12)) + BIAS).argmax(1)
        ba = balanced_accuracy_score(y, bp)
        delta = ba - lbbest_ba
        errs = (bp != y).sum()
        flag = "  *** PASS ***" if delta >= 0.0002 else ("  (close)" if delta >= 0.0001 else "")
        print(f"{alpha:>6.3f} {ba:>9.5f} {delta:>+9.5f} {errs:>6}{flag}")
        if delta > best_delta_iso:
            best_delta_iso = delta; best_alpha_iso = alpha

    print(f"\nBest raw   blend: α={best_alpha:.3f}, Δ={best_delta:+.5f}")
    print(f"Best iso   blend: α={best_alpha_iso:.3f}, Δ={best_delta_iso:+.5f}")

    # Emit if PASS
    bd = max(best_delta, best_delta_iso)
    if bd >= 0.0002:
        if best_delta >= best_delta_iso:
            ba_ = best_alpha; tag = "raw"
            test_blend = lb((test_final_lb, 1 - ba_), (w4_test, ba_))
        else:
            ba_ = best_alpha_iso; tag = "iso"
            w4_test_iso = iso_apply(w4, y, w4_test)
            test_blend = lb((test_final_lb, 1 - ba_), (w4_test_iso, ba_))
        test_bp = (np.log(test_blend.clip(1e-12)) + BIAS).argmax(1)
        labels = np.array(["Low", "Medium", "High"])[test_bp]
        sub_path = SUB / f"submission_lb4_w4_{tag}_a{int(ba_*1000):03d}.csv"
        pd.DataFrame({"id": test_ids, "Irrigation_Need": labels}).to_csv(sub_path, index=False)
        print(f"\n*** EMITTED: {sub_path} ***")
        from collections import Counter
        c = Counter(labels.tolist())
        print(f"  pred dist: {dict(c)}")
        primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
        n_diff = (primary["Irrigation_Need"].values != labels).sum()
        print(f"  rows differing from LB-best primary: {n_diff} ({100*n_diff/len(labels):.3f}%)")
    else:
        print(f"\nNO PASS (best Δ {bd:+.5f} < +0.0002 LB-transfer threshold)")

    out = {
        "lbbest_ba": float(lbbest_ba), "lbbest_errs": int(lbbest_errs),
        "w4_tuned_wide": float(best), "w4_errs": int(w4_errs),
        "jaccard_vs_lbbest": float(jacc),
        "best_raw_alpha": float(best_alpha), "best_raw_delta": float(best_delta),
        "best_iso_alpha": float(best_alpha_iso), "best_iso_delta": float(best_delta_iso),
    }
    out_path = ART / "w4_blend_gate_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nresults → {out_path}")


if __name__ == "__main__":
    main()
