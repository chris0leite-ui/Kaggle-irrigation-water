"""H-combined blend gate: evaluate H1/H2/H4/H5 candidates against v1 LB-best.

For each candidate (and combinations):
  - Standalone OOF tuned bal_acc + per-class recall
  - Test argmax diff vs v1 LB-best
  - Geomean bag with v1 LB-best at multiple alpha
  - 4-gate filter (G1 +0.0003, G2 PCR floor, G3 dual-alpha ratio, G4 net_H + asymmetry)
  - Emit submission only if all 4 gates pass

Anchors:
  - v1 LB-best (LB 0.98129)
  - rawashishsin v3 (LB 0.98109)

Inputs (run AFTER H1/H2/H4/H5 finish):
  - oof_h1_seedbag_rf.npy + test_h1_seedbag_rf.npy
  - oof_h2_et_natural.npy + test_h2_et_natural.npy
  - oof_h4_S{1,2,3}.npy + test_h4_S{1,2,3}.npy
  - oof_h5_hp_bag.npy + test_h5_hp_bag.npy

Usage:
  python scripts/h_combined_blend_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def four_gate_check(name, cand_oof, cand_test, anchor_oof, anchor_test,
                    anchor_bal, y, n_te, anchor_bias):
    """Run 4-gate filter for a single candidate vs an anchor."""
    print(f"\n=== 4-gate analysis: {name} vs anchor (anchor OOF tuned={anchor_bal:.5f}) ===")
    sweep = []
    a_pred = (safelog(anchor_test) + anchor_bias).argmax(1)
    a_pcr_oof = per_class_recall(y, (safelog(anchor_oof) + anchor_bias).argmax(1))
    print(f"  anchor PCR=[L={a_pcr_oof[0]:.4f} M={a_pcr_oof[1]:.4f} H={a_pcr_oof[2]:.4f}]")

    for alpha in [0.10, 0.20, 0.30, 0.40, 0.50]:
        # geomean log-blend at fixed anchor bias
        log_blend_oof = (1.0 - alpha) * safelog(anchor_oof) + alpha * safelog(cand_oof)
        log_blend_test = (1.0 - alpha) * safelog(anchor_test) + alpha * safelog(cand_test)
        blend_oof = _normed(np.exp(log_blend_oof))
        blend_test = _normed(np.exp(log_blend_test))

        # Evaluate at anchor's tuned bias
        blend_pred_oof = (safelog(blend_oof) + anchor_bias).argmax(1)
        blend_pred_test = (safelog(blend_test) + anchor_bias).argmax(1)
        bal = balanced_accuracy_score(y, blend_pred_oof)
        delta = bal - anchor_bal
        pcr = per_class_recall(y, blend_pred_oof)
        pcr_d = (pcr - a_pcr_oof).tolist()

        # Test-side diff & net_H + asymmetry
        diff = int((blend_pred_test != a_pred).sum())
        h_added = int(((blend_pred_test == 2) & (a_pred != 2)).sum())
        h_removed = int(((a_pred == 2) & (blend_pred_test != 2)).sum())
        net_h = h_added - h_removed
        churn_h = h_added + h_removed
        ratio = abs(net_h) / max(1, churn_h)

        # 4-gate
        g1 = delta >= 3e-4
        g2 = all(d >= -5e-4 for d in pcr_d)
        # G3: dual-alpha needs alpha=0.4 vs 0.3 ratio (handled below)
        g4 = (net_h > 0) and (ratio >= 0.5)

        sweep.append({
            "alpha": alpha, "bal_acc": float(bal),
            "delta": float(delta), "pcr_delta": pcr_d,
            "test_diff": diff, "h_added": h_added,
            "h_removed": h_removed, "net_h": net_h,
            "churn_h": churn_h, "asym_ratio": float(ratio),
            "g1": g1, "g2": g2, "g4": g4,
            "blend_test": blend_test,  # keep for emission
            "blend_pred_test": blend_pred_test,
        })
        line = (f"  alpha={alpha:.2f}  bal={bal:.5f}  d={delta:+.5f}  "
                f"PCR=[{pcr[0]:.4f},{pcr[1]:.4f},{pcr[2]:.4f}]  "
                f"diff={diff}  net_H={net_h:+d} ratio={ratio:.3f}  "
                f"g1={g1} g2={g2} g4={g4}")
        print(line)

    # G3 dual-alpha check: alpha=0.4 / alpha=0.3 ratio in [1.0, 2.0]
    a30 = next((s for s in sweep if abs(s["alpha"] - 0.30) < 1e-3), None)
    a40 = next((s for s in sweep if abs(s["alpha"] - 0.40) < 1e-3), None)
    if a30 and a40 and a30["delta"] != 0:
        g3_ratio = a40["delta"] / a30["delta"]
        for s in sweep:
            s["g3_ratio"] = float(g3_ratio)
            s["g3"] = (1.0 <= g3_ratio <= 2.0) if a30["delta"] > 0 else False
        print(f"  G3 dual-alpha ratio (a40/a30)={g3_ratio:.3f}")

    # Find best gate-pass alpha
    best = None
    for s in sweep:
        if s["g1"] and s["g2"] and s.get("g3", False) and s["g4"]:
            if best is None or s["delta"] > best["delta"]:
                best = s
    if best:
        print(f"  *** ALL-4-GATE PASS at alpha={best['alpha']:.2f}, delta={best['delta']:+.5f} ***")
    else:
        # report best 3-of-4 pass
        for s in sweep:
            n_pass = sum([s["g1"], s["g2"], s.get("g3", False), s["g4"]])
            s["n_pass"] = n_pass
        sweep.sort(key=lambda s: (s.get("n_pass", 0), s["delta"]), reverse=True)
        s = sweep[0]
        print(f"  best 3/4 gate pass: alpha={s['alpha']:.2f}  d={s['delta']:+.5f}  passes={s.get('n_pass',0)}/4")

    return sweep, best


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    prior = np.bincount(y, minlength=3) / len(y)

    # v1 LB-best anchor
    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    print(f"v1 LB-best  OOF tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")

    # H1 candidate (if produced)
    h1_oof_p = ART / "oof_h1_seedbag_rf.npy"
    h2_oof_p = ART / "oof_h2_et_natural.npy"
    h4_oof_p = ART / "oof_h4_S1.npy"
    h5_oof_p = ART / "oof_h5_hp_bag.npy"

    histgbm_oof_p = ART / "oof_h_histgbm_natural.npy"

    candidates = []
    for name, p in [("H1_seedbag", h1_oof_p), ("H2_et", h2_oof_p),
                    ("H4_S1", h4_oof_p), ("H5_hpbag", h5_oof_p),
                    ("HistGBM", histgbm_oof_p)]:
        if p.exists():
            oof = _normed(np.load(p).astype(np.float32))
            test_p = _normed(np.load(ART / p.name.replace("oof_", "test_")).astype(np.float32))
            bias, tuned = tune_log_bias(oof, y, prior)
            print(f"  {name}: OOF tuned={tuned:.5f}  bias={bias.round(4).tolist()}")
            candidates.append((name, oof, test_p, bias, tuned))
        else:
            print(f"  {name}: not yet produced")

    if not candidates:
        print("\nNo candidates ready. Run H1-H5 production first.")
        return

    # 4-gate vs v1 LB-best
    results = {}
    for name, oof, test_p, _, _ in candidates:
        sweep, best = four_gate_check(
            name, oof, test_p, v1_oof, v1_test, v1_tuned, y, n_te, v1_bias
        )
        results[name] = {
            "sweep": [{k: v for k, v in s.items() if k not in ("blend_test", "blend_pred_test")} for s in sweep],
            "best_pass": ({k: v for k, v in best.items() if k not in ("blend_test", "blend_pred_test")} if best else None),
        }

        # If gate passes, emit submission
        if best is not None:
            sub_path = SUB / f"submission_{name}_a{int(best['alpha']*100):03d}_blend_v1.csv"
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in best["blend_pred_test"]],
            })
            sub.to_csv(sub_path, index=False)
            print(f"  EMIT: {sub_path}")
            results[name]["sub_path"] = str(sub_path)

    # Bag-of-bags: if H1/H2/H4/HistGBM ALL produced, geomean them
    if len([n for n, _, _, _, _ in candidates if n in ("H1_seedbag", "H2_et", "H4_S1", "HistGBM")]) >= 2:
        print("\n=== Mega-bag: geomean of all candidates ===")
        sel = [(name, o, t) for name, o, t, _, _ in candidates if name in ("H1_seedbag", "H2_et", "H4_S1", "HistGBM")]
        log_oofs = np.stack([safelog(o) for _, o, _ in sel], axis=0)
        log_tests = np.stack([safelog(t) for _, _, t in sel], axis=0)
        mega_oof = _normed(np.exp(log_oofs.mean(axis=0)))
        mega_test = _normed(np.exp(log_tests.mean(axis=0)))
        mega_bias, mega_tuned = tune_log_bias(mega_oof, y, prior)
        print(f"  mega-bag: OOF tuned={mega_tuned:.5f}  bias={mega_bias.round(4).tolist()}")
        sweep, best = four_gate_check(
            "MEGA_BAG", mega_oof, mega_test, v1_oof, v1_test,
            v1_tuned, y, n_te, v1_bias
        )
        results["MEGA_BAG"] = {
            "components": [n for n, _, _ in sel],
            "sweep": [{k: v for k, v in s.items() if k not in ("blend_test", "blend_pred_test")} for s in sweep],
            "best_pass": ({k: v for k, v in best.items() if k not in ("blend_test", "blend_pred_test")} if best else None),
        }
        if best is not None:
            sub_path = SUB / f"submission_mega_bag_a{int(best['alpha']*100):03d}_blend_v1.csv"
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in best["blend_pred_test"]],
            })
            sub.to_csv(sub_path, index=False)
            print(f"  EMIT: {sub_path}")
            results["MEGA_BAG"]["sub_path"] = str(sub_path)

    with open(ART / "h_combined_blend_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {ART}/h_combined_blend_gate_results.json")


if __name__ == "__main__":
    main()
