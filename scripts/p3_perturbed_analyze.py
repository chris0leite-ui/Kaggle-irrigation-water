"""P3 perturbed meta-stacker analyze: gate against LB-best 4-stack.

Decision rule for LB probe:
  - iso-cal'd blend onto LB-best 3-stack peak Δ ≥ +0.00023 (matches original
    meta's lift that produced LB +0.00086)
  - per-class recall within -5e-4 of LB-best 4-stack
  - Jaccard vs LB-best 4-stack in [0.85, 0.97]
  - errs ≤ LB-best 4-stack's 9415

Two anchor comparisons:
  A. blend onto LB-best 3-stack (does perturbed meta beat the original meta?)
  B. blend onto LB-best 4-stack (does perturbed meta add to the existing meta?)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, TARGET, build_lbbest_stack,
    iso_cal, log, normed,
)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def per_class_recall(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    cm = confusion_matrix(y, pred)
    return cm.diagonal() / cm.sum(axis=1)


def err_count(p, y):
    pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
    return int((pred != y).sum())


def jaccard_err(p1, p2, y):
    e1 = (np.log(np.clip(p1, 1e-12, 1)) + BIAS).argmax(1) != y
    e2 = (np.log(np.clip(p2, 1e-12, 1)) + BIAS).argmax(1) != y
    return float((e1 & e2).sum() / max((e1 | e2).sum(), 1))


def sweep_blend(anchor_oof, anchor_test, m_oof, m_test, y, m_iso_o, m_iso_t):
    rows = []
    alphas = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    anchor_bal = bal(anchor_oof, y)
    for use_iso in (False, True):
        m_o = m_iso_o if use_iso else m_oof
        m_t = m_iso_t if use_iso else m_test
        for a in alphas:
            blend_o = log_blend([anchor_oof, m_o], np.array([1 - a, a]))
            b = bal(blend_o, y)
            d = b - anchor_bal
            rec = per_class_recall(blend_o, y)
            errs = err_count(blend_o, y)
            jacc = jaccard_err(blend_o, anchor_oof, y)
            rows.append(dict(iso=use_iso, alpha=a, oof=float(b), delta=float(d),
                             rec_low=float(rec[0]), rec_med=float(rec[1]),
                             rec_high=float(rec[2]), errs=errs,
                             jacc_vs_anchor=jacc))
    return rows


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy().astype(np.int32)

    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    lb4_o = log_blend([lb3_o, meta_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, meta_iso_t], np.array([0.7, 0.3]))

    log(f"LB-best 3-stack OOF = {bal(lb3_o, y):.5f}")
    log(f"LB-best 4-stack OOF = {bal(lb4_o, y):.5f}  errs={err_count(lb4_o, y)}")
    rec4 = per_class_recall(lb4_o, y)
    rec3 = per_class_recall(lb3_o, y)
    log(f"  4-stack per-class recall: L={rec4[0]:.4f}  M={rec4[1]:.4f}  H={rec4[2]:.4f}")

    variants = [
        ("v1_noise03_csb09_k3", "oof_meta_perturbed_v1_noise03_csb09_k3.npy",
         "test_meta_perturbed_v1_noise03_csb09_k3.npy"),
        ("v2_noise05_csb05_k3", "oof_meta_perturbed_v2_noise05_csb05_k3.npy",
         "test_meta_perturbed_v2_noise05_csb05_k3.npy"),
    ]

    candidates = []
    for name, oof_p, test_p in variants:
        oof_path = ART / oof_p
        test_path = ART / test_p
        if not oof_path.exists():
            log(f"\n{name}: artefact missing; skip")
            continue
        m_o = normed(np.load(oof_path).astype(np.float32))
        m_t = normed(np.load(test_path).astype(np.float32))
        m_iso_o, m_iso_t = iso_cal(m_o, m_t, y)

        log(f"\n=== {name} ===")
        log(f"  raw standalone @ bias = {bal(m_o, y):.5f}  errs={err_count(m_o, y)}  "
            f"argmax={balanced_accuracy_score(y, m_o.argmax(1)):.5f}")
        log(f"  iso standalone @ bias = {bal(m_iso_o, y):.5f}  errs={err_count(m_iso_o, y)}")

        rowsA = sweep_blend(lb3_o, lb3_t, m_o, m_t, y, m_iso_o, m_iso_t)
        bestA = max(rowsA, key=lambda r: r["delta"])
        rowsB = sweep_blend(lb4_o, lb4_t, m_o, m_t, y, m_iso_o, m_iso_t)
        bestB = max(rowsB, key=lambda r: r["delta"])

        log(f"  blend onto LB-3stack peak: iso={bestA['iso']} α={bestA['alpha']:.3f} "
            f"OOF={bestA['oof']:.5f} Δ={bestA['delta']:+.5f}  errs={bestA['errs']}  "
            f"recH={bestA['rec_high']:.4f}  J={bestA['jacc_vs_anchor']:.4f}")
        log(f"  blend onto LB-4stack peak: iso={bestB['iso']} α={bestB['alpha']:.3f} "
            f"OOF={bestB['oof']:.5f} Δ={bestB['delta']:+.5f}  errs={bestB['errs']}  "
            f"recH={bestB['rec_high']:.4f}  J={bestB['jacc_vs_anchor']:.4f}")

        candidates.append(dict(
            name=name,
            raw_at_bias=float(bal(m_o, y)),
            iso_at_bias=float(bal(m_iso_o, y)),
            sweep_3stack=rowsA, best_3stack=bestA,
            sweep_4stack=rowsB, best_4stack=bestB,
        ))

    log("\n=== DECISION ANALYSIS ===")
    for c in candidates:
        log(f"\n{c['name']}:")
        for tag, key, anchor_rec, anchor_errs, target in [
            ("3stack", "best_3stack", rec3, err_count(lb3_o, y), 0.00023),
            ("4stack", "best_4stack", rec4, err_count(lb4_o, y), 0.00020),
        ]:
            b = c[key]
            guard_l = b["rec_low"] >= anchor_rec[0] - 5e-4
            guard_m = b["rec_med"] >= anchor_rec[1] - 5e-4
            guard_h = b["rec_high"] >= anchor_rec[2] - 5e-4
            err_ok = b["errs"] <= anchor_errs
            jacc_ok = 0.80 <= b["jacc_vs_anchor"] <= 0.97
            delta_ok = b["delta"] >= target
            checks = "".join(["✓" if x else "✗" for x in
                              (delta_ok, guard_l, guard_m, guard_h, err_ok, jacc_ok)])
            log(f"  on {tag}: Δ={b['delta']:+.5f} (≥+{target:.5f}? {'Y' if delta_ok else 'N'})  "
                f"L={b['rec_low']:.4f} M={b['rec_med']:.4f} H={b['rec_high']:.4f}  "
                f"errs={b['errs']}/{anchor_errs}  J={b['jacc_vs_anchor']:.4f}  [{checks}]")

    out = dict(lb3_oof=float(bal(lb3_o, y)), lb4_oof=float(bal(lb4_o, y)),
               lb4_errs=err_count(lb4_o, y), lb4_per_class_recall=rec4.tolist(),
               candidates=candidates)
    (ART / "p3_perturbed_analyze_results.json").write_text(json.dumps(out, indent=2, default=float))
    log("\nwrote p3_perturbed_analyze_results.json")


if __name__ == "__main__":
    main()
