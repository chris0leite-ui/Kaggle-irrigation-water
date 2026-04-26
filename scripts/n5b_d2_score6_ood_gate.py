"""Deployment #2: OOD-gated score=6 ∩ teacher_argmax=Medium override.

Combines spec6_v2_prob (binary M->H detector, AUC 0.94) with the
10k-OOD score (this branch) as a 2-D conformal gate. Hypothesis:
the score=6 deep-dive (2026-04-26) closed because missed-H rows were
feature-indistinguishable from M rows IN AVAILABLE features; OOD
score is a NEW feature dimension that may break that ceiling.

Mechanism:
  1) Reconstruct LB-best 4-stack primary (OOF 0.98084, LB 0.98094).
  2) Restrict to override domain: dgp_score==6 AND teacher_argmax==Medium.
  3) For each (theta_spec, theta_ood) over a grid, override domain rows
     to High where (spec_v2_prob > theta_spec) AND (ood_score > theta_ood).
  4) Score macro-recall delta vs primary on OOF; gate at +0.0002 AND
     per-class recall guardrail (each >= anchor - 5e-4).
  5) If passes, deploy override on test, save submission, ASK BEFORE LB.

No retraining; runs on saved artefacts in <1 min CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed
from dgp_formula import dgp_score as dgp_score_fn

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(parents=True, exist_ok=True)
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def build_primary_4stack(y):
    """LB-best 4-stack (LB 0.98094) = 3-stack + xgb_metastack__iso @ alpha=0.30."""
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)
    p4_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    p4_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))
    return p4_o, p4_t


def main() -> None:
    print("[1] Loading y, dgp_score, OOD scores, spec6_v2...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = load_y()
    score_train = dgp_score_fn(train)
    score_test = dgp_score_fn(test)
    ood_train = np.load(ART / "oof_ood3_train.npy")  # (N, 3)
    ood_test = np.load(ART / "test_ood3.npy")
    spec_train = np.load(ART / "oof_spec6_mh_v2.npy")  # P(y=H|x, score=6)
    spec_test = np.load(ART / "test_spec6_mh_v2.npy")

    # spec6_v2 only fires on score=6 rows; off-domain rows are 0 or near-zero.
    # Verify it's a 1-D prob array.
    if spec_train.ndim == 2:
        spec_train = spec_train[:, -1]  # last col = P(High)
    if spec_test.ndim == 2:
        spec_test = spec_test[:, -1]
    print(f"    spec_train mean={spec_train.mean():.4f} max={spec_train.max():.4f}")

    # Use GMM neg-log-p (col 0) as the OOD score — strongest signal in N5b.
    ood_train_s = ood_train[:, 0]
    ood_test_s = ood_test[:, 0]

    print("[2] Reconstructing LB-best 4-stack primary...")
    p_oof, p_test = build_primary_4stack(y)
    pred_oof = (np.log(np.clip(p_oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_test = (np.log(np.clip(p_test, 1e-12, 1)) + BIAS).argmax(1)
    base_macro = balanced_accuracy_score(y, pred_oof)
    base_recall = recall_score(y, pred_oof, average=None)
    print(f"    primary OOF macro={base_macro:.5f} recall={base_recall.round(5)}")

    print("[3] Identifying override domain (score=6, teacher_argmax=Medium)...")
    dom_train = (score_train == 6) & (pred_oof == 1)
    dom_test = (score_test == 6) & (pred_test == 1)
    n_dom_train = int(dom_train.sum())
    n_dom_test = int(dom_test.sum())
    n_truly_h_in_dom = int(((y == 2) & dom_train).sum())
    print(f"    train domain n={n_dom_train}  truly-H={n_truly_h_in_dom} "
          f"(prevalence {n_truly_h_in_dom/max(1,n_dom_train)*100:.2f}%)")
    print(f"    test  domain n={n_dom_test}")
    print(f"    macro-recall break-even precision = "
          f"{n_truly_h_in_dom/max(1, n_dom_train - n_truly_h_in_dom)*100:.2f}%")

    print("[4] 2-D theta sweep on (spec_prob, ood_score)...")
    spec_qs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    ood_qs = [-1e9, np.quantile(ood_train_s[dom_train], 0.50),
              np.quantile(ood_train_s[dom_train], 0.70),
              np.quantile(ood_train_s[dom_train], 0.85),
              np.quantile(ood_train_s[dom_train], 0.95)]
    ood_labels = ["all", "p50", "p70", "p85", "p95"]

    rows = []
    best = None
    for ts in spec_qs:
        for to, tlab in zip(ood_qs, ood_labels):
            mask_train = dom_train & (spec_train > ts) & (ood_train_s > to)
            n = int(mask_train.sum())
            if n == 0:
                continue
            new_pred = pred_oof.copy()
            new_pred[mask_train] = 2  # High
            macro = balanced_accuracy_score(y, new_pred)
            recs = recall_score(y, new_pred, average=None)
            correct = int(((y == 2) & mask_train).sum())
            prec = correct / max(1, n)
            d_macro = macro - base_macro
            d_rec = recs - base_recall
            guardrail_pass = bool(np.all(d_rec >= -5e-4))
            row = dict(theta_spec=float(ts), theta_ood=tlab,
                       n_overrides=n, correct=correct, precision=round(prec, 4),
                       d_macro=round(d_macro, 6),
                       d_rec=[round(float(x), 6) for x in d_rec],
                       guardrail_pass=guardrail_pass)
            rows.append(row)
            if best is None or (guardrail_pass and d_macro > best["d_macro"]):
                if guardrail_pass:
                    best = row
            print(f"  ts={ts:.2f} to={tlab:>3s} n={n:5d} prec={prec:.3f} "
                  f"dM={d_macro:+.5f} drec={d_rec.round(5)} "
                  f"{'OK' if guardrail_pass else 'fail-guard'}")

    print("\n[5] BEST CONFIG (under guardrail):")
    if best is None:
        print("  No config passes per-class guardrail. CLOSED.")
    else:
        print(f"  ts={best['theta_spec']} to={best['theta_ood']} "
              f"n={best['n_overrides']} prec={best['precision']} "
              f"dM={best['d_macro']:+.5f}")
        gate_pass = best["d_macro"] >= 2e-4
        print(f"  +0.0002 gate: {'PASS' if gate_pass else 'FAIL (sub-gate)'}")

        if gate_pass:
            # Deploy on test with same thresholds.
            ts = best["theta_spec"]; tlab = best["theta_ood"]
            to = ood_qs[ood_labels.index(tlab)]
            mask_test = dom_test & (spec_test > ts) & (ood_test_s > to)
            n_test_over = int(mask_test.sum())
            print(f"  test overrides: n={n_test_over}")
            if n_test_over >= 10:
                new_test = pred_test.copy()
                new_test[mask_test] = 2
                LABELS = ["Low", "Medium", "High"]
                sub = pd.DataFrame({
                    "id": test["id"].values,
                    "Irrigation_Need": [LABELS[i] for i in new_test],
                })
                fname = (f"submission_n5b_d2_score6_ood_ts{int(ts*100):02d}_"
                         f"to{tlab}.csv")
                sub.to_csv(SUB / fname, index=False)
                print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")
            else:
                print(f"  test overrides < 10; not deploying.")

    out_path = ART / "n5b_d2_score6_ood_gate_results.json"
    with open(out_path, "w") as f:
        json.dump({"base_macro": base_macro, "base_recall": base_recall.tolist(),
                   "n_dom_train": n_dom_train, "n_dom_test": n_dom_test,
                   "n_truly_h_in_dom": n_truly_h_in_dom,
                   "rows": rows, "best": best}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
