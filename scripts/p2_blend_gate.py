"""P2 soft-logit-add blend gate.

Mechanism: take LB-best 4-stack OOF/test, decompose into log-probs, add
λ_3 × spec3_z to the Medium column for rows in score=3 bucket, add λ_6 ×
spec6_z to the High column for rows in score=6 bucket. spec_z is the
specialist's logit minus its global mean (centered, so λ=0 leaves the
anchor unchanged).

Surgical: only ~140k of 630k train rows are touched; the remaining ~490k
rows are the LB-best 4-stack predictions verbatim.

Sweep (λ_3, λ_6) on a 2D grid; pick the (λ_3, λ_6) that maximizes
fixed-bias macro-recall subject to per-class recall guardrail.

Gates:
  G1. ∃ (λ_3, λ_6) in interior such that blend OOF > LB-best 4-stack OOF
  G2. blend errs ≤ LB-best 4-stack errs at chosen (λ_3, λ_6)
  G3. PCR floor ≥ LB-best 4-stack PCR − 5e-4 each class
  G4. Δ vs LB-best 4-stack ≥ +2e-4 AND Jaccard < 0.97

Outputs:
  scripts/artifacts/p2_blend_gate_results.json
  submissions/submission_p2_l3{λ_3}_l6{λ_6}.csv  (only if all gates pass)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, CLS2IDX, add_distance_features  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, DATA, SUB, BIAS, build_lbbest_stack, iso_cal, log, bal_at_bias,
)

CLASSES = ["Low", "Medium", "High"]
GATE_LB_DELTA = 2e-4
GATE_JACC = 0.97
PCR_FLOOR_DELTA = 5e-4
LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def predict(p, bias=BIAS):
    return (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return cm.diagonal() / cm.sum(axis=1).clip(min=1)


def to_logit(p):
    return np.log(np.clip(p, 1e-9, 1.0)) - np.log(np.clip(1.0 - p, 1e-9, 1.0))


def apply_soft_add(stack_oof, stack_test, score_tr, score_te,
                   spec3_oof, spec3_test, spec6_oof, spec6_test,
                   lambda3: float, lambda6: float):
    """Return (oof_blend, test_blend) with the soft logit-add applied.

    The anchor's log-probs get λ_b * (spec_logit − bucket_mean(spec_logit))
    added to the target class column for rows in the bucket. Re-normalize
    via softmax over the 3 classes.
    """
    log_p_tr = np.log(np.clip(stack_oof, 1e-12, 1.0))
    log_p_te = np.log(np.clip(stack_test, 1e-12, 1.0))

    # score=3 bucket → add to Medium column (idx 1)
    if lambda3 != 0:
        m_tr = (score_tr == 3)
        m_te = (score_te == 3)
        s3_z_tr = to_logit(spec3_oof[m_tr])
        s3_z_te = to_logit(spec3_test[m_te])
        # center on full-train mean (computed only over bucket-train rows)
        c3 = float(np.nanmean(s3_z_tr))
        log_p_tr[m_tr, 1] += lambda3 * (s3_z_tr - c3)
        log_p_te[m_te, 1] += lambda3 * (s3_z_te - c3)

    # score=6 bucket → add to High column (idx 2)
    if lambda6 != 0:
        m_tr = (score_tr == 6)
        m_te = (score_te == 6)
        s6_z_tr = to_logit(spec6_oof[m_tr])
        s6_z_te = to_logit(spec6_test[m_te])
        c6 = float(np.nanmean(s6_z_tr))
        log_p_tr[m_tr, 2] += lambda6 * (s6_z_tr - c6)
        log_p_te[m_te, 2] += lambda6 * (s6_z_te - c6)

    # Re-normalize via softmax in log space
    p_tr = np.exp(log_p_tr - log_p_tr.max(axis=1, keepdims=True))
    p_tr /= p_tr.sum(axis=1, keepdims=True)
    p_te = np.exp(log_p_te - log_p_te.max(axis=1, keepdims=True))
    p_te /= p_te.sum(axis=1, keepdims=True)
    return p_tr, p_te


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    # Anchors
    log("loading anchors")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    mv1 = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    mv1_te = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    mv1_iso, mv1_iso_te = iso_cal(mv1, mv1_te, y)
    lb4_oof = log_blend([lb3_oof, mv1_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv1_iso_te], np.array([0.7, 0.3]))
    lb4_bal = bal_at_bias(lb4_oof, y)
    pred_lb4 = predict(lb4_oof)
    pcr_lb4 = per_class_recall(y, pred_lb4)
    pcr_floor = pcr_lb4 - PCR_FLOOR_DELTA
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")
    log(f"  PCR  [L,M,H] = {pcr_lb4.round(5).tolist()}")
    log(f"  floor [L,M,H] = {pcr_floor.round(5).tolist()}")

    # Specialists (NaN outside bucket)
    s3_oof = np.load(ART / "oof_p2_score3.npy")
    s3_te = np.load(ART / "test_p2_score3.npy")
    s6_oof = np.load(ART / "oof_p2_score6.npy")
    s6_te = np.load(ART / "test_p2_score6.npy")

    score_tr = add_distance_features(train)["dgp_score"].to_numpy().astype(np.int8)
    score_te = add_distance_features(test)["dgp_score"].to_numpy().astype(np.int8)

    # Sanity: bucket counts + within-bucket AUCs
    from sklearn.metrics import roc_auc_score
    m3_tr = (score_tr == 3); m6_tr = (score_tr == 6)
    auc3 = roc_auc_score((y[m3_tr] == 1).astype(int), s3_oof[m3_tr])
    auc6 = roc_auc_score((y[m6_tr] == 2).astype(int), s6_oof[m6_tr])
    log(f"  spec3 within-bucket OOF AUC = {auc3:.5f}  (n={m3_tr.sum()}, pos={int((y[m3_tr]==1).sum())})")
    log(f"  spec6 within-bucket OOF AUC = {auc6:.5f}  (n={m6_tr.sum()}, pos={int((y[m6_tr]==2).sum())})")

    # 2D sweep
    log(f"\n=== 2D λ sweep (λ_3 × λ_6, fixed recipe bias) ===")
    rows = []
    for l3 in LAMBDAS:
        for l6 in LAMBDAS:
            p_tr, _ = apply_soft_add(lb4_oof, lb4_test, score_tr, score_te,
                                      s3_oof, s3_te, s6_oof, s6_te, l3, l6)
            bal = bal_at_bias(p_tr, y)
            pb = predict(p_tr)
            pcr_b = per_class_recall(y, pb)
            errs = int((pb != y).sum())
            e1 = pred_lb4 != y; e2 = pb != y
            jacc = (e1 & e2).sum() / max((e1 | e2).sum(), 1)
            rows.append({"lambda3": l3, "lambda6": l6, "oof": float(bal),
                         "delta": float(bal - lb4_bal), "errs": errs,
                         "pcr": pcr_b.round(5).tolist(), "jaccard": float(jacc)})

    # Top 10 by OOF
    rows_sorted = sorted(rows, key=lambda r: -r["oof"])
    log(f"\n=== top 10 by OOF ===")
    log(f"{'l3':>5} {'l6':>5} {'OOF':>9} {'Δ':>9} {'errs':>5} {'Jacc':>6}  PCR[L,M,H]")
    for r in rows_sorted[:10]:
        log(f"{r['lambda3']:>5.2f} {r['lambda6']:>5.2f} {r['oof']:>9.5f} "
            f"{r['delta']:>+9.5f} {r['errs']:>5d} {r['jaccard']:>6.4f}  "
            f"{[round(p,4) for p in r['pcr']]}")

    # Best subject to PCR guardrail
    feasible = [r for r in rows
                if all(r["pcr"][i] >= pcr_floor[i] for i in range(3))
                and not (r["lambda3"] == 0 and r["lambda6"] == 0)]
    if not feasible:
        log("\nNo feasible (l3, l6) — guardrail rejects every interior point")
        best_feasible = max(rows, key=lambda r: r["delta"])
    else:
        best_feasible = max(feasible, key=lambda r: r["delta"])
    log(f"\nbest_feasible: λ_3={best_feasible['lambda3']:.2f}  λ_6={best_feasible['lambda6']:.2f}  "
        f"Δ={best_feasible['delta']:+.5f}  PCR={best_feasible['pcr']}")

    # Gate
    g1 = best_feasible["delta"] > 0
    g2 = best_feasible["errs"] <= int((pred_lb4 != y).sum())
    g3 = all(best_feasible["pcr"][i] >= pcr_floor[i] for i in range(3))
    g4 = (best_feasible["delta"] >= GATE_LB_DELTA and best_feasible["jaccard"] < GATE_JACC)
    log(f"\nG1 (Δ > 0):      {g1}")
    log(f"G2 (errs ≤ {int((pred_lb4 != y).sum())}): {g2}")
    log(f"G3 (PCR floor):  {g3}")
    log(f"G4 (Δ ≥ {GATE_LB_DELTA} & Jacc < {GATE_JACC}): {g4}  Δ={best_feasible['delta']:+.5f}  J={best_feasible['jaccard']:.4f}")
    emit = bool(g1 and g2 and g3 and g4)
    log(f"\n=== EMIT: {emit} ===")

    if emit:
        l3, l6 = best_feasible["lambda3"], best_feasible["lambda6"]
        _, p_te = apply_soft_add(lb4_oof, lb4_test, score_tr, score_te,
                                  s3_oof, s3_te, s6_oof, s6_te, l3, l6)
        pred_t = predict(p_te)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub["Irrigation_Need"] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_p2_l3{int(l3*100):03d}_l6{int(l6*100):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"WROTE {path}  (still requires explicit user approval before submit)")

    out = dict(
        lb4_bal=float(lb4_bal), lb4_pcr=pcr_lb4.tolist(),
        pcr_floor=pcr_floor.tolist(),
        spec3_auc=float(auc3), spec6_auc=float(auc6),
        sweep=rows, best_feasible=best_feasible,
        gates=dict(g1=g1, g2=g2, g3=g3, g4=g4, emit=emit),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "p2_blend_gate_results.json").write_text(json.dumps(out, indent=2))
    log(f"\nwrote {ART / 'p2_blend_gate_results.json'}")


if __name__ == "__main__":
    main()
