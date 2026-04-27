"""Ensemble v1 unweighted meta + classw meta in LB-validated PRIMARY arch.

primary' = 0.7 × LB3 + 0.3 × log_blend(v1_iso, classw_iso, weights=[w1, w2])

Sweep weight balance between v1 (LB-validated, tight calibration) and
classw (orthogonal, wider OOF lift). Test if mean-blend captures the
best of both metas.

Compare per-class trade across all variants.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def macro(p, y, b=BIAS):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + b).argmax(1))


def main() -> None:
    y = load_y()
    s3_o, s3_t = build_lbbest_stack(y)

    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)

    classw_o = normed(np.load(ART / "oof_xgb_metastack_classw.npy"))
    classw_t = normed(np.load(ART / "test_xgb_metastack_classw.npy"))
    classw_iso_o, classw_iso_t = iso_cal(classw_o, classw_t, y)

    # Baseline: v1 PRIMARY (LB 0.98094)
    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    pred_v1 = (np.log(np.clip(p_v1_o, 1e-12, 1)) + BIAS).argmax(1)
    m_v1 = balanced_accuracy_score(y, pred_v1)
    rec_v1 = recall_score(y, pred_v1, average=None)
    print(f"v1 PRIMARY (LB 0.98094): OOF={m_v1:.5f}  rec={rec_v1.round(5)}")

    # Sweep mean-blend weights between v1 and classw
    print("\nMean-blend (geometric, log-space) of v1 + classw at α=0.30:")
    print(f"  {'w_classw':>9}  {'OOF':>7}  {'Δ':>9}  {'errs':>5}  PCR  {'guard':>5}  {'emit':>4}")

    out = {"baseline_oof": float(m_v1), "rows": []}
    best_emit = None
    for w_classw in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        w_v1 = 1 - w_classw
        meta_blend_o = log_blend([v1_iso_o, classw_iso_o], np.array([w_v1, w_classw]))
        meta_blend_t = log_blend([v1_iso_t, classw_iso_t], np.array([w_v1, w_classw]))
        p_o = log_blend([s3_o, meta_blend_o], np.array([0.70, 0.30]))
        p_t = log_blend([s3_t, meta_blend_t], np.array([0.70, 0.30]))
        pred = (np.log(np.clip(p_o, 1e-12, 1)) + BIAS).argmax(1)
        m = balanced_accuracy_score(y, pred)
        rec = recall_score(y, pred, average=None)
        errs = int((pred != y).sum())
        d = m - m_v1
        drec = (rec - rec_v1).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"  {w_classw:>9.2f}  {m:.5f}  {d:+.5f}  {errs:5d}  "
              f"{rec.round(4).tolist()}  {'PASS' if guard else 'FAIL'}  "
              f"{'YES' if emit else 'no'}{marker}")
        out["rows"].append({"w_classw": w_classw, "oof": float(m), "d": float(d),
                             "errs": errs, "rec": rec.tolist(),
                             "drec": drec.tolist(), "guard": guard, "emit": emit})
        if emit and (best_emit is None or d > best_emit["d"]):
            best_emit = {"w_classw": w_classw, "d": d, "p_t": p_t}

    if best_emit is not None:
        w = best_emit["w_classw"]
        pred_test = (np.log(np.clip(best_emit["p_t"], 1e-12, 1)) + BIAS).argmax(1)
        pred_v1_test = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_test != pred_v1_test).sum())
        test_df = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test_df["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_test]})
        fname = f"submission_meta_ensemble_v1_classw_w{int(w*100):03d}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"\n  test diff vs v1 PRIMARY: {n_diff}")
        print(f"  -> SAVED {fname}")
    else:
        print("\nNo blend passes both gates; no submission emitted")

    out_path = ART / "meta_ensemble_v1_classw_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
