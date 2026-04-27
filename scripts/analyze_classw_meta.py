"""Analyze class-weighted XGB meta-stacker in the LB-validated PRIMARY arch.

Primary v1 (LB 0.98094): 0.7 * lb3 + 0.3 * meta_iso (where meta_iso is
the v1 unweighted meta-stacker's iso-cal output).

Test: replace v1 meta with the new class-weighted meta. Both architectures
fixed at α=0.30 (LB-validated). Compare:
  - OOF macro
  - Per-class recall (especially High recall — class-weighted should boost it)
  - Errors at recipe bias
  - Jaccard between predictions
  - Test-side argmax differences

Decision: if Δ ≥ +0.0003 AND per-class guardrail PASSES, consider LB probe.
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

    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
    pred_v1 = (np.log(np.clip(p_v1_o, 1e-12, 1)) + BIAS).argmax(1)
    m_v1 = balanced_accuracy_score(y, pred_v1)
    rec_v1 = recall_score(y, pred_v1, average=None)
    err_v1 = int((pred_v1 != y).sum())

    print(f"v1 PRIMARY  OOF={m_v1:.5f}  errs={err_v1}  rec={rec_v1.round(5)}")

    # Compare at α=0.30 (LB-validated arch), 0.35, 0.40
    for alpha in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        p_cw_o = log_blend([s3_o, classw_iso_o], np.array([1 - alpha, alpha]))
        pred_cw = (np.log(np.clip(p_cw_o, 1e-12, 1)) + BIAS).argmax(1)
        m_cw = balanced_accuracy_score(y, pred_cw)
        rec_cw = recall_score(y, pred_cw, average=None)
        err_cw = int((pred_cw != y).sum())
        d = m_cw - m_v1
        drec = (rec_cw - rec_v1).round(6)
        guard = bool((drec >= -5e-4).all())
        emit = guard and d >= 3e-4
        marker = "  ← EMIT" if emit else ""
        print(f"classw  α={alpha}  OOF={m_cw:.5f}  Δ={d:+.5f}  errs={err_cw}  "
              f"rec={rec_cw.round(5)}  drec={drec.tolist()}  "
              f"{'PASS' if guard else 'FAIL'}{marker}")

    print()
    # Build best LB-validated-arch submission (α=0.30 with classw)
    p_cw_t_30 = log_blend([s3_t, classw_iso_t], np.array([0.70, 0.30]))
    pred_cw_t_30 = (np.log(np.clip(p_cw_t_30, 1e-12, 1)) + BIAS).argmax(1)
    pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
    n_diff = int((pred_cw_t_30 != pred_v1_t).sum())
    print(f"@ α=0.30 (LB-validated arch): test diff vs v1 PRIMARY = {n_diff}")

    test_df = pd.read_csv("data/test.csv")
    sub = pd.DataFrame({"id": test_df["id"].values,
                         "Irrigation_Need": [LABELS[i] for i in pred_cw_t_30]})
    fname = "submission_classw_a030_lb_validated_arch.csv"
    sub.to_csv(SUB / fname, index=False)
    print(f"Saved {fname}")

    # Build α=0.35 too as a less risky alternative
    p_cw_t_35 = log_blend([s3_t, classw_iso_t], np.array([0.65, 0.35]))
    pred_cw_t_35 = (np.log(np.clip(p_cw_t_35, 1e-12, 1)) + BIAS).argmax(1)
    n_diff_35 = int((pred_cw_t_35 != pred_v1_t).sum())
    sub35 = pd.DataFrame({"id": test_df["id"].values,
                           "Irrigation_Need": [LABELS[i] for i in pred_cw_t_35]})
    fname35 = "submission_classw_a035.csv"
    sub35.to_csv(SUB / fname35, index=False)
    print(f"  α=0.35 test diff vs v1 PRIMARY = {n_diff_35}; saved {fname35}")


if __name__ == "__main__":
    main()
