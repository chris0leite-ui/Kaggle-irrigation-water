"""Blend-gate analyzer for Angles A, B, C vs LB-best primary 0.98094.

For each candidate, reports:
  standalone_tuned      tuned bal_acc at the candidate's own bias
  fixed_bias_oof        bal_acc at LB-best's recipe bias [1.43, 1.47, 3.40]
  errs_at_anchor        count vs LB-best 4-stack
  jaccard_vs_anchor     error-set Jaccard vs LB-best 4-stack
  per_class_recall      L / M / H deltas vs LB-best 4-stack
  best_blend_alpha      α ∈ [0, 0.5] sweep at fixed bias
  best_blend_delta      OOF Δ vs LB-best 4-stack

Emits submission only if Δ ≥ +5e-4 AND per-class recall guardrail PASS
(every class within −5e-4 of LB-best 4-stack).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, log, normed,
)

CLASSES = ["Low", "Medium", "High"]


def per_class_recall(p: np.ndarray, y: np.ndarray, bias: np.ndarray) -> np.ndarray:
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    out = np.zeros(3, dtype=np.float64)
    for k in range(3):
        m = y == k
        out[k] = (pred[m] == k).mean() if m.any() else 0.0
    return out


def make_anchor(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LB-best 4-stack: 3-stack ⊗ xgb_metastack_iso α=0.30."""
    s_o, s_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    o = log_blend([s_o, meta_o], np.array([0.7, 0.3]))
    t = log_blend([s_t, meta_t], np.array([0.7, 0.3]))
    return o.astype(np.float32), t.astype(np.float32)


def gate(name: str, oof_path: Path, test_path: Path, y: np.ndarray,
         anchor_o: np.ndarray, anchor_t: np.ndarray, anchor_bal: float,
         anchor_errs: int, anchor_pcr: np.ndarray, test_ids: np.ndarray
         ) -> dict:
    if not oof_path.exists():
        return {"name": name, "status": "MISSING"}
    oof = normed(np.load(oof_path).astype(np.float32))
    test_p = normed(np.load(test_path).astype(np.float32))
    pred_a = (np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1)
    pred_anchor = (np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS).argmax(1)
    errs_a = int((pred_a != y).sum())
    pcr_a = per_class_recall(oof, y, BIAS)
    err_set_a = pred_a != y
    err_set_anc = pred_anchor != y
    inter = (err_set_a & err_set_anc).sum()
    union = (err_set_a | err_set_anc).sum()
    jac = float(inter / union) if union else 1.0

    sweep = {}
    best_alpha, best_bal = 0.0, anchor_bal
    pcr_best = anchor_pcr.copy()
    errs_best = anchor_errs
    for a in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        b = log_blend([anchor_o, oof], np.array([1 - a, a]))
        from sklearn.metrics import balanced_accuracy_score
        bp = (np.log(np.clip(b, 1e-12, 1)) + BIAS).argmax(1)
        bal = float(balanced_accuracy_score(y, bp))
        sweep[f"a_{a:.3f}"] = bal
        if bal > best_bal:
            best_alpha, best_bal = a, bal
            pcr_best = per_class_recall(b, y, BIAS)
            errs_best = int((bp != y).sum())

    pcr_delta = pcr_best - anchor_pcr
    guardrail = bool((pcr_delta >= -5e-4).all())
    delta = best_bal - anchor_bal
    emit = (delta >= 5e-4) and guardrail
    stand_bal = float(((np.log(np.clip(oof, 1e-12, 1)) + BIAS).argmax(1) == y).mean())
    log(f"  {name:<24} stand_acc={stand_bal:.5f} errs={errs_a} jac={jac:.3f} "
        f"bestα={best_alpha:.3f} Δ={delta:+.5f} "
        f"pcr_dL={pcr_delta[0]:+.4f} dM={pcr_delta[1]:+.4f} dH={pcr_delta[2]:+.4f} "
        f"emit={'Y' if emit else 'N'}")

    if emit:
        b_test = log_blend([anchor_t, test_p], np.array([1 - best_alpha, best_alpha]))
        bp_test = (np.log(np.clip(b_test, 1e-12, 1)) + BIAS).argmax(1)
        sub = pd.DataFrame({"id": test_ids,
                            "Irrigation_Need": [CLASSES[i] for i in bp_test]})
        sub_path = SUB / f"submission_{name}_a{int(round(best_alpha*1000)):03d}.csv"
        sub.to_csv(sub_path, index=False)
        log(f"    EMIT {sub_path.name}")

    return dict(
        name=name, status="OK", standalone_errs=errs_a,
        standalone_pcr_LMH=[float(x) for x in pcr_a],
        jaccard_vs_anchor=jac,
        sweep=sweep,
        best_alpha=float(best_alpha),
        best_bal=float(best_bal),
        delta_vs_anchor=float(delta),
        errs_at_best=errs_best,
        pcr_delta_LMH=[float(x) for x in pcr_delta],
        guardrail_pass=guardrail, emit=emit,
    )


def main():
    y = load_y()
    test_ids = pd.read_csv("data/test.csv")["id"].values
    log("building LB-best 4-stack anchor")
    anc_o, anc_t = make_anchor(y)
    from sklearn.metrics import balanced_accuracy_score
    anc_pred = (np.log(np.clip(anc_o, 1e-12, 1)) + BIAS).argmax(1)
    anc_bal = float(balanced_accuracy_score(y, anc_pred))
    anc_errs = int((anc_pred != y).sum())
    anc_pcr = per_class_recall(anc_o, y, BIAS)
    log(f"  anchor OOF={anc_bal:.5f}  errs={anc_errs}  pcr={anc_pcr.round(4).tolist()}")

    candidates = {
        "angle_a_residual": ("oof_angle_a_residual.npy", "test_angle_a_residual.npy"),
        "angle_b_recipe_dae": ("oof_recipe_full_te_dae.npy", "test_recipe_full_te_dae.npy"),
        "angle_c_mixup": ("oof_angle_c_mixup.npy", "test_angle_c_mixup.npy"),
    }
    out = dict(
        anchor_oof=anc_bal, anchor_errs=anc_errs,
        anchor_pcr_LMH=[float(x) for x in anc_pcr],
        candidates={},
    )
    for name, (op, tp) in candidates.items():
        out["candidates"][name] = gate(
            name, ART / op, ART / tp, y, anc_o, anc_t,
            anc_bal, anc_errs, anc_pcr, test_ids,
        )
    res_path = ART / "angles_blend_gate_results.json"
    with open(res_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {res_path}")


if __name__ == "__main__":
    main()
