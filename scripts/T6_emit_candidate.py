"""T6 — emit test-side blend per the diversity-penalized path.

Path selected (beta=0.005 from T6_diversity_greedy_results.json):
  v1 RF natural (anchor)
  + xgb_nonrule alpha=0.15      (jac 0.652 — most orthogonal)
  + xgb_metastack alpha=0.30
  + recipe_pseudolabel_seed7labeler alpha=0.05
  + sklearn_rf_meta_natural_r10_with_tier1b alpha=0.05
  + tier1b_greedy_meta alpha=0.30
  + realmlp alpha=0.10

Final OOF tuned macro = 0.981105

Emit:
  C1: T6 standalone with v1's tuned bias [-1.333, -1.0, 1.5]
  C2: T6 standalone OOF-tuned bias
  C3: 4b override mechanism applied with T6 as the BASE instead of v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T6_diversity_helpers import (  # noqa: E402
    load_y_train,
    macro_recall,
    normed,
    tune_log_bias_simple,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

PATH = [
    ("sklearn_rf_meta_natural", 1.0),
    ("xgb_nonrule", 0.15),
    ("xgb_metastack", 0.30),
    ("recipe_pseudolabel_seed7labeler", 0.05),
    ("sklearn_rf_meta_natural_r10_with_tier1b", 0.05),
    ("tier1b_greedy_meta", 0.30),
    ("realmlp", 0.10),
]


def log_blend(arrays_with_alphas: list[tuple[np.ndarray, float]]) -> np.ndarray:
    log_p = None
    base_alpha = arrays_with_alphas[0][1]
    log_p = base_alpha * np.log(np.clip(arrays_with_alphas[0][0], 1e-9, None))
    for arr, alpha in arrays_with_alphas[1:]:
        # Sequential log-blend: new = (1-alpha)*current + alpha*new
        new_log = np.log(np.clip(arr, 1e-9, None))
        # current log-prob
        log_curr = log_p
        log_p = (1 - alpha) * log_curr + alpha * new_log
        # renormalize each row
        log_p = log_p - log_p.max(1, keepdims=True)
        # not strictly necessary, but keeps numerical scale stable
    p = np.exp(log_p - log_p.max(1, keepdims=True))
    return normed(p)


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def emit_csv(pred: np.ndarray, name: str) -> Path:
    test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    out = SUB / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    return out


def main():
    print("=== T6 emit test-side candidate ===\n")
    y = load_y_train()

    # Build OOF blend (replicate T6 path)
    oof_arrays = []
    for name, alpha in PATH:
        a = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        oof_arrays.append((a, alpha))
    oof_blend = log_blend(oof_arrays)

    # Build TEST blend
    test_arrays = []
    for name, alpha in PATH:
        a = normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        test_arrays.append((a, alpha))
    test_blend = log_blend(test_arrays)
    print(f"test_blend argmax dist: {np.bincount(test_blend.argmax(1), minlength=3).tolist()}")

    # OOF score with v1's tuned bias (no retune to avoid overfit)
    v1_bias = np.array([-1.333, -1.0, 1.5])
    pred_v1bias = (np.log(np.clip(oof_blend, 1e-9, None)) + v1_bias).argmax(1)
    print(f"OOF macro at v1 bias: {macro_recall(y, pred_v1bias):.6f}")

    # OOF score with retuned bias (selection-bias risk)
    bias_t6, score_t6 = tune_log_bias_simple(oof_blend, y)
    print(f"OOF macro at T6-retuned bias {bias_t6.round(3).tolist()}: {score_t6:.6f}")

    # Test predictions:
    # C1: T6 with v1 bias
    test_pred_c1 = (np.log(np.clip(test_blend, 1e-9, None)) + v1_bias).argmax(1).astype(np.int8)
    # C2: T6 with T6 retuned bias
    test_pred_c2 = (np.log(np.clip(test_blend, 1e-9, None)) + bias_t6).argmax(1).astype(np.int8)

    print(f"\nC1 (v1 bias) test argmax dist: {np.bincount(test_pred_c1, minlength=3).tolist()}")
    print(f"C2 (T6 bias) test argmax dist: {np.bincount(test_pred_c2, minlength=3).tolist()}")

    # Compare to 4b
    fb = csv_argmax("submission_idea4b_selective_override")
    diff_c1 = int((test_pred_c1 != fb).sum())
    diff_c2 = int((test_pred_c2 != fb).sum())
    print(f"\nDiff vs 4b: C1={diff_c1}, C2={diff_c2}")

    # Direction breakdown vs 4b for C1 (closest to 4b's bias)
    print(f"\nDirections C1 vs 4b:")
    dirs_c1 = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            m = (fb == fr) & (test_pred_c1 == to)
            if m.sum():
                dirs_c1[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(m.sum())
    print(f"  {dirs_c1}")
    h_added_c1 = int(((fb != 2) & (test_pred_c1 == 2)).sum())
    h_removed_c1 = int(((fb == 2) & (test_pred_c1 != 2)).sum())
    print(f"  net_H: +{h_added_c1} -{h_removed_c1} = {h_added_c1 - h_removed_c1:+d}")

    # ---- C3: apply 4b's override mechanism with T6 as base ----
    # 4b's mechanism: B + bagged_v1' + {raw, tier1b} unanimous + 14-bank maj
    # B = v1 + {raw, tier1b} k=2 unanimous (we don't have B's components in T6)
    # Simpler: apply 4b's specific 108 flips on top of T6.
    test_pred_c3 = test_pred_c1.copy()
    fb_b = csv_argmax("submission_2other_raw_tier1b_k2")
    # 4b's flips relative to B:
    fb_flip_mask = fb_b != fb
    test_pred_c3[fb_flip_mask] = fb[fb_flip_mask]

    diff_c3 = int((test_pred_c3 != fb).sum())
    h_added_c3 = int(((fb != 2) & (test_pred_c3 == 2)).sum())
    h_removed_c3 = int(((fb == 2) & (test_pred_c3 != 2)).sum())
    print(f"\nC3 (T6 + 4b's 108 flips overlaid) test argmax dist: "
          f"{np.bincount(test_pred_c3, minlength=3).tolist()}")
    print(f"  Diff vs 4b: {diff_c3}")
    print(f"  net_H vs 4b: +{h_added_c3} -{h_removed_c3} = {h_added_c3 - h_removed_c3:+d}")

    # Emit candidates
    emit_csv(test_pred_c1, "T6_C1_blend_v1bias")
    emit_csv(test_pred_c2, "T6_C2_blend_t6bias")
    emit_csv(test_pred_c3, "T6_C3_blend_with_4b_overlay")
    print("\nEmitted: T6_C1, T6_C2, T6_C3 submission CSVs")

    out = ART / "T6_emit_candidate_results.json"
    out.write_text(json.dumps({
        "oof_macro_v1bias": macro_recall(y, pred_v1bias),
        "oof_macro_t6bias": score_t6,
        "v1_bias": v1_bias.tolist(),
        "t6_bias": bias_t6.tolist(),
        "diff_vs_4b": {"C1": diff_c1, "C2": diff_c2, "C3": diff_c3},
        "net_h_vs_4b_C1": h_added_c1 - h_removed_c1,
        "net_h_vs_4b_C3": h_added_c3 - h_removed_c3,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
