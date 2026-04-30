"""T6 — directional compose: 4b base + T6's H->M direction only.

Rationale:
  TRAIN OOF direction-precision:
    H->M: T6 right 251/263 (95.4%) — well above 92% break-even
    M->H: T6 right 53/591 (9.0%) — close to break-even, historically
          DOES NOT transfer (W3_MHonly LB 0.98127 NULL)
    L<->M: tossup (55-58% precision)
  Conclusion: trust T6 only on H->M direction.

Mechanism:
  base = 4b (LB 0.98150)
  flip[i] = T6_argmax[i] iff (T6 says M and 4b says H) else 4b_argmax[i]
  This adds T6's H->M flips on top of 4b without inheriting M->H risk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T6_diversity_helpers import load_y_train, macro_recall, normed  # noqa: E402
from T6_emit_candidate import PATH, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== T6 directional compose ===\n")

    # Build T6 blend on TEST
    test_arrays = []
    for name, alpha in PATH:
        a = normed(np.load(ART / f"test_{name}.npy").astype(np.float32))
        test_arrays.append((a, alpha))
    test_blend = log_blend(test_arrays)
    v1_bias = np.array([-1.333, -1.0, 1.5])
    t6_argmax = (np.log(np.clip(test_blend, 1e-9, None)) + v1_bias).argmax(1).astype(np.int8)

    # Load 4b
    fb = csv_argmax("submission_idea4b_selective_override")

    # T6 says M and 4b says H => candidate flip
    cand = (t6_argmax == 1) & (fb == 2)
    print(f"H->M candidates from T6: {int(cand.sum())}")

    # Build candidate
    new_pred = fb.copy()
    new_pred[cand] = 1

    # Direction sanity
    b = csv_argmax("submission_2other_raw_tier1b_k2")
    diff_vs_4b = int((new_pred != fb).sum())
    diff_vs_b = int((new_pred != b).sum())
    print(f"diff vs 4b: {diff_vs_4b}")
    print(f"diff vs B:  {diff_vs_b}")

    dirs_vs_b = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            m = (b == fr) & (new_pred == to)
            if m.sum():
                dirs_vs_b[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(m.sum())
    print(f"directions vs B: {dirs_vs_b}")

    h_added = int(((b != 2) & (new_pred == 2)).sum())
    h_removed = int(((b == 2) & (new_pred != 2)).sum())
    print(f"net_H vs B: +{h_added} -{h_removed} = {h_added - h_removed:+d}")

    # ---- TRAIN OOF projected lift ----
    y = load_y_train()
    # Build TRAIN T6 OOF and 4b OOF analog
    oof_arrays = []
    for name, alpha in PATH:
        a = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        oof_arrays.append((a, alpha))
    t6_oof = log_blend(oof_arrays)
    t6_oof_argmax = (np.log(np.clip(t6_oof, 1e-9, None)) + v1_bias).argmax(1).astype(np.int8)

    # 4b OOF analog (B with bank-maj override; on TRAIN B == 4b empirically)
    from scipy.stats import mode
    from T6_diversity_helpers import tune_log_bias_simple
    from T2_conformal_helpers import load_bank
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    bv1, _ = tune_log_bias_simple(v1_oof, y)
    bra, _ = tune_log_bias_simple(raw_oof, y)
    bt1, _ = tune_log_bias_simple(t1_oof, y)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1).astype(np.int8)
    una = (a_ra == a_t1)
    fb_oof = a_v1.copy()
    om = una & (a_v1 != a_ra)
    fb_oof[om] = a_ra[om]

    # Apply same H->M-only directional compose on TRAIN OOF
    oof_cand = (t6_oof_argmax == 1) & (fb_oof == 2)
    fb_oof_new = fb_oof.copy()
    fb_oof_new[oof_cand] = 1

    n_oof_cand = int(oof_cand.sum())
    print(f"\nTRAIN OOF H->M candidate count: {n_oof_cand}")
    if n_oof_cand > 0:
        precision_M = float((y[oof_cand] == 1).mean())
        precision_H = float((y[oof_cand] == 2).mean())
        precision_L = float((y[oof_cand] == 0).mean())
        print(f"  P(true=M): {precision_M:.4f}")
        print(f"  P(true=H): {precision_H:.4f}")
        print(f"  P(true=L): {precision_L:.4f}")
        print(f"  break-even: 0.92")
        verdict = "PASS" if precision_M >= 0.92 else "FAIL"
        print(f"  verdict: {verdict}")

    base_macro = macro_recall(y, fb_oof)
    new_macro = macro_recall(y, fb_oof_new)
    print(f"\nTRAIN OOF macro:")
    print(f"  4b:                {base_macro:.6f}")
    print(f"  4b + T6 H->M only: {new_macro:.6f}")
    print(f"  delta:             {new_macro - base_macro:+.6f}")

    out_csv = SUB / "submission_T6_directional_4b_plus_t6_hm.csv"
    test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()
    sub = pd.DataFrame({
        "id": test_ids,
        "Irrigation_Need": pd.Series(new_pred).map({0: "Low", 1: "Medium", 2: "High"}),
    })
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    # Save results
    out = ART / "T6_directional_compose_results.json"
    out.write_text(json.dumps({
        "n_test_flips": int(cand.sum()),
        "n_oof_cand": n_oof_cand,
        "oof_precision_m": float((y[oof_cand] == 1).mean()) if n_oof_cand else None,
        "oof_precision_h": float((y[oof_cand] == 2).mean()) if n_oof_cand else None,
        "oof_macro_4b": float(base_macro),
        "oof_macro_new": float(new_macro),
        "oof_delta": float(new_macro - base_macro),
        "candidate_csv": str(out_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
