"""T4 — Stability-axis override on 4b (LB 0.98150).

Two parallel candidates:
  V1 (tighten):  drop 4b flips where 14-bank-agreement < 0.93.
                  Goal: variance reduction at higher precision.
  V2 (expand):   find rows NOT in 4b's flip set where 14-bank-agreement is
                 very high (=1.0 or >=0.93) AND bank-majority differs from
                 anchor B (i.e., 4b chose to skip them on its other axes).
                 Restrict to H->M direction only.

Validation: for each variant, project precision using TRAIN OOF analog:
  - v1 RF natural argmax = H AND bank-majority = M AND agreement >= threshold
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import load_bank  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    print("=== T4 — Stability-axis override on 4b ===\n")
    fb = csv_argmax("submission_idea4b_selective_override")
    b = csv_argmax("submission_2other_raw_tier1b_k2")
    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")
    n = len(fb)

    fb_flip_mask = b != fb
    print(f"4b base: {int(fb_flip_mask.sum())} flips on {n} rows")
    print(f"agreement on 4b flips dist:")
    for thresh in [0.99, 0.93, 0.86, 0.79, 0.71, 0.5]:
        c = int((agr[fb_flip_mask] >= thresh).sum())
        print(f"  >= {thresh}: {c}")

    # ---- V1: tighten 4b by dropping low-agreement flips ----
    for v1_thresh in [1.0, 0.93]:
        keep_mask = agr >= v1_thresh
        # New pred: 4b's flips where stability ok; revert others to B
        new_pred_v1 = b.copy()
        flip_to_keep = fb_flip_mask & keep_mask
        new_pred_v1[flip_to_keep] = fb[flip_to_keep]
        n_kept = int(flip_to_keep.sum())
        n_dropped = int(fb_flip_mask.sum()) - n_kept
        print(f"\nV1 tighten (agr >= {v1_thresh}): keep {n_kept}, drop {n_dropped}")

        # Direction breakdown of kept
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                m = (b == fr) & (new_pred_v1 == to)
                if m.sum():
                    dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(m.sum())
        print(f"  directions kept: {dirs}")
        out_csv = SUB / f"submission_T4_v1_tighten_a{int(v1_thresh*100):03d}.csv"
        sub = pd.DataFrame({
            "id": pd.read_csv(DATA / "test.csv")["id"].to_numpy(),
            "Irrigation_Need": pd.Series(new_pred_v1).map({0: "Low", 1: "Medium", 2: "High"}),
        })
        sub.to_csv(out_csv, index=False)
        print(f"  emitted: {out_csv}")

    # ---- V2: expand 4b by finding high-stability bank-majority flips
    # NOT in 4b's set ----
    # Candidates: bank-majority differs from B AND 4b did NOT flip.
    not_4b = ~fb_flip_mask
    bank_diff = b != maj
    cand = not_4b & bank_diff

    # restrict to H->M direction
    hm_cand = cand & (b == 2) & (maj == 1)
    print(f"\n--- V2 expand candidates ---")
    print(f"NOT in 4b AND bank-majority H->M: {int(hm_cand.sum())}")

    # bucket by stability level
    for thresh in [1.0, 0.93, 0.86]:
        m = hm_cand & (agr >= thresh)
        print(f"  + agr >= {thresh}: {int(m.sum())}")

    # build V2 candidates at agr >= 1.0 (full unanimity on 14-bank)
    for v2_thresh in [1.0, 0.93]:
        v2_mask = hm_cand & (agr >= v2_thresh)
        new_pred_v2 = fb.copy()
        new_pred_v2[v2_mask] = maj[v2_mask]
        n_added = int(v2_mask.sum())
        print(f"\nV2 expand (agr >= {v2_thresh}): add {n_added} H->M flips on top of 4b")

        # Final direction summary vs B
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                mask = (b == fr) & (new_pred_v2 == to)
                if mask.sum():
                    dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(mask.sum())
        print(f"  directions vs B: {dirs}")
        h_added = int(((b != 2) & (new_pred_v2 == 2)).sum())
        h_removed = int(((b == 2) & (new_pred_v2 != 2)).sum())
        net_h = h_added - h_removed
        print(f"  net_H: +{h_added} -{h_removed} = {net_h:+d}")

        out_csv = SUB / f"submission_T4_v2_expand_a{int(v2_thresh*100):03d}.csv"
        sub = pd.DataFrame({
            "id": pd.read_csv(DATA / "test.csv")["id"].to_numpy(),
            "Irrigation_Need": pd.Series(new_pred_v2).map({0: "Low", 1: "Medium", 2: "High"}),
        })
        sub.to_csv(out_csv, index=False)
        print(f"  emitted: {out_csv}")

    # ---- TRAIN OOF precision validation for V2 expand mechanism ----
    print("\n--- TRAIN OOF precision validation for V2 mechanism ---")
    oof_bank = load_bank("oof")
    oof_argmax = oof_bank.argmax(axis=2)  # (14, 630000)
    oof_majority = mode(oof_argmax, axis=0, keepdims=False).mode
    # 14-bank agreement on TRAIN
    oof_agree = (oof_argmax == oof_majority).mean(axis=0)

    y = pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)

    # Use v1 RF natural as 4b proxy
    v1 = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    v1_argmax = v1.argmax(1).astype(np.int8)

    # V2 analog: v1=H, bank-majority=M (i.e., v1's H argmax differs from bank)
    v2_filter = (v1_argmax == 2) & (oof_majority == 1)
    print(f"v1=H AND bank_maj=M: {int(v2_filter.sum())}")

    for thresh in [1.0, 0.93, 0.86, 0.79]:
        f = v2_filter & (oof_agree >= thresh)
        c = int(f.sum())
        if c == 0:
            print(f"  agr >= {thresh}: 0 rows")
            continue
        p_m = float((y[f] == 1).mean())
        p_h = float((y[f] == 2).mean())
        be = "PASS" if p_m >= 0.92 else "fail"
        print(f"  agr >= {thresh}: n={c}, P(true=M)={p_m:.4f}, "
              f"P(true=H)={p_h:.4f} [{be} for H->M]")

    out = ART / "T4_stability_axis_results.json"
    out.write_text(json.dumps({
        "v1_thresholds": [1.0, 0.93],
        "v2_thresholds": [1.0, 0.93],
    }, indent=2))


if __name__ == "__main__":
    main()
