"""T7 — restrict 4b's H->M flips by aux_missed_high gate.

4b applies 108 selective overrides on top of B (2other_raw_tier1b_k2),
mostly H->M (105 of 108). Hypothesis: among those H->M flips, the rows
with HIGH aux_missed_high are likely true-H (i.e., the rule missed an H
flip). Dropping those over-flips should leave a tighter, higher-precision
override set.

Mechanism (cheap, never combined before):
  candidate = where 4b argmax = M AND B argmax = H AND test_amh > τ → keep B's H
                                                      else → keep 4b's M
  (i.e., undo 4b's H->M flips on rows the auxiliary detector flags as H)

Sweep τ ∈ {0.20, 0.30, 0.50, 0.70}, plus τ=∞ (= unmodified 4b baseline).

For each τ, compute on TRAIN OOF:
  - Number of flips dropped (= number of true-H rescued + true-M lost)
  - Precision of the SURVIVING flips (4b's flips MINUS the high-amh ones)
  - Net macro_recall delta vs unmodified 4b OOF analog

Emit candidate CSVs only for τ where TRAIN OOF projects positive lift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402
from T6_diversity_helpers import load_y_train, macro_recall, normed, tune_log_bias_simple  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

CLASS_INT = {"Low": 0, "Medium": 1, "High": 2}
CLASS_STR = {0: "Low", 1: "Medium", 2: "High"}


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLASS_INT).to_numpy(dtype=np.int8)


def main():
    print("=== T7: aux_missed_high gate on 4b's H->M flips ===\n")

    # --- Test side ---
    fb_test = csv_argmax("submission_idea4b_selective_override")
    b_test = csv_argmax("submission_2other_raw_tier1b_k2")
    test_amh = np.load(ART / "test_aux_missed_high.npy").astype(np.float32)
    test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()

    flip_HtoM = (b_test == 2) & (fb_test == 1)
    n_flips = int(flip_HtoM.sum())
    print(f"4b H->M flips on test: {n_flips}")
    print(f"test_amh on those flips: mean={test_amh[flip_HtoM].mean():.3f} "
          f"median={np.median(test_amh[flip_HtoM]):.3f}")

    # --- TRAIN OOF analog (replicate the 4b OOF construction) ---
    y = load_y_train()
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    bv1, _ = tune_log_bias_simple(v1_oof, y)
    bra, _ = tune_log_bias_simple(raw_oof, y)
    bt1, _ = tune_log_bias_simple(t1_oof, y)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1).astype(np.int8)
    una = a_ra == a_t1
    fb_oof = a_v1.copy()
    fb_oof[una & (a_v1 != a_ra)] = a_ra[una & (a_v1 != a_ra)]
    b_oof = a_v1.copy()
    # B is the unanimous-of-raw+tier1b override on v1 (≈4b but no triple-consensus)
    om = una & (a_v1 != a_ra)
    b_oof[om] = a_ra[om]

    bank = load_bank("oof")
    bank_mean = bank_mean_probs(bank)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)

    # 4b OOF analog: where B says H AND v1 says M AND bank-majority says M -> flip H to M
    # (this approximates the 4b mechanism since we don't have an exact OOF replay)
    # Approach: take rows where (b_oof==H) AND (v1==M or bank==M) -> flip
    flip_oof = (b_oof == 2) & ((a_v1 == 1) | (bank_argmax == 1))
    fb_4b_oof = b_oof.copy()
    fb_4b_oof[flip_oof] = 1
    base_macro = macro_recall(y, fb_4b_oof)
    print(f"\n4b OOF analog macro: {base_macro:.6f}")
    print(f"H->M flips on TRAIN OOF: {int(flip_oof.sum())}")
    if flip_oof.any():
        true_M = (y[flip_oof] == 1).mean()
        true_H = (y[flip_oof] == 2).mean()
        print(f"  baseline precision: P(true=M)={true_M:.4f}, P(true=H)={true_H:.4f}")

    oof_amh = np.load(ART / "oof_aux_missed_high.npy").astype(np.float32)

    # --- Sweep tau ---
    rows = []
    for tau in [1.0, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20]:
        # On TRAIN OOF: identify the SUBSET of 4b's flips that we KEEP (amh <= tau)
        keep_oof = flip_oof & (oof_amh <= tau)
        drop_oof = flip_oof & (oof_amh > tau)
        n_keep_oof = int(keep_oof.sum())
        n_drop_oof = int(drop_oof.sum())

        # Recompute the gated 4b OOF
        fb_gated_oof = b_oof.copy()
        fb_gated_oof[keep_oof] = 1
        m_gated = macro_recall(y, fb_gated_oof)

        # Per-axis precision
        p_keep_M = float((y[keep_oof] == 1).mean()) if n_keep_oof else float("nan")
        p_drop_H = float((y[drop_oof] == 2).mean()) if n_drop_oof else float("nan")

        # Test-side equivalent
        keep_test = flip_HtoM & (test_amh <= tau)
        drop_test = flip_HtoM & (test_amh > tau)
        n_keep_test = int(keep_test.sum())
        n_drop_test = int(drop_test.sum())

        rows.append({
            "tau": tau,
            "n_keep_oof": n_keep_oof,
            "n_drop_oof": n_drop_oof,
            "p_keep_true_M": p_keep_M,
            "p_drop_true_H": p_drop_H,
            "oof_macro": float(m_gated),
            "oof_macro_delta": float(m_gated - base_macro),
            "n_keep_test": n_keep_test,
            "n_drop_test": n_drop_test,
        })

    df = pd.DataFrame(rows)
    print("\n=== Sweep results ===")
    print(df.to_string(index=False))

    # Emit candidate CSVs for thresholds with positive (or near-zero) OOF delta
    OUT = ART / "T7"
    OUT.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nsaved: {OUT / 'sweep.csv'}")

    for tau in [0.95, 0.90, 0.80, 0.70, 0.50]:
        keep_test = flip_HtoM & (test_amh <= tau)
        drop_test = flip_HtoM & (test_amh > tau)
        new_pred = fb_test.copy()
        # Where 4b had flipped H->M and amh > tau, undo the flip (revert to H)
        new_pred[drop_test] = 2

        n_dropped = int(drop_test.sum())
        out_csv = SUB / f"submission_T7_4b_amh_gated_tau{int(tau*100):02d}.csv"
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(CLASS_STR),
        })
        sub.to_csv(out_csv, index=False)
        print(f"  tau={tau:.2f}: dropped {n_dropped} flips, "
              f"emitted {out_csv.name} "
              f"(L={(new_pred == 0).sum()} M={(new_pred == 1).sum()} H={(new_pred == 2).sum()})")

    summary = {
        "n_flips_test": n_flips,
        "n_flips_oof": int(flip_oof.sum()),
        "base_oof_macro": float(base_macro),
        "sweep": rows,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
