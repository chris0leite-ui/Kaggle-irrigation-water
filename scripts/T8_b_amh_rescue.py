"""T8 — B + AMH-gated M->H rescue.

Structurally novel direction: every prior override mechanism on this comp
flipped H->M (4b: 105 H->M flips; T6: 45 H->M; W5/T7 also H-direction).
This explores the OPPOSITE direction: rescue true-H rows that B (LB 0.98140)
classified as M, identified by aux_missed_high (AUC 0.98).

Mechanism:
  candidate[i] = H   if (B[i] = M) AND (test_amh[i] > tau)
                 else B[i]

For each tau, compute on TRAIN OOF:
  - n_rescued: rows where B_oof says M and amh > tau
  - precision: P(true=H | rescued)
  - per-class recall delta vs B
  - macro_recall delta vs B

Emit only candidates with >100 row changes (measurable on 80/20 public).

Why this is structurally novel:
  - Direction: M->H (rescue), not H->M (override)
  - Gate: AMH-as-positive-signal (high amh => predict H), not as exclusion
  - Base: B (LB 0.98140), not 4b — independent of the triple-consensus
    override family
  - Never tried: amh has been used as decorrelating signal in metas, but
    never as a direct predict-H trigger on top of an LB-validated CSV
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
    print("=== T8: B + AMH-gated M->H rescue ===\n")

    # --- Test side ---
    b_test = csv_argmax("submission_2other_raw_tier1b_k2")
    test_amh = np.load(ART / "test_aux_missed_high.npy").astype(np.float32)
    test_ids = pd.read_csv(DATA / "test.csv")["id"].to_numpy()

    print(f"B test argmax: L={(b_test == 0).sum()} M={(b_test == 1).sum()} H={(b_test == 2).sum()}")
    print(f"test_amh on B-says-M rows ({(b_test == 1).sum()}): "
          f"mean={test_amh[b_test == 1].mean():.3f} "
          f"median={np.median(test_amh[b_test == 1]):.3f} "
          f"q95={np.quantile(test_amh[b_test == 1], 0.95):.3f}")

    # --- TRAIN OOF analog of B ---
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
    b_oof = a_v1.copy()
    b_oof[una & (a_v1 != a_ra)] = a_ra[una & (a_v1 != a_ra)]

    base_macro = macro_recall(y, b_oof)
    print(f"\nB OOF analog macro: {base_macro:.6f}")
    print(f"B OOF argmax: L={(b_oof == 0).sum()} M={(b_oof == 1).sum()} H={(b_oof == 2).sum()}")

    oof_amh = np.load(ART / "oof_aux_missed_high.npy").astype(np.float32)
    print(f"oof_amh on B-says-M rows: mean={oof_amh[b_oof == 1].mean():.3f} "
          f"q95={np.quantile(oof_amh[b_oof == 1], 0.95):.3f} "
          f"q99={np.quantile(oof_amh[b_oof == 1], 0.99):.3f}")

    # --- Sweep tau ---
    rows = []
    for tau in [0.99, 0.98, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80]:
        # OOF rescue mask: B says M AND amh > tau -> predict H
        rescue_oof = (b_oof == 1) & (oof_amh > tau)
        n_rescue_oof = int(rescue_oof.sum())
        new_b_oof = b_oof.copy()
        new_b_oof[rescue_oof] = 2

        # Per-class precision of the rescued subset
        if n_rescue_oof > 0:
            true_H = (y[rescue_oof] == 2).mean()
            true_M = (y[rescue_oof] == 1).mean()
            true_L = (y[rescue_oof] == 0).mean()
        else:
            true_H = true_M = true_L = float("nan")

        m_new = macro_recall(y, new_b_oof)

        # Test side
        rescue_test = (b_test == 1) & (test_amh > tau)
        n_rescue_test = int(rescue_test.sum())

        # Per-class recall change
        n_true_H = int((y == 2).sum())
        n_true_M = int((y == 1).sum())
        delta_H_recall = (rescue_oof & (y == 2)).sum() / n_true_H
        delta_M_recall = -(rescue_oof & (y == 1)).sum() / n_true_M

        rows.append({
            "tau": tau,
            "n_rescue_oof": n_rescue_oof,
            "p_true_H": float(true_H) if n_rescue_oof else None,
            "p_true_M": float(true_M) if n_rescue_oof else None,
            "p_true_L": float(true_L) if n_rescue_oof else None,
            "delta_H_recall": float(delta_H_recall),
            "delta_M_recall": float(delta_M_recall),
            "oof_macro": float(m_new),
            "oof_macro_delta": float(m_new - base_macro),
            "n_rescue_test": n_rescue_test,
        })

    df = pd.DataFrame(rows)
    print("\n=== Sweep results ===")
    print(df.to_string(index=False))

    OUT = ART / "T8"
    OUT.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT / "sweep.csv", index=False)
    print(f"\nsaved: {OUT / 'sweep.csv'}")

    # Emit candidates: only thresholds with measurable test changes (>30 rows)
    # AND non-negative TRAIN OOF macro projection
    print("\n=== Emitting candidates ===")
    for r in rows:
        tau = r["tau"]
        if r["n_rescue_test"] < 30:
            print(f"  tau={tau:.2f}: only {r['n_rescue_test']} test changes — skip (below noise floor)")
            continue
        if r["oof_macro_delta"] < 0:
            print(f"  tau={tau:.2f}: OOF delta {r['oof_macro_delta']:+.6f} negative — skip")
            continue
        # Build candidate
        rescue_test = (b_test == 1) & (test_amh > tau)
        new_pred = b_test.copy()
        new_pred[rescue_test] = 2

        out_csv = SUB / f"submission_T8_B_amh_rescue_tau{int(tau*100):02d}.csv"
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(CLASS_STR),
        })
        sub.to_csv(out_csv, index=False)
        print(f"  tau={tau:.2f}: rescued {r['n_rescue_test']} rows — emitted {out_csv.name} "
              f"(L={(new_pred == 0).sum()} M={(new_pred == 1).sum()} H={(new_pred == 2).sum()})  "
              f"OOF Δ={r['oof_macro_delta']:+.6f}")

    summary = {
        "base_oof_macro": float(base_macro),
        "sweep": rows,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
