"""Step 3/3: blend the TE-regression OOF/test into the LB-best stacks.

Variant switch via env var TE_VARIANT in {orig (default), oof}:
  orig -> reads oof_xgb_te_reg.npy        / test_xgb_te_reg.npy
  oof  -> reads oof_xgb_te_reg_oof.npy    / test_xgb_te_reg_oof.npy
Output filenames suffix accordingly.
with FIXED greedy bias (no per-alpha bias retune to avoid the
binhigh-style overfit).

Two baselines are tested:
  (1) greedy alone (OOF 0.97375).
  (2) greedy + xgb_nonrule @ alpha_nr=0.15 in log space (LB-best,
      OOF 0.97421).

For each baseline we sweep log-blend weight alpha on the new
TE-regression component:
  blend = alpha * log(P_te_reg) + (1 - alpha) * log(P_baseline)
then renormalise + add the FIXED greedy bias and argmax.

Decision rule:
  - if best_alpha == 0: log "no signal", no submission.
  - elif best_delta vs LB-best < 0.0005: write a borderline submission
    file but mark "do not submit" in the JSON (matches non-rule
    discipline established 2026-04-21).
  - else: write a final submission and surface for LB probe.

Outputs:
  scripts/artifacts/blend_te_reg_results.json
  submissions/submission_blend_te_reg_{vs_greedy,vs_lbbest}.csv
    (only when sweep peaks above the relevant threshold)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


SEED = 42
VARIANT = os.environ.get("TE_VARIANT", "orig").lower()
if VARIANT not in ("orig", "oof", "subcell"):
    raise SystemExit(
        f"TE_VARIANT must be 'orig', 'oof', or 'subcell', got {VARIANT!r}"
    )
SUFFIX = {"orig": "", "oof": "_oof", "subcell": "_subcell"}[VARIANT]
OOF_NEW_PATH = f"oof_xgb_te_reg{SUFFIX}.npy"
TEST_NEW_PATH = f"test_xgb_te_reg{SUFFIX}.npy"
RESULTS_PATH = f"blend_te_reg_results{SUFFIX}.json"
SUB_VS_GREEDY = f"submission_blend_te_reg{SUFFIX}_vs_greedy.csv"
SUB_VS_LBBEST = f"submission_blend_te_reg{SUFFIX}_vs_lbbest.csv"
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def sweep(p_new: np.ndarray, p_base: np.ndarray, y: np.ndarray,
          bias: np.ndarray, base_oof: float, label: str) -> dict:
    grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40]
    rows = []
    for a in grid:
        b = log_blend2(p_new, p_base, a)
        lp = np.log(np.clip(b, 1e-9, 1.0)) + bias
        ba = balanced_accuracy_score(y, lp.argmax(axis=1))
        d = ba - base_oof
        rows.append({"alpha": a, "oof": float(ba), "delta_vs_base": float(d)})
        log(f"  [{label:11s}] alpha_new={a:.3f}  OOF={ba:.5f}  Δ={d:+.5f}")
    return {
        "label": label,
        "base_oof": float(base_oof),
        "sweep": rows,
        "best": max(rows, key=lambda r: r["oof"]),
    }


def main() -> None:
    t0 = time.time()
    log(f"variant={VARIANT}  reading {OOF_NEW_PATH} / {TEST_NEW_PATH}")
    log("loading components")
    oof_new = np.load(ART / OOF_NEW_PATH)
    test_new = np.load(ART / TEST_NEW_PATH)
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nr = np.load(ART / "oof_xgb_nonrule.npy")
    test_nr = np.load(ART / "test_xgb_nonrule.npy")
    log(f"  oof_te_reg {oof_new.shape}  oof_greedy {oof_greedy.shape}  "
        f"oof_nonrule {oof_nr.shape}")

    nonrule_res = json.loads(Path(ART / "nonrule_results.json").read_text())
    bias_greedy = np.array(nonrule_res["greedy_bias"])
    greedy_oof = nonrule_res["greedy_tuned_oof"]
    log(f"  greedy bias = {bias_greedy.round(4).tolist()}   greedy OOF = {greedy_oof:.5f}")

    # Build the LB-best base = log_blend2(nonrule, greedy, 0.15).
    alpha_nr = 0.15
    oof_lbbest = log_blend2(oof_nr, oof_greedy, alpha_nr)
    test_lbbest = log_blend2(test_nr, test_greedy, alpha_nr)

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)

    lbbest_oof = balanced_accuracy_score(
        y, (np.log(np.clip(oof_lbbest, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
    )
    log(f"  LB-best base (greedy + nonrule@0.15) OOF (fixed greedy bias) "
        f"= {lbbest_oof:.5f}")

    # Standalone diagnostic for the new component.
    raw_arg = balanced_accuracy_score(y, oof_new.argmax(axis=1))
    log(f"standalone new component argmax bal_acc = {raw_arg:.5f}")

    log("sweep #1: blend new component into greedy alone")
    s1 = sweep(oof_new, oof_greedy, y, bias_greedy, greedy_oof, "vs_greedy")
    log("sweep #2: blend new component into LB-best (greedy + nonrule@0.15)")
    s2 = sweep(oof_new, oof_lbbest, y, bias_greedy, lbbest_oof, "vs_lbbest")

    out = {
        "alpha_nr_for_lbbest": alpha_nr,
        "greedy_oof": float(greedy_oof),
        "lbbest_oof": float(lbbest_oof),
        "standalone_argmax": float(raw_arg),
        "vs_greedy": s1,
        "vs_lbbest": s2,
    }

    # Submission decisions.
    out["submissions"] = []
    for s, base_p_test, fname, base_label, base_oof in [
        (s1, test_greedy, SUB_VS_GREEDY, "greedy", greedy_oof),
        (s2, test_lbbest, SUB_VS_LBBEST, "lbbest", lbbest_oof),
    ]:
        best = s["best"]
        d = best["delta_vs_base"]
        if best["alpha"] == 0.0 or d <= 0:
            log(f"[{base_label}]  no OOF lift -> no submission")
            out["submissions"].append({"base": base_label, "action": "no_submission"})
            continue
        b_test = log_blend2(test_new, base_p_test, best["alpha"])
        lp = np.log(np.clip(b_test, 1e-9, 1.0)) + bias_greedy
        preds = lp.argmax(axis=1)
        path = SUB / fname
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            path, index=False
        )
        # Also report OOF confusion at the picked alpha for the record.
        b_oof = log_blend2(oof_new, base_p_test if False else
                           (oof_greedy if base_label == "greedy" else oof_lbbest),
                           best["alpha"])
        cm = confusion_matrix(
            y, (np.log(np.clip(b_oof, 1e-9, 1.0)) + bias_greedy).argmax(axis=1)
        )
        log(f"[{base_label}]  best alpha={best['alpha']:.3f}  OOF={best['oof']:.5f}  "
            f"Δ={d:+.5f}  -> wrote {path}")
        log(f"  OOF confusion matrix:\n"
            f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")
        out["submissions"].append({
            "base": base_label,
            "submission_path": str(path),
            "alpha": best["alpha"],
            "oof": best["oof"],
            "delta_vs_base": d,
            "borderline_threshold_met": d >= 5e-4,
        })

    with open(ART / RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {ART}/{RESULTS_PATH}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
