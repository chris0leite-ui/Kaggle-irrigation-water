"""Fresh saturation check on ExcelFormer-based candidates from b33b795.

Apply the same precision-back-out diagnostic on each ExF candidate.
The 4 ExF candidates are LB-untested. Project precision via independent
signal (v1 + 14-bank), accounting for the lineage-correlation finding
(commit 742287f): 14-bank is partly correlated with 4b's input axes.

Per the new lineage finding:
  - H→M direction: bank works (95% precision validated by 4b's 0.98150)
  - M→H direction: bank's lineage WITH ExF is suspect (ExF lineage unknown
    — ExF is on V10 recipe FE, so lineage-correlated with raw + tier1b)

Decision criteria for each candidate:
  - Compute candidate's flip set (vs 4b)
  - Direction breakdown (M→H, H→M, etc.)
  - Independent-axis voting on flip rows
  - Projected precision per direction
  - LB projection
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def to_arg(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def directions(a, c, mask):
    d = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((a == fr) & (c == to) & mask).sum())
            if n > 0:
                d[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    return d


# Macro-recall break-even precision per override direction
# Using observed test class counts as proxies
N_L = 159718
N_M = 100261
N_H = 10279

BREAK_EVEN = {
    "M->H": N_M / (N_M + N_H),  # ≈ 0.907
    "H->M": N_H / (N_M + N_H),  # ≈ 0.0837 (very low — rare class easy to gain)
    "L->M": N_L / (N_L + N_M),  # ≈ 0.614
    "M->L": N_L / (N_L + N_M),  # but for M→L, gain class L, lose class M
    "L->H": N_L / (N_L + N_H),  # ≈ 0.939
    "H->L": N_H / (N_L + N_H),  # ≈ 0.060
}


def project_macro(direction: str, n: int, prec: float) -> float:
    """Macro-recall delta if `n` overrides at `prec` precision in this direction."""
    fr, to = direction.split("->")
    fr_idx = {"L": 0, "M": 1, "H": 2}[fr]
    to_idx = {"L": 0, "M": 1, "H": 2}[to]
    counts = {0: N_L, 1: N_M, 2: N_H}
    n_target = counts[to_idx]   # gain on this class
    n_source = counts[fr_idx]   # loss on this class
    corr = prec * n
    wrong = (1 - prec) * n
    # macro-recall = average of per-class recall
    # gain on target class: +corr / n_target
    # loss on source class: -wrong / n_source
    return (corr / n_target - wrong / n_source) / 3


def main():
    print("=== Fresh check on ExcelFormer-based candidates ===\n")

    # Load anchor
    fb = to_arg("submission_idea4b_selective_override")
    b = to_arg("submission_2other_raw_tier1b_k2")
    raw = to_arg("submission_rawashishsin_2600_standalone")
    tier1b = to_arg("submission_tier1b_greedy_meta")
    v1 = to_arg("submission_sklearn_rf_meta_natural_standalone_v1_lb98129")
    maj = np.load(ART / "stability_test_majority.npy")
    agr = np.load(ART / "stability_test_agreement.npy")
    p_bag = np.load(ART / "_test_bagged_v1_probs.npy")  # (270k, 3)
    biased = np.log(np.clip(p_bag, 1e-9, 1)) + np.array([0.43, 0.87, 3.20])
    bagged_arg = biased.argmax(axis=1).astype(np.int8)

    # Load ExF posterior if available
    exf_oof_p = ART / "test_excelformer.npy"
    if exf_oof_p.exists():
        exf_p = np.load(exf_oof_p)
        if exf_p.ndim == 2 and exf_p.shape[1] == 3:
            exf_arg = exf_p.argmax(axis=1).astype(np.int8)
        else:
            exf_arg = None
    else:
        exf_arg = None
    print(f"ExF arg available: {exf_arg is not None}\n")

    # Candidates to check
    cand_names = [
        "submission_4b_plus_safe3_exf_v1_MtoH",
        "submission_4b_plus_safe4_exf_v1_raw_MtoH",
        "submission_4b_plus_exf_v1_3axis_MtoH",
        "submission_4b_plus_exf_v1_raw_4axis_MtoH",
        "submission_4b_plus_w5_only",
        "submission_4b_plus_w5_3axis",
        "submission_4b_plus_w5_strict85",
        # strict90 already LB-tested at 0.98143
    ]

    results = {}
    for cn in cand_names:
        if not (SUB / f"{cn}.csv").exists():
            print(f"--- {cn}: MISSING ---\n")
            continue
        cand = to_arg(cn)
        flip_mask = cand != fb
        flip_idx = np.where(flip_mask)[0]
        n_flips = len(flip_idx)
        if n_flips == 0:
            print(f"--- {cn}: 0 flips vs 4b ---\n")
            results[cn] = {"n_flips": 0, "verdict": "STRUCTURALLY EMPTY"}
            continue

        dirs = directions(fb, cand, flip_mask)
        # also compute total directions vs B (the LB anchor reference)
        tot_dirs_vs_B = directions(b, cand, b != cand)
        h_added = int(((b != 2) & (cand == 2)).sum())
        h_removed = int(((b == 2) & (cand != 2)).sum())

        print(f"=== {cn} ===")
        print(f"  flips vs 4b: {n_flips}")
        print(f"  directions (4b → cand): {dirs}")
        print(f"  total flips vs B: {(b != cand).sum()}, dirs: {tot_dirs_vs_B}")
        print(f"  net_H vs B: +{h_added} -{h_removed} = {h_added-h_removed:+d}")

        # Independent-axis voting on flip rows
        # For each flip, check what each axis says about the FLIP direction
        flip_classes = cand[flip_mask]  # what cand says
        # NOTE: per lineage finding, 14-bank, raw, tier1b are correlated lineages
        # The TRULY independent axes per confound analysis are:
        #   - bagged_v1 / 14-bank (use bagged_v1 here)
        #   - rule (we don't have rule directly, but bagged_v1 is the closest proxy)
        #   - raw / tier1b
        v1_agree = int((v1[flip_mask] == flip_classes).sum())
        bag_agree = int((bagged_arg[flip_mask] == flip_classes).sum())
        bank_agree = int((maj[flip_mask] == flip_classes).sum())
        raw_agree = int((raw[flip_mask] == flip_classes).sum())
        t1b_agree = int((tier1b[flip_mask] == flip_classes).sum())
        agr_mean = float(agr[flip_mask].mean())
        agr_p25 = float(np.percentile(agr[flip_mask], 25))

        print(f"  axis agreement on flip rows (n={n_flips}):")
        print(f"    bagged_v1: {bag_agree}/{n_flips} ({100*bag_agree/n_flips:.1f}%)")
        print(f"    14-bank:   {bank_agree}/{n_flips} ({100*bank_agree/n_flips:.1f}%)")
        print(f"    v1 RF:     {v1_agree}/{n_flips} ({100*v1_agree/n_flips:.1f}%)")
        print(f"    raw:       {raw_agree}/{n_flips} ({100*raw_agree/n_flips:.1f}%)")
        print(f"    tier1b:    {t1b_agree}/{n_flips} ({100*t1b_agree/n_flips:.1f}%)")
        print(f"    14-bank agreement (continuous) on flip rows: mean={agr_mean:.3f} p25={agr_p25:.3f}")

        # Per-direction projection: given the worst-case axis, project precision
        # Use the axis with LOWEST agreement as proxy for true precision floor
        all_axes = {"bagged_v1": bag_agree, "14-bank": bank_agree, "v1": v1_agree, "raw": raw_agree, "tier1b": t1b_agree}
        worst_axis_name = min(all_axes, key=all_axes.get)
        worst_pct = all_axes[worst_axis_name] / n_flips

        # macro projection at observed worst-axis-agreement-% (proxy precision)
        projected_lb_at = {}
        # per-direction LB projection
        base_4b = 0.98150
        macro_total = 0.0
        for dir_name, n_dir in dirs.items():
            be = BREAK_EVEN.get(dir_name, 0.5)
            print(f"    {dir_name} ({n_dir} flips, break-even={be:.3f}):")
            # use 3 precision points
            for prec_label, prec in [("worst-axis%", worst_pct), ("v1+bank-mean", (v1_agree+bank_agree)/(2*n_flips)), ("naive 95%", 0.95)]:
                m = project_macro(dir_name, n_dir, prec)
                print(f"      @ {prec_label}={prec:.3f}: macro_delta = {m:+.6f}")
        # Sum macros across directions at worst-axis precision
        macro_worst = sum(project_macro(d, n, worst_pct) for d, n in dirs.items())
        macro_v1bank = sum(project_macro(d, n, (v1_agree+bank_agree)/(2*n_flips)) for d, n in dirs.items())
        macro_naive = sum(project_macro(d, n, 0.95) for d, n in dirs.items())
        print(f"  TOTAL projected LB delta vs 4b:")
        print(f"    worst-axis ({worst_axis_name} at {worst_pct:.3f}): {macro_worst:+.6f} -> LB {base_4b + macro_worst:.5f}")
        print(f"    v1+bank-mean ({(v1_agree+bank_agree)/(2*n_flips):.3f}): {macro_v1bank:+.6f} -> LB {base_4b + macro_v1bank:.5f}")
        print(f"    naive 95%: {macro_naive:+.6f} -> LB {base_4b + macro_naive:.5f}")

        # Verdict
        if macro_worst < -0.0001:
            verdict = "PROJECTED REGRESSION (worst-axis floor < -1bp) — DO NOT PROBE"
        elif macro_v1bank < 0:
            verdict = "BORDERLINE NEGATIVE — likely null on LB"
        elif macro_v1bank < +0.0001:
            verdict = "BORDERLINE POSITIVE — within noise floor"
        else:
            verdict = "POSITIVE PROJECTION — worth probing"
        print(f"  VERDICT: {verdict}\n")

        results[cn] = {
            "n_flips_vs_4b": n_flips,
            "directions": dirs,
            "axis_agreement_pct": {k: v/n_flips for k, v in all_axes.items()},
            "worst_axis": worst_axis_name,
            "macro_worst": float(macro_worst),
            "macro_v1bank": float(macro_v1bank),
            "verdict": verdict,
        }

    out = ART / "fresh_check_excelformer_candidates_results.json"
    out.write_text(json.dumps({
        "candidates": results,
        "break_even": BREAK_EVEN,
        "anchor_lb": 0.98150,
    }, indent=2, default=str))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
