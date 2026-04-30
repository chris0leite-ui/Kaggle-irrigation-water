"""Build composite candidates on top of 4b: combine M->L (strict90 / strict85 / 3axis)
with W5's 9 M->H flips. Independent-direction stack that's untested as a
single submission.

W5: 9 M->H rows. M->H break-even = 9.3% (very low). Even at 30% precision
this contributes positively under macro-recall.

Also build the strictest unprobed variant: 4b + W5(M->H) + strict90(M->L)
+ optionally 3axis. Each direction has different break-even and different
risk profile; they don't interfere with each other (disjoint rows).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")

N_L, N_M, N_H = 159460, 100261, 10279

LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def load(name: str) -> np.ndarray:
    return pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"].map(LMH_REV).to_numpy(np.int8)


def directions_breakdown(anchor, cand):
    LMH = ["L", "M", "H"]
    out = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((anchor == fr) & (cand == to)).sum())
            if n > 0:
                out[f"{LMH[fr]}->{LMH[to]}"] = n
    return out


def macro_delta(directions, precisions_per_direction):
    Ns = {"L": N_L, "M": N_M, "H": N_H}
    md = 0.0
    for d, n in directions.items():
        fr, to = d.split("->")
        p = precisions_per_direction.get(d, 0.5)
        n_corr = n * p
        n_wrong = n * (1 - p)
        md += (n_corr / Ns[to] - n_wrong / Ns[fr]) / 3
    return md


def main():
    print("=== Build composite candidates: 4b + W5(M->H) + strict M->L ===\n")
    fb = load("submission_idea4b_selective_override")
    w5 = load("submission_W5_i5_MtoH_only")
    s90 = load("submission_4b_plus_ml_strict90")
    s85 = load("submission_4b_plus_ml_strict85")
    s3ax = load("submission_4b_plus_ml_3axis")

    # Verify direction profiles
    print("Pure direction profiles vs 4b:")
    print(f"  W5:        {directions_breakdown(fb, w5)}")
    print(f"  strict90:  {directions_breakdown(fb, s90)}")
    print(f"  strict85:  {directions_breakdown(fb, s85)}")
    print(f"  3axis:     {directions_breakdown(fb, s3ax)}")
    print()

    # Composite: 4b + W5's 9 M->H + strict90's 38 M->L
    # Disjoint rows (M->L and M->H affect different cells)
    composites = []

    for ml_name, ml_pred in [("strict90", s90), ("strict85", s85), ("3axis", s3ax)]:
        comp = fb.copy()

        # Apply W5's M->H flips (rows where w5=2 and fb=1)
        mh_mask = (fb == 1) & (w5 == 2)
        comp[mh_mask] = 2

        # Apply ml's M->L flips (rows where ml=0 and fb=1)
        ml_mask = (fb == 1) & (ml_pred == 0)
        comp[ml_mask] = 0

        # Sanity: should be disjoint
        overlap = (mh_mask & ml_mask).sum()
        assert overlap == 0, f"overlap mh + ml: {overlap}"

        dirs = directions_breakdown(fb, comp)
        n_diff = int((comp != fb).sum())

        # Per-direction precision priors (calibrated from existing analysis):
        # M->H: W5 reassessment said ~39% (per CLAUDE.md). Break-even 9.3%.
        # M->L: bank-agr-conditioned on bank=L, ~55-65% range typical.
        proj_pessimistic = {
            "M->H": 0.30,
            "M->L": 0.50,
            "L->M": 0.40,
        }
        proj_realistic = {
            "M->H": 0.45,
            "M->L": 0.60,
            "L->M": 0.50,
        }
        proj_optimistic = {
            "M->H": 0.55,
            "M->L": 0.65,
            "L->M": 0.55,
        }

        d_pess = macro_delta(dirs, proj_pessimistic)
        d_real = macro_delta(dirs, proj_realistic)
        d_opt = macro_delta(dirs, proj_optimistic)

        rec = {
            "name": f"4b_plus_w5_{ml_name}",
            "diff": n_diff,
            "dirs": dirs,
            "proj_lb_pess": round(0.98150 + d_pess, 5),
            "proj_lb_real": round(0.98150 + d_real, 5),
            "proj_lb_opt": round(0.98150 + d_opt, 5),
            "pred": comp,
        }
        composites.append(rec)
        print(f"  {rec['name']:30s} diff={n_diff:3d}  dirs={rec['dirs']}")
        print(f"    proj LB: pess(@30/50%)={rec['proj_lb_pess']}  "
              f"real(@45/60%)={rec['proj_lb_real']}  opt(@55/65%)={rec['proj_lb_opt']}")
    print()

    # Emit composites — but only those with realistic ≥ 4b's 0.98150
    test_ids = list(range(900000, 900000 + 270000))  # placeholder if data missing
    if Path("data/test.csv").exists():
        test_ids = pd.read_csv("data/test.csv")["id"].tolist()
    elif Path("data/sample_submission.csv").exists():
        test_ids = pd.read_csv("data/sample_submission.csv")["id"].tolist()
    else:
        # Reuse 4b sub for IDs
        test_ids = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")["id"].tolist()

    for rec in composites:
        if rec["proj_lb_real"] < 0.98150 - 0.00003:
            print(f"  skip emit {rec['name']}: realistic projection regression")
            continue
        out = SUB / f"submission_{rec['name']}.csv"
        df = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(rec["pred"]).map(LMH_NAMES),
        })
        df.to_csv(out, index=False)
        print(f"  emitted: {out.name}  diff={rec['diff']}  "
              f"proj_real_LB={rec['proj_lb_real']}")

    print()
    print("=== Recommendation ===")
    print()
    print("The composite stacks two independent direction signals:")
    print("  - W5's 9 M->H flips at break-even 9.3% (very low bar)")
    print("  - strict90's 38 M->L flips at break-even 61.4% (moderate)")
    print()
    print("Pessimistic case: M->H@30%, M->L@50% → tiny regression (~-2bp)")
    print("Realistic case:   M->H@45%, M->L@60% → tiny lift (~+1bp)")
    print("Optimistic case:  M->H@55%, M->L@65% → modest lift (~+3-4bp)")
    print()
    print("Compared to strict90 alone (38 M->L): same risk floor, ")
    print("slightly higher ceiling because W5's 9 M->H rows have very low")
    print("break-even — they almost can't hurt at any reasonable precision.")
    print()
    print("Asymmetric-upside composite candidate: 4b_plus_w5_strict90")


if __name__ == "__main__":
    main()
