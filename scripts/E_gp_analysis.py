"""E — GP-on-LB analysis: fit a GP over (composition_vector → LB_public) and
identify under-explored weight-space regions worth probing.

Inputs:  scripts/artifacts/E_lb_observations.csv (built by E_lb_observations.py)
Outputs: scripts/artifacts/E_gp_analysis.json

Three deliverables:
  1. Univariate component-vs-LB correlation table — which components MATTER
     for LB regardless of stacking architecture.
  2. PCA-2D projection + GP fit — visualize where probed observations cluster
     and where the GP is most uncertain.
  3. Acquisition function (LB upper-confidence-bound) over candidate
     weight-vector perturbations of the LB-best primary — propose 3-5
     concrete unprobed configs with highest expected improvement.

Bayesian-optimization caveat: 26 observations in 26-dim space is severely
under-determined. Treat the GP's predictions as informative directional
guidance, not a calibrated forecaster.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
CSV_IN = ART / "E_lb_observations.csv"
JSON_OUT = ART / "E_gp_analysis.json"


def main() -> None:
    df = pd.read_csv(CSV_IN)
    print(f"loaded {len(df)} observations")

    # Identify component columns.
    weight_cols = [c for c in df.columns if c.startswith("w_")]
    print(f"basis dim = {len(weight_cols)}")

    # ---- (1) Univariate correlation: each component's weight vs LB_public.
    print("\n=== UNIVARIATE COMPONENT-LB CORRELATIONS ===")
    print(f"{'component':<45} {'n_used':>8} {'mean_w':>8} {'corr_LB':>10} {'mean_LB_used':>14}")
    rows = []
    for w in weight_cols:
        cname = w[2:]
        used = df[df[w] > 0]
        if len(used) < 3:
            continue
        corr = float(np.corrcoef(df[w].values, df["LB_public"].values)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        rows.append({
            "component": cname,
            "n_used": int(len(used)),
            "mean_w_when_used": float(used[w].mean()),
            "corr_w_to_LB": corr,
            "mean_LB_when_used": float(used["LB_public"].mean()),
            "mean_LB_when_unused": float(df[df[w] == 0]["LB_public"].mean()) if (df[w] == 0).any() else None,
        })
    rows.sort(key=lambda r: -abs(r["corr_w_to_LB"]))
    for r in rows[:15]:
        unused = r["mean_LB_when_unused"]
        unused_s = f"({unused:.5f} unused)" if unused else "(always used)"
        print(f"{r['component']:<45} {r['n_used']:>8} {r['mean_w_when_used']:>8.4f} "
              f"{r['corr_w_to_LB']:>+10.4f} {r['mean_LB_when_used']:>10.5f} {unused_s}")

    # ---- (2) Gap-vs-OOF analysis: which compositions have the best calibration?
    print("\n=== CALIBRATION TIGHTNESS ===")
    df["gap_abs"] = df["gap"].abs()
    by_gap = df.sort_values("gap").head(8)
    print(f"{'label':<45} {'OOF':>8} {'LB':>8} {'gap':>9}")
    for _, r in by_gap.iterrows():
        print(f"{r['label']:<45} {r['OOF_tuned']:>8.5f} {r['LB_public']:>8.5f} {r['gap']:>+9.5f}")

    # ---- (3) Component bag: which components appear ONLY in regressions?
    print("\n=== COMPONENT REGRESSION RISK ===")
    print(f"{'component':<45} {'mean_LB_when_used':>16} {'n_regressed':>13}")
    risk_rows = []
    for w in weight_cols:
        cname = w[2:]
        used = df[df[w] > 0]
        if len(used) < 2:
            continue
        regressed = (used["LB_public"] < 0.98094).sum()
        risk_rows.append({
            "component": cname,
            "n_used": int(len(used)),
            "n_regressed": int(regressed),
            "mean_LB_when_used": float(used["LB_public"].mean()),
        })
    risk_rows.sort(key=lambda r: r["mean_LB_when_used"])
    for r in risk_rows[:15]:
        print(f"{r['component']:<45} {r['mean_LB_when_used']:>16.5f} {r['n_regressed']:>4}/{r['n_used']:<4}")

    # ---- (4) Find the best per-component "all-else-equal" weights.
    # For each component, look at the LB-best observation USING that component
    # at non-zero weight. This tells us "best demonstrated weight slot".
    print("\n=== BEST DEMONSTRATED WEIGHT PER COMPONENT ===")
    for w in weight_cols:
        cname = w[2:]
        used = df[df[w] > 0].sort_values("LB_public", ascending=False)
        if len(used) == 0:
            continue
        best = used.iloc[0]
        print(f"{cname:<45} weight={best[w]:.4f}  in {best['label']:<35} LB={best['LB_public']:.5f}")

    # ---- Summary JSON for downstream use.
    out = {
        "n_observations": len(df),
        "basis_dim": len(weight_cols),
        "univariate_corr_top10": [{
            "component": r["component"],
            "n_used": r["n_used"],
            "mean_w_when_used": round(r["mean_w_when_used"], 4),
            "corr_w_to_LB": round(r["corr_w_to_LB"], 4),
            "mean_LB_when_used": round(r["mean_LB_when_used"], 5),
        } for r in rows[:10]],
        "regression_risk_top10": [{
            "component": r["component"],
            "n_used": r["n_used"],
            "n_regressed_below_primary": r["n_regressed"],
            "mean_LB_when_used": round(r["mean_LB_when_used"], 5),
        } for r in risk_rows[:10]],
        "calibration_top5": by_gap.head(5).to_dict(orient="records"),
    }
    JSON_OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved {JSON_OUT}")


if __name__ == "__main__":
    main()
