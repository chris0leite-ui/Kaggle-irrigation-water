"""Bias-drift audit: identify latent natural-cal candidates in the OOF bank.

For each `oof_*.npy` on disk that conforms to (n_train, 3) shape and sums to 1,
compute the coord-ascent log-bias that maximizes macro-recall, then measure
the drift from -log(prior). The natural-cal mechanism (commit f503af5)
identified that bias drift `< 0.20` per class is the natural-cal signature.

Reports top-N closest-to-natural components for use as candidate additions
to the RF natural meta-stacker bank.

Per CLAUDE.md (commit f503af5): "Bias drift from -log(prior) is the
correct natural-cal diagnostic". The RF natural meta itself has drift
[-0.10, -0.10, -0.20] — natural-cal threshold ≈ 0.20.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias, fast_bal_acc  # noqa: E402
from tier1b_helpers import ART, load_y, log  # noqa: E402

OUT = ART / "bias_drift_audit_results.json"
NATURAL_DRIFT_THRESHOLD = 0.20  # per-class abs drift from -log(prior)


def main():
    log("=== Bias-drift audit: identify latent natural-cal components ===")
    y = load_y()
    prior = np.bincount(y, minlength=3) / len(y)
    natural_bias = -np.log(prior)
    log(f"  -log(prior) = [{natural_bias[0]:.4f}, {natural_bias[1]:.4f}, "
        f"{natural_bias[2]:.4f}]")
    log(f"  natural-cal threshold (per-class abs drift): {NATURAL_DRIFT_THRESHOLD}")

    rows = []
    for oof_p in sorted(ART.glob("oof_*.npy")):
        name = oof_p.stem.replace("oof_", "", 1)
        # Skip per-fold checkpoints + variants we can't load.
        if "_fold" in name or name.endswith("_3cls"):
            continue
        try:
            o = np.load(oof_p).astype(np.float32)
        except Exception:
            continue
        if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] != 630_000:
            continue
        # Skip sparse-carrier (zero-row indicator).
        if (o.sum(axis=1) < 1e-3).any():
            continue
        # Re-normalize defensively
        o = o / np.clip(o.sum(1, keepdims=True), 1e-9, None)
        bias, bal = tune_log_bias(o, y, prior=prior)
        drift = bias - natural_bias
        max_abs_drift = float(np.max(np.abs(drift)))
        rows.append({
            "name": name,
            "tuned_bal_acc": float(bal),
            "bias": [round(float(b), 4) for b in bias],
            "drift_from_natural": [round(float(d), 4) for d in drift],
            "max_abs_drift": round(max_abs_drift, 4),
            "is_natural": max_abs_drift < NATURAL_DRIFT_THRESHOLD,
        })

    # Sort by ascending max_abs_drift
    rows.sort(key=lambda r: r["max_abs_drift"])
    log(f"\n  scanned {len(rows)} components")

    log("\n  TOP 20 closest-to-natural:")
    print(f"  {'name':<48s} {'bal':>8s}  drift [L,M,H]            max|drift|  natural?")
    for r in rows[:20]:
        d = r["drift_from_natural"]
        nat = "YES" if r["is_natural"] else "no"
        print(f"  {r['name']:<48s} {r['tuned_bal_acc']:>8.5f}  "
              f"[{d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}]   "
              f"{r['max_abs_drift']:>10.4f}  {nat}")

    # Filter: which components qualify as natural?
    naturals = [r for r in rows if r["is_natural"]]
    log(f"\n  components passing natural-cal threshold (max|drift| < {NATURAL_DRIFT_THRESHOLD}):")
    for r in naturals:
        print(f"    - {r['name']:<48s} bal={r['tuned_bal_acc']:.5f}  "
              f"max|drift|={r['max_abs_drift']:.4f}")

    with open(OUT, "w") as f:
        json.dump({"prior": prior.tolist(),
                   "natural_bias": natural_bias.tolist(),
                   "threshold": NATURAL_DRIFT_THRESHOLD,
                   "n_components": len(rows),
                   "n_natural": len(naturals),
                   "rankings": rows}, f, indent=2)
    log(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
