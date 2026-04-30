"""D3 — audit gates for v1 retrained on cleanlab-cleaned labels.

Gates per CLAUDE.md leakage-defense:
  G1  standalone tuned macro vs v1 baseline 0.98063
  G2  Jaccard vs v1 + 14 bank components
  G3  minimal-input meta: log_blend(v1, D) tune over alpha
  G4  per-class recall delta (especially H, the rare class with 1.36% flag rate)
  G5  test-side argmax flip count vs v1 (sanity: too few = no signal, too many = noise)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import BANK_NAMES  # noqa: E402
from T6_diversity_helpers import (  # noqa: E402
    load_y_train,
    macro_recall,
    normed,
    tune_log_bias_simple,
)

ART = Path("scripts/artifacts")


def log_blend(arrs, alphas):
    s = np.zeros_like(arrs[0])
    for a, w in zip(arrs, alphas):
        s = s + w * np.log(np.clip(a, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def per_class_recall(y, pred):
    out = []
    for c in range(3):
        m = y == c
        if m.sum():
            out.append(float((pred[m] == c).mean()))
    return out


def audit_one(name: str, y: np.ndarray, v1: np.ndarray, sv1: float,
              v1_argmax: np.ndarray, results: dict) -> None:
    p = ART / f"oof_v1_D_{name}.npy"
    pt = ART / f"test_v1_D_{name}.npy"
    if not p.exists():
        print(f"\nMISSING {p} — skip {name}")
        return
    o = normed(np.load(p).astype(np.float32))
    t = normed(np.load(pt).astype(np.float32))

    print(f"\n=== D-{name} audit ===")
    bD, sD = tune_log_bias_simple(o, y)
    print(f"G1 standalone tuned macro = {sD:.6f}  (v1 baseline = {sv1:.6f}  delta = {sD - sv1:+.6f})")
    print(f"   bias = {bD.round(4).tolist()}")

    # Per-class recall at tuned bias
    tuned_pred = (np.log(np.clip(o, 1e-12, 1)) + bD).argmax(1)
    pcr = per_class_recall(y, tuned_pred)
    v1_tune_b, _ = tune_log_bias_simple(v1, y)
    v1_tuned_pred = (np.log(np.clip(v1, 1e-12, 1)) + v1_tune_b).argmax(1)
    pcr_v1 = per_class_recall(y, v1_tuned_pred)
    print(f"G4 per-class recall: L={pcr[0]:.5f} M={pcr[1]:.5f} H={pcr[2]:.5f}")
    print(f"   v1 baseline      L={pcr_v1[0]:.5f} M={pcr_v1[1]:.5f} H={pcr_v1[2]:.5f}")
    print(f"   delta            L={pcr[0]-pcr_v1[0]:+.5f} M={pcr[1]-pcr_v1[1]:+.5f} H={pcr[2]-pcr_v1[2]:+.5f}")

    # Jaccard vs v1 + banks
    print("G2 Jaccard:")
    d_argmax = o.argmax(1)
    j_v1 = float((d_argmax == v1_argmax).mean())
    print(f"   vs v1                                          Jaccard={j_v1:.4f}")
    j_results = {"v1": j_v1}
    for nm in BANK_NAMES:
        bp = ART / f"oof_{nm}.npy"
        if not bp.exists():
            continue
        a = normed(np.load(bp).astype(np.float32))
        ja = float((d_argmax == a.argmax(1)).mean())
        j_results[nm] = ja
        flag = "  <- low overlap" if ja < 0.97 else ""
        print(f"   {nm:>43s}  Jaccard={ja:.4f}{flag}")

    # Minimal-input meta: log_blend(v1, D)
    print("G3 minimal-input meta v1 + D:")
    best = (sv1, 0.0)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        b = log_blend([v1, o], [1.0 - alpha, alpha])
        _, sB = tune_log_bias_simple(b, y)
        marker = "  <- best" if sB > best[0] else ""
        print(f"   alpha={alpha:.2f}  tuned={sB:.6f}  delta_vs_v1={sB - sv1:+.6f}{marker}")
        if sB > best[0]:
            best = (sB, alpha)
    print(f"   best: alpha={best[1]:.2f}  macro={best[0]:.6f}  delta={best[0] - sv1:+.6f}")

    # Test-side flip count vs v1
    v1_test = normed(np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32))
    flips = int((t.argmax(1) != v1_test.argmax(1)).sum())
    print(f"G5 test-side argmax flips vs v1 test: {flips} / {len(t)}  ({flips/len(t):.3%})")

    results[name] = {
        "standalone_tuned": float(sD),
        "delta_vs_v1": float(sD - sv1),
        "pcr": pcr,
        "best_dual_alpha": float(best[1]),
        "best_dual_macro": float(best[0]),
        "best_dual_delta_vs_v1": float(best[0] - sv1),
        "jaccards": j_results,
        "test_flips_vs_v1": flips,
    }


def main() -> None:
    print("=== D audit ===")
    y = load_y_train()
    v1 = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    bv1, sv1 = tune_log_bias_simple(v1, y)
    v1_argmax = v1.argmax(1)
    print(f"v1 alone tuned macro: {sv1:.6f}")

    results = {"v1_baseline_tuned": float(sv1)}
    for name in ("drop", "relabel"):
        audit_one(name, y, v1, sv1, v1_argmax, results)

    out = ART / "D_audit_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
