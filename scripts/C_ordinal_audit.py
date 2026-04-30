"""C audit — leakage-defense gates for the ordinal cumulative-link OOF.

Gates (per CLAUDE.md leakage-defense rule + Caruana add-step):
  G0  artifact present + OOF non-trivially calibrated
  G1  standalone tuned macro vs recipe_full_te baseline (recipe-bias)
  G2  Jaccard vs each of the 14 existing bank components (need <0.97 to clear
      stacking-inflation ceiling for stacker addition)
  G3  Caruana add-step gain to v1 RF natural standalone (the LB-best path)
  G4  minimal-input meta: 2-comp (v1 + ordinal) tune. If <= v1 alone -> drop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import BANK_NAMES  # noqa: E402
from T6_diversity_helpers import (  # noqa: E402
    argmax_jaccard,
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


def main() -> None:
    print("=== C ordinal audit ===\n")
    y = load_y_train()

    # G0 — artifact present
    oof_path = ART / "oof_C_ordinal_xgb.npy"
    if not oof_path.exists():
        print(f"MISSING {oof_path} — production run not finished. Abort.")
        return
    o_C = normed(np.load(oof_path).astype(np.float32))
    print(f"G0 oof_C shape={o_C.shape}  argmax dist L/M/H="
          f"{[int((o_C.argmax(1)==k).sum()) for k in range(3)]}")

    # G1 — standalone tuned macro
    raw_arg = balanced_recall(y, o_C.argmax(1))
    bC, sC = tune_log_bias_simple(o_C, y)
    print(f"G1 raw argmax macro = {raw_arg:.6f}")
    print(f"G1 tuned macro      = {sC:.6f}  bias={bC.round(4).tolist()}")
    rfte_path = ART / "oof_recipe_full_te.npy"
    if rfte_path.exists():
        rfte = normed(np.load(rfte_path).astype(np.float32))
        _, sR = tune_log_bias_simple(rfte, y)
        print(f"   recipe_full_te  tuned = {sR:.6f}   delta_C-rfte = {sC - sR:+.6f}")

    # G2 — Jaccard vs 14 banks
    print("\nG2 Jaccard (argmax) vs 14 bank components:")
    cargmax = o_C.argmax(1)
    jaccards = {}
    for nm in BANK_NAMES:
        p = ART / f"oof_{nm}.npy"
        if not p.exists():
            print(f"   {nm}: MISSING")
            continue
        a = normed(np.load(p).astype(np.float32))
        ja = (cargmax == a.argmax(1)).mean()
        jaccards[nm] = float(ja)
        flag = " <= 0.97 (low overlap, candidate)" if ja < 0.97 else ""
        print(f"   {nm:>45s}  Jaccard={ja:.4f}{flag}")

    # G4 — minimal-input meta vs v1 alone
    v1_path = ART / "oof_sklearn_rf_meta_natural.npy"
    v1 = normed(np.load(v1_path).astype(np.float32))
    bv1, sv1 = tune_log_bias_simple(v1, y)
    print(f"\nG4 v1 alone tuned macro = {sv1:.6f}")
    # alpha sweep on log-blend(v1, C)
    print("   2-comp log-blend (v1, C) tuned:")
    best_dual = (sv1, 0.0)
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        blend = log_blend([v1, o_C], [1.0 - alpha, alpha])
        _, sB = tune_log_bias_simple(blend, y)
        marker = "  <- new best" if sB > best_dual[0] else ""
        print(f"     alpha={alpha:.2f}  tuned macro={sB:.6f}  delta_vs_v1={sB - sv1:+.6f}{marker}")
        if sB > best_dual[0]:
            best_dual = (sB, alpha)
    print(f"   best 2-comp: alpha={best_dual[1]:.2f}  macro={best_dual[0]:.6f}  delta={best_dual[0] - sv1:+.6f}")

    # G3 — Caruana add-step to v1: argmax_alpha macro(log_blend(v1, C; 1-a, a)) - lambda * jaccard
    # already done in G4; emit summary
    out = {
        "raw_argmax": float(raw_arg),
        "standalone_tuned": float(sC),
        "v1_alone_tuned": float(sv1),
        "best_dual_alpha": float(best_dual[1]),
        "best_dual_macro": float(best_dual[0]),
        "best_dual_delta_vs_v1": float(best_dual[0] - sv1),
        "jaccards": jaccards,
    }
    out_p = ART / "C_ordinal_audit_results.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_p}")


def balanced_recall(y, p):
    rec = []
    for c in range(3):
        m = y == c
        if m.sum() == 0:
            continue
        rec.append((p[m] == c).mean())
    return float(np.mean(rec))


if __name__ == "__main__":
    main()
