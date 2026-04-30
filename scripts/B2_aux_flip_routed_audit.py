"""B2 — aux-flip-detector-routed override audit on 4b.

Distinct from B (which used DGP score-band as flip-likelihood proxy):
  - The aux_flipped_from_rule head is a LEARNED predictor of P(y != rule).
  - When aux_flip_prob is LOW, the host NN likely did NOT flip this row
    -> y = rule_pred with high confidence, so any 4b disagreement is 4b noise.
  - Override 4b -> rule_pred when (rule != 4b) AND (aux_flip < tau).

Distinct from #3 multitask-aux meta-stacker (saturated as a stacker feature):
  - Here, aux_flip is a HARD ROUTING GATE, not a stacker input feature.
  - The override is a per-row decision, not an argmax-of-blend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from B_rule_anchor_precision_audit import build_4b_oof_analog  # noqa: E402
from T6_diversity_helpers import load_y_train, macro_recall  # noqa: E402
from dgp_formula import dgp_score  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")


def main() -> None:
    print("=== B2: aux-flip-routed override audit ===\n")
    train = pd.read_csv(DATA / "train.csv")
    y = load_y_train()
    score = dgp_score(train).astype(np.int16)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    fb = build_4b_oof_analog(y)
    aux = np.load(ART / "oof_aux_flipped_from_rule.npy").astype(np.float32)

    print(f"aux_flip stats: min={aux.min():.4f} median={np.median(aux):.4f} "
          f"q90={np.quantile(aux,0.9):.4f} max={aux.max():.4f}")
    print(f"y vs rule mismatch rate: {(y != rule_pred).mean():.5f}")
    print(f"aux_flip top-1% AUC sanity (high flip_prob -> mismatch): "
          f"{((aux > np.quantile(aux,0.99)) & (y != rule_pred)).sum() / (aux > np.quantile(aux,0.99)).sum():.4f}")

    # Base OOF macro
    base = macro_recall(y, fb)
    print(f"\n4b OOF analog macro: {base:.6f}")

    diff = rule_pred != fb
    print(f"\nrule != 4b: {int(diff.sum())} TRAIN OOF rows")

    # Sweep aux_flip threshold (low aux -> trust rule, override)
    print("\n--- aux_flip < tau filter (rule != 4b) ---")
    print("Read: 'prec' = P(y == rule_pred | filter); break-even ~92% after haircut\n")
    sweep = []
    for tau in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        m = diff & (aux < tau)
        n = int(m.sum())
        if n == 0:
            continue
        prec = float((rule_pred[m] == y[m]).mean())
        new_pred = fb.copy()
        new_pred[m] = rule_pred[m]
        delta = macro_recall(y, new_pred) - base
        # direction breakdown (top-3)
        dirs = []
        for fr in range(3):
            for to in range(3):
                if fr == to:
                    continue
                mm = m & (fb == fr) & (rule_pred == to)
                if mm.sum() == 0:
                    continue
                dirs.append((f"{fr}->{to}", int(mm.sum()), float((rule_pred[mm] == y[mm]).mean())))
        print(f"  aux<{tau:.3f}  n={n:>6d}  prec={prec:.4f}  oof-delta={delta:+.6f}")
        for k, n_, p_ in dirs:
            print(f"     dir {k}  n={n_:>5d}  prec={p_:.4f}")
        sweep.append({"tau": tau, "n": n, "prec": prec, "oof_delta": delta,
                      "directions": [{"d": k, "n": n_, "prec": p_} for k, n_, p_ in dirs]})

    # Cross with bank-confirms (the proven 4b axis)
    print("\n--- aux_flip < tau AND bank_argmax == rule_pred ---")
    from T2_conformal_helpers import bank_mean_probs, load_bank
    bank = load_bank("oof")
    bm = bank_mean_probs(bank)
    bank_argmax = bm.argmax(axis=1).astype(np.int8)
    bank_confirms = bank_argmax == rule_pred

    sweep_bc = []
    for tau in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        m = diff & bank_confirms & (aux < tau)
        n = int(m.sum())
        if n == 0:
            continue
        prec = float((rule_pred[m] == y[m]).mean())
        new_pred = fb.copy()
        new_pred[m] = rule_pred[m]
        delta = macro_recall(y, new_pred) - base
        print(f"  aux<{tau:.3f}  bank-confirms  n={n:>6d}  prec={prec:.4f}  oof-delta={delta:+.6f}")
        sweep_bc.append({"tau": tau, "n": n, "prec": prec, "oof_delta": delta})

    out = ART / "B2_aux_flip_routed_audit_results.json"
    out.write_text(json.dumps({
        "base_macro": base,
        "n_diff": int(diff.sum()),
        "sweep_diff": sweep,
        "sweep_diff_bank_confirms": sweep_bc,
    }, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
