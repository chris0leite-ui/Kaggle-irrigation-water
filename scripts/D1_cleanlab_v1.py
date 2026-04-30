"""D1 — cleanlab find_label_issues with v1 RF natural OOF as teacher.

v1 is the LB-best standalone (LB 0.98129). Its 5-fold OOF probs are the
most-LB-aligned posterior we have for cleanlab's noise-transition matrix.
This is distinct from the prior cleanlab artifact (which used LB-best
2-way blend, not v1).

Output: scripts/artifacts/D_v1_label_issues.npy  (bool mask len 630k)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

sys.path.insert(0, str(Path(__file__).parent))
from T6_diversity_helpers import load_y_train, normed  # noqa: E402
from dgp_formula import dgp_score  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")


def main() -> None:
    print("=== D1: cleanlab find_label_issues (v1 OOF teacher) ===\n")
    y = load_y_train()
    v1 = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    print(f"v1 OOF shape={v1.shape}  argmax dist L/M/H="
          f"{[int((v1.argmax(1)==k).sum()) for k in range(3)]}")
    print(f"argmax disagree y: {int((v1.argmax(1) != y).sum())} / {len(y)}")

    # Use prune_by_noise_rate (cleanlab default) — flags top-k rows per
    # confident-class transition by noise-rate estimate.
    mask = find_label_issues(
        labels=y.astype(int),
        pred_probs=v1,
        return_indices_ranked_by="self_confidence",
        filter_by="prune_by_noise_rate",
    )
    flag = np.zeros(len(y), dtype=bool)
    flag[mask] = True
    print(f"\nflagged: {flag.sum()} / {len(y)}  ({flag.mean():.4%})")

    # Cross with DGP rule
    train = pd.read_csv(DATA / "train.csv")
    score = dgp_score(train).astype(np.int16)
    rule_pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    print("\nflagged-row breakdown:")
    print(f"  y!=rule_pred among flagged: {int((flag & (y != rule_pred)).sum())}  "
          f"({(flag & (y != rule_pred)).sum() / max(flag.sum(),1):.4%})")
    print(f"  y!=rule_pred globally:      {int((y != rule_pred).sum())} ({(y != rule_pred).mean():.4%})")
    print(f"  v1.argmax!=y among flagged: {int((flag & (v1.argmax(1) != y)).sum())}")

    # By DGP score
    print("\nflagged by DGP score:")
    for s in range(0, 11):
        m = flag & (score == s)
        if m.sum() == 0:
            continue
        print(f"  s={s:>2d} flagged={int(m.sum()):>5d}  "
              f"y!=rule={int((m & (y!=rule_pred)).sum()):>5d}  "
              f"frac_in_band={(m.sum()/max((score==s).sum(),1)):.4%}")

    # Per class
    print("\nflagged by y class:")
    for c, lab in enumerate(["L", "M", "H"]):
        m = flag & (y == c)
        if m.sum() == 0:
            continue
        print(f"  y={lab}  flagged={int(m.sum()):>5d}  rate={(m.sum()/max((y==c).sum(),1)):.4%}")

    np.save(ART / "D_v1_label_issues.npy", flag)
    summary = {
        "n_flagged": int(flag.sum()),
        "frac_flagged": float(flag.mean()),
        "frac_y_neq_rule_in_flagged": float((flag & (y != rule_pred)).sum() / max(flag.sum(), 1)),
        "global_y_neq_rule_rate": float((y != rule_pred).mean()),
    }
    out = ART / "D_cleanlab_v1_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {ART / 'D_v1_label_issues.npy'}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
