"""Cleanlab diagnostic on LB-best blend OOF.

Step 1 of A0: use the LB-best 2-way blend (recipe * pseudolabel) as the
teacher posterior. Feed its per-row OOF probabilities to
`cleanlab.filter.find_label_issues` to flag candidate label-noise rows
in the 630k synthetic train set, then characterize:

1. How many rows flagged, per class?
2. Do the flagged rows concentrate on DGP score {3, 6, 7, 8} — the
   boundary-band scores where the ~10,304 known NN flips live?
3. What fraction of flagged rows disagree with rule_pred in the
   direction the rule predicts?
4. Self-confidence distribution: are flagged rows low-confidence
   (boundary) or high-confidence-wrong (strong signal flips)?

Outputs
-------
scripts/artifacts/cleanlab_label_issues.npy  (bool mask, length 630k)
scripts/artifacts/cleanlab_diagnose.json     (summary stats)

No retraining. This is only diagnosis. Intervention (drop/reweight)
happens in a follow-up script IF the diagnosis is promising.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues

from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
CLASSES = ("Low", "Medium", "High")
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def rule_predict(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    pred = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    return score.to_numpy(dtype=np.int8), pred


def log_blend(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    la = np.log(np.clip(a, 1e-12, 1.0))
    lb = np.log(np.clip(b, 1e-12, 1.0))
    z = w * la + (1.0 - w) * lb
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(CLASS_TO_INT).to_numpy(dtype=np.int64)
    score, rule_pred = rule_predict(train)
    print(f"[load] train {len(train):,} rows  y classes {np.bincount(y)}")

    recipe = np.load(ART / "oof_recipe_full_te.npy")
    pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    assert recipe.shape == (len(train), 3), recipe.shape
    assert pseudo.shape == (len(train), 3), pseudo.shape
    teacher = log_blend(recipe, pseudo, 0.5)
    teacher_pred = teacher.argmax(axis=1)
    teacher_argmax_bal = balanced_accuracy_score(y, teacher_pred)
    teacher_entropy = -(teacher * np.log(np.clip(teacher, 1e-12, 1))).sum(axis=1)
    print(f"[teacher] 50/50 log-blend  argmax bal={teacher_argmax_bal:.6f}  "
          f"mean_entropy={teacher_entropy.mean():.4f}")

    for method in ("prune_by_noise_rate",):
        t1 = time.time()
        issues = find_label_issues(
            labels=y,
            pred_probs=teacher,
            filter_by=method,
            return_indices_ranked_by=None,
            verbose=False,
        )
        elapsed = time.time() - t1
        n_issues = int(issues.sum())
        pct = 100.0 * n_issues / len(y)
        print(f"[cleanlab {method}]  issues={n_issues:,} ({pct:.3f}%)  in {elapsed:.1f}s")

        flagged_by_class = [int((issues & (y == k)).sum()) for k in range(3)]
        flagged_frac_by_class = [
            flagged_by_class[k] / int((y == k).sum()) for k in range(3)
        ]
        print(f"  flagged by observed class  Low={flagged_by_class[0]:,}  "
              f"Medium={flagged_by_class[1]:,}  High={flagged_by_class[2]:,}")
        print(f"  frac per class  Low={flagged_frac_by_class[0]:.4f}  "
              f"Medium={flagged_frac_by_class[1]:.4f}  High={flagged_frac_by_class[2]:.4f}")

        flagged_score_hist = np.bincount(score[issues], minlength=10)
        total_score_hist = np.bincount(score, minlength=10)
        print("  flagged by DGP score:")
        for s in range(10):
            tot = total_score_hist[s]
            flg = flagged_score_hist[s]
            pct_s = 100.0 * flg / max(tot, 1)
            print(f"    score={s}: {flg:5d}/{tot:6d} ({pct_s:5.2f}%)")

        rule_mismatch = (y != rule_pred)
        flagged_rule_mismatch = int((issues & rule_mismatch).sum())
        total_rule_mismatch = int(rule_mismatch.sum())
        capture_rate = flagged_rule_mismatch / max(total_rule_mismatch, 1)
        precision_vs_rule = flagged_rule_mismatch / max(n_issues, 1)
        print(f"  rule-vs-observed mismatch total = {total_rule_mismatch:,}")
        print(f"  cleanlab flagged of those       = {flagged_rule_mismatch:,} "
              f"({capture_rate:.3f} recall, {precision_vs_rule:.3f} precision)")

        teacher_agrees_with_rule = (teacher_pred == rule_pred)
        flagged_agree_with_rule = int((issues & teacher_agrees_with_rule).sum())
        print(f"  flagged rows where teacher_pred == rule_pred: "
              f"{flagged_agree_with_rule:,} ({100.0 * flagged_agree_with_rule / max(n_issues, 1):.2f}%)")

        self_conf = teacher[np.arange(len(y)), y]
        flagged_self_conf_mean = float(self_conf[issues].mean())
        unflagged_self_conf_mean = float(self_conf[~issues].mean())
        print(f"  mean teacher self-confidence  flagged={flagged_self_conf_mean:.4f}  "
              f"unflagged={unflagged_self_conf_mean:.4f}")

        teacher_pred_matches_y = (teacher_pred == y)
        flagged_and_teacher_right = int((issues & teacher_pred_matches_y).sum())
        flagged_and_teacher_wrong = int((issues & ~teacher_pred_matches_y).sum())
        print(f"  teacher agrees with observed label on flagged: "
              f"{flagged_and_teacher_right:,}  "
              f"disagrees: {flagged_and_teacher_wrong:,}")

        out_mask = ART / f"cleanlab_issues_{method}.npy"
        np.save(out_mask, issues)
        summary = dict(
            method=method,
            n_total=int(len(y)),
            n_issues=n_issues,
            pct_flagged=pct,
            teacher_oof_argmax_bal=float(teacher_argmax_bal),
            flagged_by_class=flagged_by_class,
            flagged_frac_by_class=flagged_frac_by_class,
            flagged_by_score={int(s): int(flagged_score_hist[s]) for s in range(10)},
            total_by_score={int(s): int(total_score_hist[s]) for s in range(10)},
            rule_mismatch_total=int(total_rule_mismatch),
            flagged_rule_mismatch=int(flagged_rule_mismatch),
            capture_rate_vs_rule=float(capture_rate),
            precision_vs_rule=float(precision_vs_rule),
            flagged_teacher_agrees_with_rule=int(flagged_agree_with_rule),
            mean_self_conf_flagged=float(flagged_self_conf_mean),
            mean_self_conf_unflagged=float(unflagged_self_conf_mean),
            flagged_and_teacher_right=int(flagged_and_teacher_right),
            flagged_and_teacher_wrong=int(flagged_and_teacher_wrong),
            elapsed_sec=float(elapsed),
        )
        out_json = ART / f"cleanlab_diagnose_{method}.json"
        out_json.write_text(json.dumps(summary, indent=2))

    print(f"[done] {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
