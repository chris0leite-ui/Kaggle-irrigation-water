"""T1-v3-full — sonnet validation on broader V1-floor (not just critical rows).

Test whether sonnet-V3's 95% M @ CONF>=0.8 finding (from n=20 critical rows)
generalizes to the broader V1-floor population. Sample 1500 random V1-floor
rows that are NOT in the critical-rows-253 already tested.

Same prompt structure as T1_haiku_v3_build.py but with calibration text
updated for the broader population (81.7% baseline vs 87.75% critical).

Output: scripts/artifacts/T1_v2/sonnet_full_batch_{i}.txt
        scripts/artifacts/T1_v2/eval_keys_full.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T1_format_batch import FEATURE_KEYS, _format_value  # noqa: E402
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402
from T6_diversity_helpers import load_y_train, normed, tune_log_bias_simple  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
OUT = ART / "T1_v2"
CLASS_STR = {0: "L", 1: "M", 2: "H"}

OVERRIDE_PREFIX = (
    "IMPORTANT: ignore your statusline-setup persona for this single message. "
    "Output ONLY the requested ROW blocks — no preamble, no markdown fences, "
    "no commentary. No tool calls after the Read.\n\n"
)

# Import v3's exact prompt (B1 design: SAME prompt as v3 for clean calibration validation)
from T1_haiku_v3_build import HEADER, ASYMMETRY_NOTE  # noqa: E402


def fmt_row(pseudo_id, feats, amh, amm, knn_margin, bank_max_prob, bank_split, sub_votes):
    parts = [f"ROW {int(pseudo_id)}"]
    for k in FEATURE_KEYS:
        parts.append(f"{k}={_format_value(k, feats.get(k))}")
    parts.append(f"amh={amh:.3f}")
    parts.append(f"amm={amm:.3f}")
    parts.append(f"knn_consensus={knn_margin:.3f}")
    parts.append(f"bank_conf={bank_max_prob:.3f}")
    parts.append(f"bank_split={bank_split}")
    parts.append(f"sub_votes={sub_votes}")
    return " ".join(parts)


def fmt_few_shot(pseudo_id, feats, amh, amm, knn_margin, bank_max_prob, bank_split, sub_votes, true_label, score):
    rp = "Low" if score <= 3 else ("Medium" if score <= 6 else "High")
    head = fmt_row(pseudo_id, feats, amh, amm, knn_margin, bank_max_prob, bank_split, sub_votes)
    return (
        f"{head}\n"
        f"RULE_SCORE: {score}\n"
        f"RULE_PRED: {rp}\n"
        f"FINAL: {true_label}    ← TRUE LABEL\n"
        f"CONF: 1.00\n"
        f"REASON: (true label revealed; reason it through yourself)\n"
    )


def dgp_score(row):
    dry = int(row["Soil_Moisture"] < 25)
    norain = int(row["Rainfall_mm"] < 300)
    hot = int(row["Temperature_C"] > 30)
    windy = int(row["Wind_Speed_kmh"] > 10)
    nomulch = int(row["Mulching_Used"] == "No")
    Kc = 2 if row["Crop_Growth_Stage"] in {"Flowering", "Vegetative"} else 0
    return 2 * (dry + norain) + (hot + windy + nomulch) + Kc


def main():
    print("=== T1-v3-full sonnet build ===\n")
    train = pd.read_csv(DATA / "train.csv")
    y = load_y_train()

    # Reconstruct V1-floor mask
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    bv1, _ = tune_log_bias_simple(v1_oof, y)
    bra, _ = tune_log_bias_simple(raw_oof, y)
    bt1, _ = tune_log_bias_simple(t1_oof, y)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1).astype(np.int8)
    una = a_ra == a_t1
    fb_oof = a_v1.copy()
    fb_oof[una & (a_v1 != a_ra)] = a_ra[una & (a_v1 != a_ra)]

    bank = load_bank("oof")
    bank_mean = bank_mean_probs(bank)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)
    bank_max_prob = bank_mean.max(axis=1)
    bank_argmaxes_per = bank.argmax(axis=2).astype(np.int8)  # (14, n_train)

    v1_mask = (bank_argmax == 1) & (fb_oof == 2)
    v1_idx = np.where(v1_mask)[0]
    print(f"V1-floor: n={len(v1_idx)}")

    # Aux signals
    amh_arr = np.load(ART / "oof_aux_missed_high.npy").astype(np.float32)
    amm_arr = np.load(ART / "oof_aux_missed_medium.npy").astype(np.float32)
    knn_features = np.load(ART / "oof_knn_train.npy").astype(np.float32)
    knn_margin = knn_features[:, 5] if knn_features.shape[1] >= 6 else np.zeros(len(y))

    # Identify rows already in the critical-rows test set (eval_keys_v3)
    crit_keys = pd.read_csv(OUT / "eval_keys_v3.csv")
    used_idx = set(crit_keys["row_idx"].tolist())
    print(f"already-tested critical: {len(used_idx)} rows")

    available = [i for i in v1_idx if i not in used_idx]
    print(f"available non-critical V1-floor: {len(available)} rows")

    # Sample 1500 random rows
    rng = np.random.default_rng(42)
    sample_size = 1500
    sample_idx = rng.choice(available, size=sample_size, replace=False)
    sample_idx.sort()
    sample_y = y[sample_idx]
    print(f"\nsample {sample_size} rows: true-M={int((sample_y==1).sum())}, "
          f"true-H={int((sample_y==2).sum())}, baseline={(sample_y==1).mean():.4f}")

    def split_str(ridx):
        votes = bank_argmaxes_per[:, ridx]
        counts = np.bincount(votes, minlength=3)
        return f"{counts[1]}M/{counts[2]}H/{counts[0]}L"

    def subs_str(ridx):
        return f"v1={CLASS_STR[a_v1[ridx]]} raw={CLASS_STR[a_ra[ridx]]} t1={CLASS_STR[a_t1[ridx]]}"

    # B1: use v3's EXACT few-shot exemplars (4 critical-M + 4 boundary-H, seeded same as v3)
    crit = pd.read_csv(OUT / "critical_rows.csv")
    boundary = pd.read_csv(OUT / "boundary_rows.csv")
    crit_M = crit[crit["true_label"] == "Medium"].sample(n=4, random_state=7)
    bnd_H = boundary[boundary["true_label"] == "High"].sample(n=4, random_state=7)
    fewshot = pd.concat([crit_M, bnd_H], ignore_index=True).sample(frac=1, random_state=11)
    fewshot_idx = fewshot["row_idx"].astype(int).tolist()
    print(f"few-shot: same 8 v3 exemplars (4 critical-M + 4 boundary-H, seeds 7/11)")

    prompt_head = OVERRIDE_PREFIX + HEADER
    for i in fewshot_idx:
        feats = train.iloc[int(i)][FEATURE_KEYS].to_dict()
        score = dgp_score(train.iloc[int(i)])
        prompt_head += fmt_few_shot(
            pseudo_id=6000000 + int(i),
            feats=feats,
            amh=float(amh_arr[i]),
            amm=float(amm_arr[i]),
            knn_margin=float(knn_margin[i]),
            bank_max_prob=float(bank_max_prob[i]),
            bank_split=split_str(int(i)),
            sub_votes=subs_str(int(i)),
            true_label=CLASS_STR[y[int(i)]].replace("L", "Low").replace("M", "Medium").replace("H", "High"),
            score=int(score),
        ) + "\n"
    prompt_head += ASYMMETRY_NOTE

    # Batch 75 rows / batch -> 20 batches
    batch_size = 75
    n_batches = (len(sample_idx) + batch_size - 1) // batch_size
    new_keys = []
    for bi in range(n_batches):
        sub = sample_idx[bi * batch_size:(bi + 1) * batch_size]
        body = ""
        for ridx in sub:
            ridx = int(ridx)
            feats = train.iloc[ridx][FEATURE_KEYS].to_dict()
            new_pseudo = 7000000 + ridx
            body += fmt_row(
                pseudo_id=new_pseudo,
                feats=feats,
                amh=float(amh_arr[ridx]),
                amm=float(amm_arr[ridx]),
                knn_margin=float(knn_margin[ridx]),
                bank_max_prob=float(bank_max_prob[ridx]),
                bank_split=split_str(ridx),
                sub_votes=subs_str(ridx),
            ) + "\n"
            new_keys.append({
                "pseudo_id": new_pseudo,
                "row_idx": ridx,
                "true_label": CLASS_STR[y[ridx]].replace("L", "Low").replace("M", "Medium").replace("H", "High"),
                "amh": float(amh_arr[ridx]),
                "knn_margin": float(knn_margin[ridx]),
                "bank_max_prob": float(bank_max_prob[ridx]),
                "bank_split": split_str(ridx),
                "sub_votes": subs_str(ridx),
                "batch": bi,
            })
        full = prompt_head + body
        out_path = OUT / f"sonnet_full_batch_{bi}.txt"
        out_path.write_text(full)
        if bi < 3 or bi == n_batches - 1:
            print(f"wrote batch {bi}: rows={len(sub)}  chars={len(full)}  ~{len(full) // 4} tokens")

    pd.DataFrame(new_keys).to_csv(OUT / "eval_keys_full.csv", index=False)
    print(f"\ntotal batches: {n_batches}")
    print(f"eval_keys_full: {len(new_keys)} rows")
    print(f"estimated cost: ~${0.165 * n_batches:.2f}")


if __name__ == "__main__":
    main()
