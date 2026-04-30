"""T1 — apply 4-axis decision rule and emit override candidate CSV.

Inputs:
  scripts/artifacts/T1_responses_batch_<i>.txt   — raw haiku replies
  submissions/submission_idea4b_selective_override.csv
  scripts/artifacts/test_<bank_component>.npy    — 14 test arrays

Decision rule (per kickoff doc + prompts/subagent_llm_judge.md):
  override 4b on row r ONLY IF all of:
    (1) llm_final[r] != 4b_argmax[r]            — LLM disagrees with 4b
    (2) llm_conf[r]  >= 0.7                     — LLM is confident
    (3) bank_argmax[r] == llm_final[r]          — 14-bank-majority agrees
    (4) (4b_argmax[r] == High) and (llm_final[r] == Medium)  — H->M only

Outputs:
  submissions/submission_T1_llm_judge_override.csv
  scripts/artifacts/T1_compose_override_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T1_parse_responses import parse_response_text  # noqa: E402
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")

CLASS_INT = {"Low": 0, "Medium": 1, "High": 2}
CLASS_STR = {0: "Low", 1: "Medium", 2: "High"}


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLASS_INT).to_numpy(dtype=np.int8)


def collect_responses() -> pd.DataFrame:
    """Concatenate all T1_responses_batch_*.txt parses."""
    frames = []
    for p in sorted(ART.glob("T1_responses_batch_*.txt")):
        text = p.read_text()
        df = parse_response_text(text)
        df["batch"] = p.stem
        frames.append(df)
    if not frames:
        raise SystemExit("no T1_responses_batch_*.txt files found")
    return pd.concat(frames, ignore_index=True)


def main():
    print("=== T1 compose override ===\n")
    llm = collect_responses()
    print(f"parsed {len(llm)} LLM responses across {llm['batch'].nunique()} batches")
    llm = llm.drop_duplicates(subset=["test_id"], keep="first")
    print(f"unique test_id rows: {len(llm)}")
    print(f"FINAL distribution: {llm['llm_final'].value_counts().to_dict()}")
    print(f"CONF stats: mean={llm['llm_conf'].mean():.3f} median={llm['llm_conf'].median():.3f}")
    print(f"CONF >= 0.7: {int((llm['llm_conf'] >= 0.7).sum())}")

    # 4b argmax (test-side)
    fb = csv_argmax("submission_idea4b_selective_override")
    print(f"\n4b argmax: L={(fb==0).sum()} M={(fb==1).sum()} H={(fb==2).sum()}")

    # 14-bank mean probs + argmax (test-side)
    bank = load_bank("test")  # (14, 270k, 3)
    bank_mean = bank_mean_probs(bank)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)

    # Map LLM rows by test_id -> row index
    test = pd.read_csv(DATA / "test.csv")
    id_to_idx = {int(tid): i for i, tid in enumerate(test["id"].values)}
    llm["row_idx"] = llm["test_id"].astype(int).map(id_to_idx)
    llm = llm.dropna(subset=["row_idx"])
    llm["row_idx"] = llm["row_idx"].astype(int)

    # Vectors aligned to llm rows
    idx = llm["row_idx"].to_numpy()
    fb_vec = fb[idx]
    bank_vec = bank_argmax[idx]
    llm_final_int = llm["llm_final"].map(CLASS_INT).to_numpy(dtype=np.int8)
    llm_conf = llm["llm_conf"].to_numpy(dtype=np.float32)

    cond1 = llm_final_int != fb_vec                 # LLM disagrees with 4b
    cond2 = llm_conf >= 0.7
    cond3 = bank_vec == llm_final_int               # bank-maj agrees with LLM
    cond4 = (fb_vec == 2) & (llm_final_int == 1)    # H -> M only

    print(f"\naxis (1) llm != 4b:           {int(cond1.sum())}")
    print(f"axis (2) llm_conf >= 0.7:     {int(cond2.sum())}")
    print(f"axis (3) bank_maj == llm:     {int(cond3.sum())}")
    print(f"axis (4) 4b=H and llm=M:      {int(cond4.sum())}")

    fire = cond1 & cond2 & cond3 & cond4
    print(f"\nALL 4 axes fire: {int(fire.sum())} rows")

    # Diagnostic counts of weaker filters
    print(f"  axes 1+2+3 (any direction):        {int((cond1 & cond2 & cond3).sum())}")
    print(f"  axes 1+2+4 (skip bank-maj):        {int((cond1 & cond2 & cond4).sum())}")
    print(f"  axes 1+3+4 (skip CONF):            {int((cond1 & cond3 & cond4).sum())}")

    # Apply override: copy 4b argmax, flip on `fire`
    new_pred = fb.copy()
    new_pred[idx[fire]] = 1  # flip to Medium

    n_changed = int((new_pred != fb).sum())
    print(f"\ntotal flips applied: {n_changed}")
    print(f"new_pred distribution: L={(new_pred==0).sum()} M={(new_pred==1).sum()} H={(new_pred==2).sum()}")

    # Direction sanity vs 4b
    h_added = int(((fb != 2) & (new_pred == 2)).sum())
    h_removed = int(((fb == 2) & (new_pred != 2)).sum())
    print(f"net_H vs 4b: +{h_added} -{h_removed} = {h_added - h_removed:+d}")

    # Emit candidate CSV
    out_csv = SUB / "submission_T1_llm_judge_override.csv"
    sub = pd.DataFrame({
        "id": test["id"].values,
        "Irrigation_Need": pd.Series(new_pred).map(CLASS_STR),
    })
    sub.to_csv(out_csv, index=False)
    print(f"\nemitted: {out_csv}")

    # Save firing details for downstream TRAIN OOF validation
    fire_df = llm[fire].copy()
    fire_df["row_idx"] = idx[fire]
    fire_df.to_csv(ART / "T1_fire_details.csv", index=False)
    print(f"saved: {ART / 'T1_fire_details.csv'}")

    # Results JSON
    out = ART / "T1_compose_override_results.json"
    out.write_text(json.dumps({
        "n_llm_rows": int(len(llm)),
        "axis_counts": {
            "axis_1_llm_neq_4b": int(cond1.sum()),
            "axis_2_conf_gte_07": int(cond2.sum()),
            "axis_3_bank_eq_llm": int(cond3.sum()),
            "axis_4_h_to_m": int(cond4.sum()),
        },
        "n_fire_all_4_axes": int(fire.sum()),
        "n_flips_applied": n_changed,
        "net_H_vs_4b": h_added - h_removed,
        "new_pred_dist": {
            "Low": int((new_pred == 0).sum()),
            "Medium": int((new_pred == 1).sum()),
            "High": int((new_pred == 2).sum()),
        },
        "candidate_csv": str(out_csv),
        "llm_conf_stats": {
            "mean": float(llm["llm_conf"].mean()),
            "median": float(llm["llm_conf"].median()),
            "n_ge_07": int((llm["llm_conf"] >= 0.7).sum()),
        },
    }, indent=2))
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
