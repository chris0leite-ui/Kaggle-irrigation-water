#!/usr/bin/env python3
"""Post-Kaggle handler for rawashishsin-replica.

Pulls Kaggle artifacts -> moves to scripts/artifacts/ -> runs 4-gate filter.

Usage:
  python scripts/rawashishsin_post.py [v2|v3]   (default: v2)
"""
import os, sys, json, shutil, subprocess
from pathlib import Path

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v2"
KERNEL_ID = {
    "v2": "chrisleitescha/irrigation-rawashishsin-replica",
    "v3": "chrisleitescha/irrigation-rawashishsin-2600",
}[VARIANT]

OUT_LOCAL = Path(f"kaggle_kernel/output_rawashishsin_{VARIANT}")
OUT_LOCAL.mkdir(parents=True, exist_ok=True)
ART = Path("scripts/artifacts")

# 1. Pull
print(f"[pull] {KERNEL_ID} -> {OUT_LOCAL}")
env = os.environ.copy()
subprocess.run(
    ["kaggle", "kernels", "output", KERNEL_ID, "-p", str(OUT_LOCAL)],
    env=env, check=True
)

# 2. Move to artifacts (canonical names)
oof_src = OUT_LOCAL / "oof_rawashishsin.npy"
test_src = OUT_LOCAL / "test_rawashishsin.npy"
results_src = OUT_LOCAL / "rawashishsin_results.json"

cand_name = "rawashishsin" if VARIANT == "v2" else "rawashishsin_2600"
oof_dst = ART / f"oof_{cand_name}.npy"
test_dst = ART / f"test_{cand_name}.npy"
results_dst = ART / f"{cand_name}_results.json"

assert oof_src.exists(), f"missing {oof_src}; check kernel completed cleanly"
assert test_src.exists(), f"missing {test_src}"

shutil.copy(oof_src, oof_dst)
shutil.copy(test_src, test_dst)
if results_src.exists():
    shutil.copy(results_src, results_dst)
print(f"[copy] -> {oof_dst}")
print(f"[copy] -> {test_dst}")

# 3. Print summary
if results_dst.exists():
    res = json.load(open(results_dst))
    print("\n=== KERNEL RESULTS ===")
    for k, v in res.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {k}: {v[:3]}... (len {len(v)})")
        else:
            print(f"  {k}: {v}")

# 4. Run 4-gate filter
print(f"\n=== 4-GATE FILTER on {cand_name} ===")
subprocess.run(
    ["python3", "scripts/blend_gate_4gate.py",
     "--candidate", cand_name,
     "--use-iso"],
    check=True
)
print()
print("Done.")
