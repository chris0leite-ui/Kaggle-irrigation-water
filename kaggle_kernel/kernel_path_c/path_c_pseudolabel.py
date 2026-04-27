"""Path C — pseudo-label retraining with LB-best 4-stack as labeler.

Wraps scripts/recipe_pseudolabel.py (uploaded as a Kaggle dataset) and
runs it with LABELER_TEST_PATH / LABELER_BIAS_JSON pointing at the LB-best
4-stack reconstruction (OOF 0.98084 / LB 0.98094, also bundled in the
scripts dataset).

Outputs (in /kaggle/working/):
  oof_recipe_pseudolabel_path_c_stage1.npy
  test_recipe_pseudolabel_path_c_stage1.npy
  recipe_pseudolabel_path_c_stage1_results.json
"""
from __future__ import annotations
import os
import shutil
import sys
import time
import subprocess
from pathlib import Path

print(f"[boot] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

# Locate the scripts dataset.
KAGGLE_INPUT = Path("/kaggle/input")
SCRIPTS_DIR = None
for p in KAGGLE_INPUT.rglob("recipe_pseudolabel.py"):
    SCRIPTS_DIR = p.parent
    break
if SCRIPTS_DIR is None:
    raise FileNotFoundError("could not locate scripts dataset")
print(f"[boot] scripts at {SCRIPTS_DIR}", flush=True)

# Set up a workdir mirroring the local repo layout (recipe_pseudolabel.py
# expects scripts/ + data/ + scripts/artifacts/ relative to cwd).
WORK = Path("/tmp/path_c_workdir")
WORK.mkdir(exist_ok=True)
(WORK / "scripts").mkdir(exist_ok=True)
(WORK / "scripts/artifacts").mkdir(exist_ok=True)
(WORK / "data").mkdir(exist_ok=True)

# Copy scripts.
for f in ("common.py", "recipe_features.py", "recipe_ote.py",
          "recipe_full_te.py", "recipe_pseudolabel.py"):
    shutil.copy2(SCRIPTS_DIR / f, WORK / "scripts" / f)
# Copy labeler artifacts.
shutil.copy2(SCRIPTS_DIR / "test_path_c_primary_labeler.npy",
             WORK / "scripts/artifacts" / "test_path_c_primary_labeler.npy")
shutil.copy2(SCRIPTS_DIR / "path_c_primary_labeler_results.json",
             WORK / "scripts/artifacts" / "path_c_primary_labeler_results.json")

# Symlink data files from competition input.
for fname in ("train.csv", "test.csv", "sample_submission.csv"):
    for p in KAGGLE_INPUT.rglob(fname):
        target = WORK / "data" / fname
        if target.exists():
            target.unlink()
        target.symlink_to(p.resolve())
        print(f"[boot] linked data/{fname} -> {p}", flush=True)
        break
# original 10k irrigation dataset (used by recipe_full_te FE pipeline).
ORIG = None
for cand in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
             "irrigation-prediction.csv"):
    for p in KAGGLE_INPUT.rglob(cand):
        ORIG = p
        break
    if ORIG:
        break
if ORIG is None:
    # any non-train/test csv as fallback
    for p in KAGGLE_INPUT.rglob("*.csv"):
        if p.name not in ("train.csv", "test.csv", "sample_submission.csv"):
            ORIG = p
            break
if ORIG is None:
    raise FileNotFoundError("no original-dataset CSV found")
# recipe_full_te.py reads data/archive.zip as a ZIP archive containing the
# original CSV. Build that zip from the orig CSV.
import zipfile
target = WORK / "data" / "archive.zip"
if target.exists():
    target.unlink()
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ORIG, arcname=ORIG.name)
print(f"[boot] zipped {ORIG.name} into data/archive.zip", flush=True)

# Run recipe_pseudolabel.py with LB-best 4-stack as labeler.
env = os.environ.copy()
env["LABELER_TEST_PATH"] = "scripts/artifacts/test_path_c_primary_labeler.npy"
env["LABELER_BIAS_JSON"] = "scripts/artifacts/path_c_primary_labeler_results.json"
env["PSEUDO_TAU"] = "0.99"
env["PSEUDO_SUFFIX"] = "path_c_stage1"

print(f"[boot] launching recipe_pseudolabel.py with env:", flush=True)
for k in ("LABELER_TEST_PATH", "LABELER_BIAS_JSON", "PSEUDO_TAU", "PSEUDO_SUFFIX"):
    print(f"  {k}={env[k]}", flush=True)

t0 = time.time()
ret = subprocess.run(
    [sys.executable, "scripts/recipe_pseudolabel.py"],
    cwd=WORK, env=env, check=False,
)
print(f"[boot] recipe_pseudolabel.py exit {ret.returncode} after {time.time()-t0:.1f}s", flush=True)

# Copy outputs from cwd-relative paths to /kaggle/working/.
OUT = Path("/kaggle/working")
for f in (
    "scripts/artifacts/oof_recipe_pseudolabel_path_c_stage1.npy",
    "scripts/artifacts/test_recipe_pseudolabel_path_c_stage1.npy",
    "scripts/artifacts/recipe_pseudolabel_path_c_stage1_results.json",
):
    src = WORK / f
    if src.exists():
        dst = OUT / src.name
        shutil.copy2(src, dst)
        print(f"[boot] copied {src.name} -> {dst}", flush=True)
    else:
        print(f"[boot] WARN: missing {src}", flush=True)

print(f"[boot] done at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
