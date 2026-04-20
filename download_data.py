"""
Download the ORIGINAL Irrigation Prediction dataset from Kaggle.

NOT REQUIRED for the baseline. This fetches the standalone
`l3llff/irrigation-water` dataset — the real-world data that the
Playground Series S6E4 synthetic train/test were generated from. It's
useful only as an optional "extra training data" experiment (the
competition rules explicitly allow incorporating the original).

For the competition data itself (train.csv, test.csv, sample_submission.csv),
run ./bootstrap.sh instead — that uses `kaggle competitions download -c
playground-series-s6e4` which authenticates with KAGGLE_API_TOKEN alone.

One-time setup before first run of THIS script:
  1. Visit https://www.kaggle.com/datasets/l3llff/irrigation-water in a
     browser while signed in, click Download, and accept the dataset's
     terms of service. Without this step the API returns 403.
  2. Provide credentials via either:
     - `.env` file with KAGGLE_USERNAME + KAGGLE_KEY (legacy format), OR
     - `~/.kaggle/kaggle.json` with {"username": ..., "key": ...}.
     Note: a bare KAGGLE_API_TOKEN (new KGAT_ format) is not sufficient
     for the `kaggle datasets` endpoint — the python client needs a
     username/key pair written into kaggle.json.

Usage:
  python download_data.py
"""

import os
import json
import pathlib
from dotenv import load_dotenv

load_dotenv()


def setup_kaggle_credentials():
    username = os.getenv("KAGGLE_USERNAME")
    # Support both KAGGLE_KEY and the newer KAGGLE_API_TOKEN env var name
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")

    if username and key:
        kaggle_dir = pathlib.Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        creds_path = kaggle_dir / "kaggle.json"
        creds_path.write_text(json.dumps({"username": username, "key": key}))
        creds_path.chmod(0o600)
        print(f"Kaggle credentials written to {creds_path}")
    elif not (pathlib.Path.home() / ".kaggle" / "kaggle.json").exists():
        raise EnvironmentError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "(or KAGGLE_API_TOKEN) in .env, or place kaggle.json in ~/.kaggle/kaggle.json"
        )


def download_dataset(dataset: str, output_dir: str = "data"):
    import kaggle  # imported after credentials are set up

    pathlib.Path(output_dir).mkdir(exist_ok=True)
    print(f"Downloading dataset '{dataset}' to '{output_dir}/'...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print("Download complete.")


if __name__ == "__main__":
    setup_kaggle_credentials()

    # The original Irrigation Prediction dataset — the real-world data that
    # the Playground Series S6E4 synthetic data was generated from. Remember
    # to accept the dataset's terms of service once in a browser before the
    # API will serve it (see module docstring).
    DATASET = "l3llff/irrigation-water"

    download_dataset(DATASET)
