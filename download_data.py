"""
Download the irrigation water dataset from Kaggle.

Setup:
  1. Copy .env.example to .env and fill in your Kaggle credentials, OR
  2. Place your kaggle.json in ~/.kaggle/kaggle.json

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

    # Update this to the target Kaggle dataset slug (owner/dataset-name)
    DATASET = "l3llff/irrigation-water"

    download_dataset(DATASET)
