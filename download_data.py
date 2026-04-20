"""
Download the ORIGINAL Irrigation Prediction dataset from Kaggle.

NOT REQUIRED for the baseline. This fetches the standalone
`l3llff/irrigation-water` dataset — the real-world data that the
Playground Series S6E4 synthetic train/test were generated from. It's
useful only as an optional "extra training data" experiment (the
competition rules explicitly allow incorporating the original).

For the competition data itself (train.csv, test.csv, sample_submission.csv),
run ./bootstrap.sh instead.

One-time setup before first run:
  Visit https://www.kaggle.com/datasets/l3llff/irrigation-water in a
  browser while signed in, click Download, and accept the dataset's
  terms of service. Without this step the API returns 403.

Credential modes (pick one):
  A. KAGGLE_API_TOKEN env var (new KGAT_... format, preferred)
  B. KAGGLE_USERNAME + KAGGLE_KEY env vars (legacy API key)
  C. Existing ~/.kaggle/kaggle.json

Usage:
  python download_data.py
"""

import os
import json
import pathlib
from dotenv import load_dotenv

load_dotenv()

DATASET = "l3llff/irrigation-water"


def _is_kgat_token(value: str) -> bool:
    return value.startswith("KGAT_")


def setup_kaggle_credentials():
    username = os.getenv("KAGGLE_USERNAME")
    api_token = os.getenv("KAGGLE_API_TOKEN")
    legacy_key = os.getenv("KAGGLE_KEY")

    kaggle_dir = pathlib.Path.home() / ".kaggle"
    creds_path = kaggle_dir / "kaggle.json"

    if api_token and _is_kgat_token(api_token):
        # KGAT tokens authenticate via the KAGGLE_API_TOKEN env var; the
        # Kaggle SDK reads it directly — no kaggle.json needed.
        print("Using KGAT access token from KAGGLE_API_TOKEN.")
        if creds_path.exists():
            # Remove any stale legacy kaggle.json so it doesn't conflict.
            creds_path.unlink()
        return

    if username and (legacy_key or api_token):
        key = legacy_key or api_token
        kaggle_dir.mkdir(exist_ok=True)
        creds_path.write_text(json.dumps({"username": username, "key": key}))
        creds_path.chmod(0o600)
        print(f"Kaggle credentials written to {creds_path}")
        return

    if creds_path.exists():
        print(f"Using existing credentials at {creds_path}")
        return

    raise EnvironmentError(
        "Kaggle credentials not found.\n"
        "  Option A: set KAGGLE_API_TOKEN to a KGAT_... token\n"
        "  Option B: set KAGGLE_USERNAME + KAGGLE_KEY\n"
        "  Option C: place kaggle.json in ~/.kaggle/kaggle.json"
    )


def download_dataset(dataset: str = DATASET, output_dir: str = "data"):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        raise SystemExit("Run: pip install kaggle")

    pathlib.Path(output_dir).mkdir(exist_ok=True)
    print(f"Downloading '{dataset}' to '{output_dir}/'...")

    api = KaggleApiExtended()
    try:
        api.authenticate()
    except Exception as e:
        raise SystemExit(
            f"Authentication failed: {e}\n"
            "If using a KGAT token, ensure outbound HTTPS access to kaggle.com is available.\n"
            "If the error is 403: accept the dataset's terms at "
            "https://www.kaggle.com/datasets/l3llff/irrigation-water"
        )

    api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print("Download complete.")


if __name__ == "__main__":
    setup_kaggle_credentials()
    download_dataset()
