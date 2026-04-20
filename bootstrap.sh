#!/usr/bin/env bash
# bootstrap.sh — re-hydrate the container after a restart.
# Prompts once for your Kaggle API token (KGAT_...), then installs deps and
# downloads the competition data. Nothing is written to chat or logs.

set -euo pipefail
cd "$(dirname "$0")"

echo "--- installing requirements ---"
pip install -q -r requirements.txt

if [[ -f data/train.csv && -f data/test.csv ]]; then
    echo "--- data already present, skipping download ---"
    ls -lh data/*.csv
    exit 0
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    read -rsp "Kaggle API token (KGAT_...): " KAGGLE_API_TOKEN
    echo
    export KAGGLE_API_TOKEN
fi

echo "--- downloading playground-series-s6e4 ---"
kaggle competitions download -c playground-series-s6e4 -p data/
unzip -qo data/playground-series-s6e4.zip -d data/
rm -f data/playground-series-s6e4.zip

echo "--- done ---"
ls -lh data/*.csv
