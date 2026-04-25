#!/usr/bin/env bash
# bootstrap.sh — re-hydrate the container after a restart.
# Installs deps and downloads:
#   - data/{train,test,sample_submission}.csv  (Playground Series S6E4)
#   - data/archive.zip                          (l3llff/irrigation-water, the
#                                                10k original dataset used by
#                                                recipe's ORIG mean/std FE)
#
# Reads Kaggle credentials from KAGGLE_API_TOKEN (KGAT_...) or prompts.

set -euo pipefail
cd "$(dirname "$0")"

echo "--- installing requirements ---"
pip install -q -r requirements.txt

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
    if [[ ! -f data/train.csv || ! -f data/archive.zip ]]; then
        read -rsp "Kaggle API token (KGAT_...): " KAGGLE_API_TOKEN
        echo
        export KAGGLE_API_TOKEN
    fi
fi

if [[ ! -f data/train.csv || ! -f data/test.csv ]]; then
    echo "--- downloading playground-series-s6e4 ---"
    kaggle competitions download -c playground-series-s6e4 -p data/
    unzip -qo data/playground-series-s6e4.zip -d data/
    rm -f data/playground-series-s6e4.zip
else
    echo "--- competition data already present ---"
fi

if [[ ! -f data/archive.zip ]]; then
    echo "--- downloading l3llff/irrigation-water (original 10k) ---"
    # Accept the dataset's ToS once at https://www.kaggle.com/datasets/l3llff/irrigation-water
    # before first run; without it the API returns 403.
    kaggle datasets download -d l3llff/irrigation-water -p data/
    # The dataset zip lands as data/irrigation-water.zip; recipe expects archive.zip.
    mv data/irrigation-water.zip data/archive.zip
else
    echo "--- archive.zip already present ---"
fi

echo "--- done ---"
ls -lh data/
