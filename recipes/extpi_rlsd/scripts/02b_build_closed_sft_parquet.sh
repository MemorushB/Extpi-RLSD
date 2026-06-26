#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean_closed_sft.jsonl}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${EXTPI_DATA_ROOT}/datasets/closed_sft/train.parquet}"
MANIFEST_JSON="${MANIFEST_JSON:-${EXTPI_DATA_ROOT}/datasets/closed_sft/manifest.json}"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/build_closed_sft_parquet.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_parquet "${OUTPUT_PARQUET}" \
  --manifest_json "${MANIFEST_JSON}" \
  "$@"
