#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean_closed_sft.jsonl}"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/generate_closed_teacher_data.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  "$@"
