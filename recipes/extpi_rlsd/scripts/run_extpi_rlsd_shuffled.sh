#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train_shuffled_pi.jsonl}"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/build_shuffled_pi.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  "$@"

"${SCRIPT_DIR}/run_extpi_rlsd.sh"
