#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train_shuffled_pi.jsonl}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train_shuffled_pi.parquet}"
PI_TRACE_FIELD="${PI_TRACE_FIELD:-qwen32b_pi_trace}"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/build_shuffled_pi.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --seed "${SHUFFLE_SEED:-42}" \
  --pi_trace_field "${PI_TRACE_FIELD}"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/jsonl_to_verl_parquet.py" \
  --input_jsonl "${OUTPUT_JSONL}" \
  --output_parquet "${OUTPUT_PARQUET}" \
  --pi_trace_field "${PI_TRACE_FIELD}"

TRAIN_FILE="${OUTPUT_PARQUET}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-extpi_rlsd_shuffled_qwen3_1p7b_lora}" \
"${SCRIPT_DIR}/run_extpi_rlsd.sh" "$@"
