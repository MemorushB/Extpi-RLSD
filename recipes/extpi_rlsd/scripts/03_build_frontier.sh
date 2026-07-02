#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean_qwen32b_pi_screened.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp}"
PI_TRACE_FIELD="${PI_TRACE_FIELD:-qwen32b_pi_trace}"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/build_frontier.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --pi_trace_field "${PI_TRACE_FIELD}" \
  "$@"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/jsonl_to_verl_parquet.py" \
  --input_jsonl "${OUTPUT_DIR}/mvp_train.jsonl" \
  --output_parquet "${OUTPUT_DIR}/mvp_train.parquet" \
  --pi_trace_field "${PI_TRACE_FIELD}"
python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/jsonl_to_verl_parquet.py" \
  --input_jsonl "${OUTPUT_DIR}/matched_dev.jsonl" \
  --output_parquet "${OUTPUT_DIR}/matched_dev.parquet" \
  --pi_trace_field "${PI_TRACE_FIELD}"
