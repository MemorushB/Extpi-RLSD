#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

EVAL_FILE="${EVAL_FILE:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/matched_dev.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_NAME="${RUN_NAME:-$(basename "${CHECKPOINT_PATH:-${MODEL_PATH}}")}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-${RUN_NAME}}"
OUTPUT_JSON="${OUTPUT_JSON:-${EXTPI_DATA_ROOT}/outputs/eval/${RUN_NAME}/${CHECKPOINT_NAME}.json}"

args=(
  --eval_file "${EVAL_FILE}"
  --model "${MODEL_PATH}"
  --output_json "${OUTPUT_JSON}"
  --run "${RUN_NAME}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --backend "${EVAL_BACKEND:-hf}"
  --num_samples "${NUM_SAMPLES:-12}"
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}"
  --temperature "${EVAL_TEMPERATURE:-1.0}"
  --top_p "${EVAL_TOP_P:-1.0}"
)
if [ -n "${CHECKPOINT_PATH}" ]; then
  args+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [ -n "${ADAPTER_PATH}" ]; then
  args+=(--adapter_path "${ADAPTER_PATH}")
fi
if [ -n "${MAX_ROWS:-}" ]; then
  args+=(--max_rows "${MAX_ROWS}")
fi

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/evaluate_checkpoints.py" "${args[@]}" "$@"
