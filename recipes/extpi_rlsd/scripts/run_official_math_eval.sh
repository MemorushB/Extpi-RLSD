#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

EVAL_DATA_DIR="${EVAL_DATA_DIR:-${EXTPI_DATA_ROOT}/datasets/eval/official_math}"
DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_NAME="${RUN_NAME:-$(basename "${CHECKPOINT_PATH:-${MODEL_PATH}}")}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXTPI_DATA_ROOT}/outputs/eval_official/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

outputs=()
for dataset in ${DATASETS}; do
  eval_file="${EVAL_DATA_DIR}/${dataset}.parquet"
  if [ ! -f "${eval_file}" ]; then
    echo "Missing eval file: ${eval_file}" >&2
    echo "Run recipes/extpi_rlsd/scripts/04_prepare_official_eval_data.sh first." >&2
    exit 1
  fi

  output_json="${OUTPUT_ROOT}/${dataset}_${CHECKPOINT_NAME}.json"
  log_file="${LOG_DIR}/${dataset}_${CHECKPOINT_NAME}.log"
  args=(
    --eval_file "${eval_file}"
    --dataset_name "${dataset}"
    --model "${MODEL_PATH}"
    --output_json "${output_json}"
    --run "${RUN_NAME}"
    --checkpoint_name "${CHECKPOINT_NAME}"
    --backend "${EVAL_BACKEND:-hf}"
    --num_samples "${NUM_SAMPLES:-12}"
    --seeds "${EVAL_SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11}"
    --max_new_tokens "${MAX_NEW_TOKENS:-38912}"
    --temperature "${EVAL_TEMPERATURE:-1.0}"
    --top_p "${EVAL_TOP_P:-1.0}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
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
  case "${EVAL_ENABLE_THINKING:-0}" in
    1|true|TRUE|yes|YES)
      args+=(--enable_thinking)
      ;;
  esac

  echo "Running official eval: dataset=${dataset}, output=${output_json}" >&2
  (
    set -o pipefail
    python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/evaluate_checkpoints.py" "${args[@]}" "$@" 2>&1 | tee "${log_file}"
  )
  outputs+=("${output_json}")
done

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/aggregate_results.py" \
  --input "${outputs[@]}" \
  --output_csv "${OUTPUT_ROOT}/summary.csv" \
  --output_md "${OUTPUT_ROOT}/summary.md"

echo "Wrote ${OUTPUT_ROOT}/summary.csv" >&2
echo "Wrote ${OUTPUT_ROOT}/summary.md" >&2
