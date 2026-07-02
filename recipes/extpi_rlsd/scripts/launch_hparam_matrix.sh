#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${HPARAM_ENTRYPOINT:-single}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\",\"file\"]}"

case "${ENTRYPOINT}" in
  single)
    source "${SCRIPT_DIR}/_env.sh"
    TRAIN_SCRIPT="${SCRIPT_DIR}/run_extpi_rlsd.sh"
    ;;
  multi)
    source "${SCRIPT_DIR}/_env_multi.sh"
    TRAIN_SCRIPT="${SCRIPT_DIR}/run_extpi_rlsd_multi.sh"
    ;;
  *)
    echo "HPARAM_ENTRYPOINT must be single or multi; got ${ENTRYPOINT}" >&2
    exit 1
    ;;
esac

if [ -z "${RUN_ID:-}" ]; then
  echo "Set RUN_ID, for example: RUN_ID=R1-01 bash recipes/extpi_rlsd/scripts/launch_hparam_matrix.sh" >&2
  exit 1
fi

MATRIX_CSV="${MATRIX_CSV:-${EXTPI_REPO_ROOT}/experiments/extpi_rlsd/hparam_matrix.csv}"
eval "$(
  python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/hparam_matrix.py" export-env \
    --matrix "${MATRIX_CSV}" \
    --run_id "${RUN_ID}"
)"

HPARAM_RUN_DIR="${HPARAM_RUN_DIR:-${EXTPI_DATA_ROOT}/outputs/hparams/${RUN_ID}}"
mkdir -p "${HPARAM_RUN_DIR}"
export RUN_MANIFEST="${RUN_MANIFEST:-${HPARAM_RUN_DIR}/run_manifest.json}"
export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${HPARAM_RUN_DIR}/metrics.jsonl}"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/hparam_matrix.py" describe \
  --matrix "${MATRIX_CSV}" \
  --run_id "${RUN_ID}" \
  --output_json "${HPARAM_RUN_DIR}/hparams.json"

"${TRAIN_SCRIPT}" "$@"

case "${HPARAM_SKIP_PROXY_EVAL:-0}" in
  1|true|TRUE|yes|YES)
    echo "Skipping proxy eval because HPARAM_SKIP_PROXY_EVAL=${HPARAM_SKIP_PROXY_EVAL}" >&2
    exit 0
    ;;
esac

CHECKPOINT_STEP="${HPARAM_CHECKPOINT_STEP:-${SAVE_FREQ}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXTPI_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}/global_step_${CHECKPOINT_STEP}/actor/huggingface}"
if [ ! -d "${CHECKPOINT_PATH}" ]; then
  echo "Missing proxy eval checkpoint directory: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

EVAL_JSON="${HPARAM_EVAL_JSON:-${HPARAM_RUN_DIR}/proxy_eval_step${CHECKPOINT_STEP}.json}"
python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/evaluate_checkpoints.py" \
  --eval_file "${VAL_FILE}" \
  --model "${MODEL_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output_json "${EVAL_JSON}" \
  --run "${RUN_ID}" \
  --checkpoint_name "step_${CHECKPOINT_STEP}" \
  --backend "${HPARAM_EVAL_BACKEND:-hf}" \
  --num_samples "${HPARAM_EVAL_NUM_SAMPLES}" \
  --seeds "${HPARAM_EVAL_SEEDS}" \
  --max_new_tokens "${HPARAM_EVAL_MAX_NEW_TOKENS}" \
  --max_rows "${HPARAM_EVAL_MAX_ROWS}" \
  --temperature "${EVAL_TEMPERATURE:-1.0}" \
  --top_p "${EVAL_TOP_P:-1.0}" \
  --pi_trace_field "${PI_TRACE_FIELD}"

summary_args=(
  --run_id "${RUN_ID}"
  --metrics_jsonl "${VERL_FILE_LOGGER_PATH}"
  --eval_json "${EVAL_JSON}"
  --output_json "${HPARAM_RUN_DIR}/summary.json"
  --output_csv "${HPARAM_SUMMARY_CSV:-${EXTPI_DATA_ROOT}/outputs/hparams/summary.csv}"
)
if [ -n "${HPARAM_BASELINE_EVAL_JSON:-}" ]; then
  summary_args+=(--baseline_eval_json "${HPARAM_BASELINE_EVAL_JSON}")
fi
python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/hparam_matrix.py" summarize "${summary_args[@]}"
