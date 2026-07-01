#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:-${EXTPI_DATA_ROOT}/datasets/closed_sft/train.parquet}"
VAL_FILE="${VAL_FILE:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
PROJECT_NAME="${PROJECT_NAME:-extpi_rlsd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-closed_sft_qwen3_1p7b_lora_multi}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${EXTPI_DATA_ROOT}/datasets/closed_sft/manifest.json}"
SCALE_MODE="${SCALE_MODE:-fixed}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-${SFT_MICRO_BATCH_SIZE:-1}}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"
SFT_LR="${SFT_LR:-5e-6}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-8192}"
SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-${MAX_SEQUENCE_LENGTH}}"
SFT_LORA_RANK="${SFT_LORA_RANK:-32}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-64}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
SFT_PREFLIGHT="${SFT_PREFLIGHT:-1}"

if [ "${SCALE_MODE}" = "linear" ]; then
  DEFAULT_SFT_TRAIN_BATCH_SIZE=$((8 * NGPUS_PER_NODE * NNODES))
elif [ "${SCALE_MODE}" = "fixed" ]; then
  DEFAULT_SFT_TRAIN_BATCH_SIZE=8
else
  echo "SCALE_MODE must be fixed or linear; got ${SCALE_MODE}" >&2
  exit 1
fi
SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-${DEFAULT_SFT_TRAIN_BATCH_SIZE}}"
WORLD_SIZE=$((NGPUS_PER_NODE * NNODES))
if [ $((SFT_TRAIN_BATCH_SIZE % WORLD_SIZE)) -ne 0 ]; then
  echo "SFT_TRAIN_BATCH_SIZE must be divisible by WORLD_SIZE=${WORLD_SIZE}; got ${SFT_TRAIN_BATCH_SIZE}" >&2
  exit 1
fi

val_args=(data.val_files=null)
if [ -n "${VAL_FILE}" ]; then
  val_args=(data.val_files="${VAL_FILE}")
fi

if [ "${SFT_PREFLIGHT}" != "0" ]; then
  python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/inspect_sft_parquet.py" \
    --parquet "${TRAIN_FILE}" \
    --model "${MODEL_PATH}" \
    --max_length "${MAX_SEQUENCE_LENGTH}" \
    --enable_thinking_default false \
    --fail_on_overlong
fi

write_extpi_run_manifest "${EXPERIMENT_NAME}" \
  --model "student=${MODEL_PATH}" \
  --config_kv "method=closed_sft" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "scale_mode=${SCALE_MODE}" \
  --config_kv "sft_train_batch_size=${SFT_TRAIN_BATCH_SIZE}" \
  --config_kv "micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE_PER_GPU}" \
  --config_kv "sft_epochs=${SFT_EPOCHS}" \
  --config_kv "max_sequence_length=${MAX_SEQUENCE_LENGTH}" \
  --config_kv "lora_rank=${SFT_LORA_RANK}" \
  --config_kv "lora_alpha=${SFT_LORA_ALPHA}" \
  --config_kv "multi_gpu=true"

torchrun --standalone \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  "${val_args[@]}" \
  data.messages_key=messages \
  data.enable_thinking_default=False \
  data.train_batch_size="${SFT_TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_length="${MAX_SEQUENCE_LENGTH}" \
  data.max_token_len_per_gpu="${SFT_MAX_TOKEN_LEN_PER_GPU}" \
  data.truncation=error \
  data.num_workers="${SFT_NUM_WORKERS:-0}" \
  optim.lr="${SFT_LR}" \
  model.path="${MODEL_PATH}" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank="${SFT_LORA_RANK}" \
  model.lora_alpha="${SFT_LORA_ALPHA}" \
  trainer.default_local_dir="${EXTPI_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.total_epochs="${SFT_EPOCHS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
