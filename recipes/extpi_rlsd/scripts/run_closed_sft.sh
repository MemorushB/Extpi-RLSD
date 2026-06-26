#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

TRAIN_FILE="${TRAIN_FILE:-${EXTPI_DATA_ROOT}/datasets/closed_sft/train.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
PROJECT_NAME="${PROJECT_NAME:-extpi_rlsd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-closed_sft_qwen3_1p7b_lora}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${EXTPI_DATA_ROOT}/datasets/closed_sft/manifest.json}"

write_extpi_run_manifest "${EXPERIMENT_NAME}" \
  --model "student=${MODEL_PATH}" \
  --config_kv "method=closed_sft" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "train_batch_size=${SFT_TRAIN_BATCH_SIZE:-32}" \
  --config_kv "micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE:-1}" \
  --config_kv "total_epochs=${SFT_EPOCHS:-4}"

python3 -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.messages_key=messages \
  data.enable_thinking_default=False \
  data.train_batch_size="${SFT_TRAIN_BATCH_SIZE:-32}" \
  data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-1}" \
  data.max_length="${MAX_SEQUENCE_LENGTH:-4096}" \
  optim.lr="${SFT_LR:-5e-6}" \
  model.partial_pretrain="${MODEL_PATH}" \
  model.lora_rank=32 \
  model.lora_alpha=64 \
  trainer.default_local_dir="${EXTPI_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.total_epochs="${SFT_EPOCHS:-4}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
