#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

TRAIN_FILE="${TRAIN_FILE:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train.parquet}"
VAL_FILE="${VAL_FILE:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/matched_dev.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
PROJECT_NAME="${PROJECT_NAME:-extpi_rlsd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_1p7b_lora}"
METHOD_NAME="${METHOD_NAME:-grpo}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/split_manifest.json}"

manifest_model_args=(--model "student=${MODEL_PATH}")
if [ -n "${TEACHER_MODEL:-}" ]; then
  manifest_model_args+=(--model "teacher=${TEACHER_MODEL}")
fi

write_extpi_run_manifest "${EXPERIMENT_NAME}" \
  "${manifest_model_args[@]}" \
  --config_kv "method=${METHOD_NAME}" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "total_training_steps=${TOTAL_TRAINING_STEPS:-5}" \
  --config_kv "rollout_n=${ROLLOUT_N:-4}"

python3 -m verl.trainer.main_ppo \
  trainer.use_v1=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  custom_reward_function.path=recipes/extpi_rlsd/rewards/math_verify_reward.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-3e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.default_local_dir="${EXTPI_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.logger='["console"]' \
  "$@"
