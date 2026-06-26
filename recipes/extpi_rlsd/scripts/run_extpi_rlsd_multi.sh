#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TRAIN_FILE="${TRAIN_FILE:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/mvp_train.parquet}"
VAL_FILE="${VAL_FILE:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/matched_dev.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
PROJECT_NAME="${PROJECT_NAME:-extpi_rlsd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-extpi_rlsd_qwen3_1p7b_lora_multi}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${EXTPI_DATA_ROOT}/datasets/splits/frontier_mvp/split_manifest.json}"
RLSD_LAMBDA="${RLSD_LAMBDA:-0.5}"
RLSD_CLIP_RANGE="${RLSD_CLIP_RANGE:-0.2}"
TEACHER_MAX_PROMPT_LENGTH="${TEACHER_MAX_PROMPT_LENGTH:-2048}"
TEACHER_UPDATE_MODE="${TEACHER_UPDATE_MODE:-base_no_adapter}"
SCALE_MODE="${SCALE_MODE:-fixed}"

if [ "${TEACHER_UPDATE_MODE}" != "base_no_adapter" ]; then
  echo "Multi-GPU ExtPI MVP currently supports base_no_adapter only." >&2
  exit 1
fi
if [ "${SCALE_MODE}" = "linear" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=$((8 * NGPUS_PER_NODE * NNODES))
elif [ "${SCALE_MODE}" = "fixed" ]; then
  DEFAULT_TRAIN_BATCH_SIZE=8
else
  echo "SCALE_MODE must be fixed or linear; got ${SCALE_MODE}" >&2
  exit 1
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"

write_extpi_run_manifest "${EXPERIMENT_NAME}" \
  --model "student=${MODEL_PATH}" \
  --config_kv "method=extpi_rlsd_multi" \
  --config_kv "model_path=${MODEL_PATH}" \
  --config_kv "train_file=${TRAIN_FILE}" \
  --config_kv "val_file=${VAL_FILE}" \
  --config_kv "scale_mode=${SCALE_MODE}" \
  --config_kv "train_batch_size=${TRAIN_BATCH_SIZE}" \
  --config_kv "rlsd_lambda=${RLSD_LAMBDA}" \
  --config_kv "rlsd_clip_range=${RLSD_CLIP_RANGE}" \
  --config_kv "teacher_update_mode=${TEACHER_UPDATE_MODE}" \
  --config_kv "teacher_max_prompt_length=${TEACHER_MAX_PROMPT_LENGTH}" \
  --config_kv "total_training_steps=${TOTAL_TRAINING_STEPS:-5}" \
  --config_kv "rollout_n=${ROLLOUT_N:-4}"

python3 -m verl.trainer.extpi_rlsd.main_extpi_rlsd \
  trainer.use_v1=False \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  custom_reward_function.path=recipes/extpi_rlsd/rewards/math_verify_reward.py \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path=recipes/extpi_rlsd/rewards/math_verify_reward.py \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-3e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.policy_loss.rlsd_enabled=True \
  actor_rollout_ref.actor.policy_loss.rlsd_lambda="${RLSD_LAMBDA}" \
  actor_rollout_ref.actor.policy_loss.rlsd_reweight_clip_range="${RLSD_CLIP_RANGE}" \
  actor_rollout_ref.actor.policy_loss.rlsd_negative_only="${RLSD_NEGATIVE_ONLY:-False}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-4}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.5}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  +extpi_rlsd.teacher_update_mode="${TEACHER_UPDATE_MODE}" \
  +extpi_rlsd.teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH}" \
  +extpi_rlsd.allow_teacher_prompt_truncation="${ALLOW_TEACHER_PROMPT_TRUNCATION:-True}" \
  +extpi_rlsd.online_teacher_thinking=False \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-5}" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.default_local_dir="${EXTPI_DATA_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.logger="${TRAINER_LOGGER}" \
  "$@"
