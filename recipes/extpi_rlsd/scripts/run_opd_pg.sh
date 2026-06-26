#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_pg_qwen8b_sampled_k1}"
export METHOD_NAME="${METHOD_NAME:-opd_pg_qwen8b}"
export TEACHER_MODEL
export MODEL_PATH

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/check_tokenizer_compat.py" \
  --student_model "${MODEL_PATH}" \
  --teacher_model "${TEACHER_MODEL}"

"${SCRIPT_DIR}/run_grpo.sh" \
  distillation.enabled=True \
  distillation.n_gpus_per_node=1 \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.key=extpi_rlsd/math \
  distillation.teacher_models.teacher_model.num_replicas=1 \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL:-0.35}" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp=5.0 \
  distillation.distillation_loss.log_prob_min_clamp=-5.0 \
  "$@"
