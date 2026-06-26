#!/usr/bin/env bash
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_multi.sh"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_pg_qwen8b_sampled_k1_multi_teacher_pool}"
export METHOD_NAME="${METHOD_NAME:-opd_pg_qwen8b_multi_teacher_pool}"
export TEACHER_MODEL
export MODEL_PATH

TEACHER_NGPUS_PER_NODE="${TEACHER_NGPUS_PER_NODE:-1}"
if [ "${TEACHER_NGPUS_PER_NODE}" -lt 1 ]; then
  echo "TEACHER_NGPUS_PER_NODE must be >=1 for multi teacher-pool OPD." >&2
  exit 1
fi
if [ $((NGPUS_PER_NODE + TEACHER_NGPUS_PER_NODE)) -gt "${EXTPI_VISIBLE_GPU_COUNT}" ]; then
  echo "Requested actor GPUs (${NGPUS_PER_NODE}) + teacher GPUs (${TEACHER_NGPUS_PER_NODE}) exceeds visible GPUs (${EXTPI_VISIBLE_GPU_COUNT})." >&2
  echo "Set NGPUS_PER_NODE to the actor pool size, for example 3 with CUDA_VISIBLE_DEVICES=0,1,2,3 and TEACHER_NGPUS_PER_NODE=1." >&2
  exit 1
fi

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/check_tokenizer_compat.py" \
  --student_model "${MODEL_PATH}" \
  --teacher_model "${TEACHER_MODEL}"

"${SCRIPT_DIR}/run_grpo_multi.sh" \
  distillation.enabled=True \
  distillation.n_gpus_per_node="${TEACHER_NGPUS_PER_NODE}" \
  distillation.nnodes="${TEACHER_NNODES:-${NNODES}}" \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP_SIZE:-1}" \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL:-0.5}" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.loss_max_clamp="${OPD_LOGRATIO_CLIP_ABS:-5.0}" \
  distillation.distillation_loss.log_prob_min_clamp="-${OPD_LOGRATIO_CLIP_ABS:-5.0}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "$@"
