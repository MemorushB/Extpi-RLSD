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
export PPO_ENTRYPOINT="verl.trainer.extpi_rlsd.main_extpi_rlsd"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/check_tokenizer_compat.py" \
  --student_model "${MODEL_PATH}" \
  --teacher_model "${TEACHER_MODEL}"

"${SCRIPT_DIR}/run_grpo.sh" \
  actor_rollout_ref.actor.policy_loss.opd_pg_enabled=True \
  actor_rollout_ref.actor.policy_loss.opd_logratio_clip_abs="${OPD_LOGRATIO_CLIP_ABS:-5.0}" \
  ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  +extpi_rlsd.direct_opd_teacher_backend=inline_external_hf \
  +extpi_rlsd.direct_opd_teacher_model_path="${TEACHER_MODEL}" \
  +extpi_rlsd.direct_opd_teacher_micro_batch_size=1 \
  +extpi_rlsd.direct_opd_teacher_device="${TEACHER_DEVICE:-cuda}" \
  +extpi_rlsd.direct_opd_teacher_dtype=bfloat16 \
  +extpi_rlsd.direct_opd_teacher_max_prompt_length="${TEACHER_MAX_PROMPT_LENGTH:-2048}" \
  +extpi_rlsd.direct_opd_teacher_temperature=1.0 \
  +extpi_rlsd.direct_opd_teacher_thinking=False \
  +extpi_rlsd.allow_teacher_prompt_truncation="${ALLOW_TEACHER_PROMPT_TRUNCATION:-True}" \
  "$@"
