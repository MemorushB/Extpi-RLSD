#!/usr/bin/env bash
# Shared environment defaults for multi-GPU ExtPI-RLSD runs.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXTPI_VISIBLE_GPU_COUNT="${EXTPI_VISIBLE_GPU_COUNT:-$(python3 - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([item for item in visible.split(",") if item.strip()]) if visible else 1)
PY
)}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-${EXTPI_VISIBLE_GPU_COUNT}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NGPUS_PER_NODE}}"

if [ "${NGPUS_PER_NODE}" -lt 2 ]; then
  echo "Use single-card scripts for NGPUS_PER_NODE=1; multi scripts expect >=2 GPUs." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXTPI_DATA_ROOT="${EXTPI_DATA_ROOT:-/data/users/rchen/extpi-rlsd}"
export EXTPI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${EXTPI_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Multi-GPU resource pools should use Ray's normal CUDA isolation.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

if [ -z "${TRAINER_LOGGER:-}" ]; then
  case "${EXTPI_ENABLE_WANDB:-1}" in
    0|false|FALSE|no|NO)
      export TRAINER_LOGGER='["console"]'
      ;;
    *)
      export TRAINER_LOGGER='["console","wandb"]'
      ;;
  esac
fi
if [[ "${TRAINER_LOGGER}" == *wandb* ]]; then
  export WANDB_DIR="${WANDB_DIR:-${EXTPI_DATA_ROOT}/outputs/wandb}"
  mkdir -p "${WANDB_DIR}"
fi

write_extpi_run_manifest() {
  local experiment_name="$1"
  shift
  local manifest_path="${RUN_MANIFEST:-${EXTPI_DATA_ROOT}/outputs/${experiment_name}/run_manifest.json}"
  local manifest_args=(
    --output "${manifest_path}"
    --seed "${SEED:-42}"
    --config_kv "experiment_name=${experiment_name}"
    --config_kv "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    --config_kv "nnodes=${NNODES}"
    --config_kv "nproc_per_node=${NPROC_PER_NODE}"
    --config_kv "ngpus_per_node=${NGPUS_PER_NODE}"
    --config_kv "multi_gpu=true"
    --config_kv "trainer_logger=${TRAINER_LOGGER}"
  )
  if [ -n "${SPLIT_MANIFEST:-}" ]; then
    manifest_args+=(--dataset_manifest "${SPLIT_MANIFEST}")
  fi
  python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/write_run_manifest.py" "${manifest_args[@]}" "$@"
  echo "Wrote run manifest: ${manifest_path}" >&2
}
