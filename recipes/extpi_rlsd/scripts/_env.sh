#!/usr/bin/env bash
# Shared environment defaults for single-H200 ExtPI-RLSD runs.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
if [ "${CUDA_VISIBLE_DEVICES}" != "6" ]; then
  echo "ExtPI-RLSD scripts may only use gpu6 by default; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"
if [ "${NPROC_PER_NODE}" != "1" ] || [ "${NGPUS_PER_NODE}" != "1" ]; then
  echo "ExtPI-RLSD MVP defaults to one GPU: NPROC_PER_NODE=1 and NGPUS_PER_NODE=1." >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [ -n "${EXTPI_FORCE_PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="${EXTPI_FORCE_PYTORCH_CUDA_ALLOC_CONF}"
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export EXTPI_DATA_ROOT="${EXTPI_DATA_ROOT:-/data/users/rchen/extpi-rlsd}"
export EXTPI_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${EXTPI_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
    --config_kv "nproc_per_node=${NPROC_PER_NODE}"
    --config_kv "ngpus_per_node=${NGPUS_PER_NODE}"
    --config_kv "trainer_logger=${TRAINER_LOGGER}"
  )
  if [ -n "${SPLIT_MANIFEST:-}" ]; then
    manifest_args+=(--dataset_manifest "${SPLIT_MANIFEST}")
  fi
  python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/write_run_manifest.py" "${manifest_args[@]}" "$@"
  echo "Wrote run manifest: ${manifest_path}" >&2
}
