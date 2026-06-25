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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXTPI_DATA_ROOT="${EXTPI_DATA_ROOT:-/data/users/rchen/extpi-rlsd}"
