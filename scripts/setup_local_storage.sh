#!/usr/bin/env bash
# Create local large-file storage under /data and symlink it into the project.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${EXTPI_DATA_ROOT:-/data/users/rchen/extpi-rlsd}"

mkdir -p \
  "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/checkpoints" \
  "${DATA_ROOT}/outputs" \
  "${DATA_ROOT}/cache"

link_dir() {
  local name="$1"
  local target="$2"
  local link="${PROJECT_DIR}/${name}"
  if [ -L "${link}" ]; then
    local current
    current="$(readlink "${link}")"
    if [ "${current}" = "${target}" ]; then
      return
    fi
    rm "${link}"
  elif [ -e "${link}" ]; then
    echo "[setup_local_storage] ${link} exists and is not a symlink; refusing to replace it." >&2
    exit 1
  fi
  ln -s "${target}" "${link}"
}

link_dir data "${DATA_ROOT}/datasets"
link_dir checkpoints "${DATA_ROOT}/checkpoints"
link_dir outputs "${DATA_ROOT}/outputs"
link_dir cache "${DATA_ROOT}/cache"

echo "ExtPI-RLSD storage is ready at ${DATA_ROOT}"
