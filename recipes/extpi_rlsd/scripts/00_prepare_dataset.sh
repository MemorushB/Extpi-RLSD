#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/prepare_opsd_data.py" \
  --output_dir "${EXTPI_DATA_ROOT}/datasets/opsd_clean" \
  "$@"
