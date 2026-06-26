#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

OUTPUT_DIR="${OFFICIAL_EVAL_DATA_DIR:-${EXTPI_DATA_ROOT}/datasets/eval/official_math}"
PROMPT_STYLE="${OFFICIAL_EVAL_PROMPT_STYLE:-opsd}"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/prepare_official_math_eval.py" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_style "${PROMPT_STYLE}" \
  "$@"
