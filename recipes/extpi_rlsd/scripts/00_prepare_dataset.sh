#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

args=("$@")
if [ -n "${EVAL_JSONL:-}" ]; then
  IFS=':' read -r -a eval_jsonl_paths <<< "${EVAL_JSONL}"
  for eval_jsonl_path in "${eval_jsonl_paths[@]}"; do
    args+=(--eval_jsonl "${eval_jsonl_path}")
  done
fi

has_eval_jsonl=0
for arg in "${args[@]}"; do
  if [[ "${arg}" == "--eval_jsonl"* ]]; then
    has_eval_jsonl=1
    break
  fi
done
if [ "${has_eval_jsonl}" -ne 1 ] && [ "${ALLOW_MISSING_EVAL_CONTAMINATION:-0}" != "1" ]; then
  echo "Refusing to prepare OPSD without eval contamination files." >&2
  echo "Pass --eval_jsonl /path/to/eval.jsonl, set EVAL_JSONL=path1:path2, or set ALLOW_MISSING_EVAL_CONTAMINATION=1 for local development only." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/../../../tools/extpi_rlsd/prepare_opsd_data.py" \
  --output_dir "${EXTPI_DATA_ROOT}/datasets/opsd_clean" \
  "${args[@]}"
