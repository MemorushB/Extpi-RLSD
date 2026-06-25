#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

RESULTS_DIR="${RESULTS_DIR:-${EXTPI_DATA_ROOT}/outputs/eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXTPI_DATA_ROOT}/outputs/reports}"
mkdir -p "${OUTPUT_DIR}"

mapfile -t RESULT_FILES < <(find "${RESULTS_DIR}" -maxdepth 2 -type f -name '*.json' | sort)
if [ "${#RESULT_FILES[@]}" -eq 0 ]; then
  echo "No evaluation JSON files found under ${RESULTS_DIR}." >&2
  echo "Run checkpoint evaluation first, then re-run this aggregation script." >&2
  exit 1
fi

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/aggregate_results.py" \
  --input "${RESULT_FILES[@]}" \
  --output_csv "${OUTPUT_DIR}/extpi_rlsd_eval_summary.csv" \
  --output_md "${OUTPUT_DIR}/extpi_rlsd_eval_summary.md"

echo "Wrote ${OUTPUT_DIR}/extpi_rlsd_eval_summary.csv" >&2
echo "Wrote ${OUTPUT_DIR}/extpi_rlsd_eval_summary.md" >&2
