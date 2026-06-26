#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INPUT_JSONL="${INPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean_qwen8b_pi.jsonl}"
PLAIN_JSONL="${PLAIN_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/recipient_plain_qwen17b.jsonl}"
PI_JSONL="${PI_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/recipient_pi_qwen17b.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/all_clean_qwen8b_pi_screened.jsonl}"
MANIFEST_JSON="${MANIFEST_JSON:-${EXTPI_DATA_ROOT}/datasets/opsd_clean/recipient_uplift_manifest.json}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
RECIPIENT_SEEDS="${RECIPIENT_SEEDS:-0,1,2,3}"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/generate_recipient_completions.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${PLAIN_JSONL}" \
  --mode plain \
  --model "${MODEL_PATH}" \
  --backend "${RECIPIENT_BACKEND:-vllm}" \
  --seeds "${RECIPIENT_SEEDS}" \
  --max_new_tokens "${RECIPIENT_MAX_NEW_TOKENS:-1024}" \
  --temperature "${RECIPIENT_TEMPERATURE:-1.0}" \
  --top_p "${RECIPIENT_TOP_P:-1.0}" \
  "$@"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/generate_recipient_completions.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${PI_JSONL}" \
  --mode pi \
  --model "${MODEL_PATH}" \
  --backend "${RECIPIENT_BACKEND:-vllm}" \
  --seeds "${RECIPIENT_SEEDS}" \
  --max_new_tokens "${RECIPIENT_MAX_NEW_TOKENS:-1024}" \
  --temperature "${RECIPIENT_TEMPERATURE:-1.0}" \
  --top_p "${RECIPIENT_TOP_P:-1.0}" \
  "$@"

python3 "${EXTPI_REPO_ROOT}/tools/extpi_rlsd/compute_recipient_uplift.py" \
  --input_jsonl "${INPUT_JSONL}" \
  --plain_jsonl "${PLAIN_JSONL}" \
  --pi_jsonl "${PI_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --manifest "${MANIFEST_JSON}" \
  --expected_samples 4
