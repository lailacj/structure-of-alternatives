#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the Qwen cache root or exact snapshot directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_DEVICE_MAP="${QWEN_DEVICE_MAP:-auto}"
MANIFEST_PATH="${MANIFEST_PATH:-focus_alt_exp_pipeline/scoring_manifests/focus_hu_remaining_qwen_manifest.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-focus_alt_exp_pipeline/model_scores/focus_hu_remaining_qwen_scores.csv}"

exec "${PYTHON_BIN}" focus_alt_exp_pipeline/code/score_qwen_scoring_manifest.py \
  --manifest "${MANIFEST_PATH}" \
  --model-path "${QWEN_MODEL_PATH}" \
  --output "${OUTPUT_PATH}" \
  --dtype "${QWEN_DTYPE}" \
  --device-map "${QWEN_DEVICE_MAP}" \
  --resume \
  "$@"
