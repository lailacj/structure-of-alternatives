#!/usr/bin/env bash
# Run inside a user-allocated GPU session; this script never submits a job.
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the exact Qwen2-7B snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e directory}"
RNX_MODEL_SNAPSHOT="${QWEN_MODEL_PATH%/}"
if [[ "${RNX_MODEL_SNAPSHOT##*/}" != "453ed1575b739b5b03ce3758b23befdb0967f40e" ]]; then
  echo "Use the pinned snapshot directory, not the cache root or another revision." >&2
  exit 2
fi

RNX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${RNX_SCRIPT_DIR}/../.."

exec "${PYTHON_BIN:-python}" focus_alt_exp_pipeline/code/score_qwen_scoring_manifest.py \
  --manifest "${MANIFEST_PATH:-focus_alt_exp_pipeline/scoring_manifests/rnx_x_but_not_y_manifest.csv}" \
  --model-path "${QWEN_MODEL_PATH}" \
  --output "${OUTPUT_PATH:-focus_alt_exp_pipeline/model_scores/rnx_x_but_not_y_qwen_scores.csv}" \
  --dtype "${QWEN_DTYPE:-bfloat16}" \
  --device-map "${QWEN_DEVICE_MAP:-auto}" \
  --resume \
  "$@"
