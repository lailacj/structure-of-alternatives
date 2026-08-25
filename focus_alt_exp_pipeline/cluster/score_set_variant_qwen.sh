#!/usr/bin/env bash
set -euo pipefail

# Run after build_set_variant_scoring_manifest.py.  The bigram vocabulary must
# include every required bigram listed in ${SET_VARIANT_DIR}/required_candidates.txt.
: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the Qwen cache root or exact snapshot directory}"
: "${VOCAB_1GRAM:?Set VOCAB_1GRAM to the cluster unigram candidate vocabulary}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_DEVICE_MAP="${QWEN_DEVICE_MAP:-auto}"
SET_VARIANT_DIR="${SET_VARIANT_DIR:-focus_alt_exp_pipeline/scoring_manifests/set_variant_qwen}"
QWEN_LOG_PROBS_DIR="${QWEN_LOG_PROBS_DIR:-focus_alt_exp_pipeline/model_scores/set_variant_qwen_log_probs}"

exec "${PYTHON_BIN}" focus_alt_exp_pipeline/code/precompute_qwen_vocab_log_probs.py \
  --prompts-csv "${SET_VARIANT_DIR}/prompts.csv" \
  --prompt-context-col prompt_id \
  --prompt-col generation_prompt \
  --vocab-1gram "${VOCAB_1GRAM}" \
  --bigram-support-manifest "${SET_VARIANT_DIR}/selection_manifest.json" \
  --model-path "${QWEN_MODEL_PATH}" \
  --output-dir "${QWEN_LOG_PROBS_DIR}" \
  --dtype "${QWEN_DTYPE}" \
  --device-map "${QWEN_DEVICE_MAP}" \
  --local-files-only \
  --hf-offline \
  "$@"
