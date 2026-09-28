#!/usr/bin/env bash
#SBATCH -J corrected_set_post
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# Required explicit destinations: no default ever points at historical results.
: "${SNAPSHOT_ROOT:?Path to the immutable code/input snapshot}"
: "${CORRECTED_RESULTS_DIR:?New result directory}"
: "${CORRECTED_ARRAY_DIR:?Corrected single-space array directory}"
: "${PYTHON_BIN:?Cluster Python with numpy, pandas, transformers}"
: "${QWEN_MODEL_PATH:?Exact pinned Qwen snapshot}"

MANIFEST_DIR="${SNAPSHOT_ROOT}/focus_alt_exp_pipeline/scoring_manifests/set_variant_qwen"
FRAMED_SCORES="${SNAPSHOT_ROOT}/focus_alt_exp_pipeline/model_scores/focus_hu_remaining_qwen_scores.csv"
CODE_DIR="${SNAPSHOT_ROOT}/focus_alt_exp_pipeline/code"
CORRECTED_SOURCE="${CORRECTED_RESULTS_DIR}/inputs/source_rows.csv"

if [[ "${CORRECTED_RESULTS_DIR}" == */results/set_variant_qwen || "${CORRECTED_ARRAY_DIR}" == */qwen_set_variant_log_probs ]]; then
  echo "Refusing historical array/result destinations" >&2
  exit 2
fi
if [[ -e "${CORRECTED_SOURCE}" || -e "${CORRECTED_RESULTS_DIR}/prediction_grid.csv" ]]; then
  echo "Refusing to overwrite an existing corrected result; use a fresh destination" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${SNAPSHOT_ROOT}"

"${PYTHON_BIN}" "${CODE_DIR}/export_corrected_set_variant_sources.py" \
  --manifest-dir "${MANIFEST_DIR}" --log-probs-dir "${CORRECTED_ARRAY_DIR}" \
  --framed-scores "${FRAMED_SCORES}" --model-path "${QWEN_MODEL_PATH}" \
  --output-dir "${CORRECTED_RESULTS_DIR}/inputs"

# Keep the historical candidate grids, random seed, sampling count, and fold
# assignments for this scoring-only correction. Expanding K is a separate test.
"${PYTHON_BIN}" "${CODE_DIR}/build_set_variant_prediction_grid.py" \
  --source-rows "${CORRECTED_SOURCE}" --fold-reference-source-rows "${MANIFEST_DIR}/source_rows.csv" \
  --log-probs-dir "${CORRECTED_ARRAY_DIR}" --output "${CORRECTED_RESULTS_DIR}/prediction_grid.csv" \
  --top-k-values 1,2,3,5,10,20,30,50,75,100 --top-p-values 0.25,0.5,0.6,0.7,0.8,0.9,0.95 \
  --num-reps 500 --max-prefix-size 32768 --seed 7 --fold-count 10

"${PYTHON_BIN}" "${CODE_DIR}/evaluate_set_variant_grid.py" \
  --prediction-grid "${CORRECTED_RESULTS_DIR}/prediction_grid.csv" \
  --output-dir "${CORRECTED_RESULTS_DIR}/cv_results"

"${PYTHON_BIN}" "${CODE_DIR}/build_linking_structure_tables.py" \
  --source-rows "${CORRECTED_SOURCE}" --results-dir "${CORRECTED_RESULTS_DIR}" \
  --output-dir "${CORRECTED_RESULTS_DIR}/linking_structure_tables"
