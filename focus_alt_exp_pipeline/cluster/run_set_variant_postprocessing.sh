#!/usr/bin/env bash
#SBATCH -J set_variant_post
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o focus_alt_exp_pipeline/cluster/slurm_output/set_variant_post.%j.out
#SBATCH -e focus_alt_exp_pipeline/cluster/slurm_output/set_variant_post.%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/ljohnst7/data/ljohnst7/structure-of-alternatives}"
PYTHON_BIN="${PYTHON_BIN:-/users/ljohnst7/data/ljohnst7/oscar_jobs/venv/llm_nextword_env/bin/python}"
LOG_PROBS_DIR="${LOG_PROBS_DIR:-/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/focus_alt_exp_pipeline/results/set_variant_qwen}"

cd "${REPO_ROOT}"
mkdir -p "${RESULTS_DIR}"

"${PYTHON_BIN}" focus_alt_exp_pipeline/code/validate_set_variant_qwen_scores.py \
  --log-probs-dir "${LOG_PROBS_DIR}" \
  --report "${RESULTS_DIR}/score_validation.json"

"${PYTHON_BIN}" focus_alt_exp_pipeline/code/build_set_variant_prediction_grid.py \
  --log-probs-dir "${LOG_PROBS_DIR}" \
  --output "${RESULTS_DIR}/prediction_grid.csv" \
  --num-reps 500 \
  --max-prefix-size 32768 \
  --seed 7 \
  --fold-count 10

"${PYTHON_BIN}" focus_alt_exp_pipeline/code/evaluate_set_variant_grid.py \
  --prediction-grid "${RESULTS_DIR}/prediction_grid.csv" \
  --output-dir "${RESULTS_DIR}/cv_results"
