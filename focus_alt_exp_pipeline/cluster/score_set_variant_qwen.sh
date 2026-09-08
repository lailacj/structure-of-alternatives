#!/usr/bin/env bash
#SBATCH -J qwen_set_variant
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-359%2
#SBATCH -o focus_alt_exp_pipeline/cluster/slurm_output/qwen_set_variant.%A_%a.out
#SBATCH -e focus_alt_exp_pipeline/cluster/slurm_output/qwen_set_variant.%A_%a.err

set -euo pipefail

# Run after both set-variant builders. Each array task handles one prompt and
# resumes from that prompt's .progress.json and partially filled .npy file.
# Slurm executes a spooled copy of this script, so do not derive the repository
# location from BASH_SOURCE.
REPO_ROOT="${REPO_ROOT:-/users/ljohnst7/data/ljohnst7/structure-of-alternatives}"

PYTHON_BIN="${PYTHON_BIN:-/users/ljohnst7/data/ljohnst7/oscar_jobs/venv/llm_nextword_env/bin/python}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_DEVICE_MAP="${QWEN_DEVICE_MAP:-auto}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e}"
VOCAB_1GRAM="${VOCAB_1GRAM:-/users/ljohnst7/data/ljohnst7/ngrams/set_variant_qwen_support/vocab_1gram.txt}"
SET_VARIANT_DIR="${SET_VARIANT_DIR:-${REPO_ROOT}/focus_alt_exp_pipeline/scoring_manifests/set_variant_qwen}"
QWEN_LOG_PROBS_DIR="${QWEN_LOG_PROBS_DIR:-/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs}"
TARGET_VOCAB_SIZE="${TARGET_VOCAB_SIZE:-121301}"

PROMPT_COUNT="$("${PYTHON_BIN}" -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    print(sum(1 for _ in csv.DictReader(stream)))
' "${SET_VARIANT_DIR}/prompts.csv")"
CONTEXT_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= PROMPT_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} is outside prompt range 0-$((PROMPT_COUNT - 1))" >&2
    exit 2
  fi
  PROMPT_ID="$("${PYTHON_BIN}" -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
print(rows[int(sys.argv[2])]["prompt_id"])
' "${SET_VARIANT_DIR}/prompts.csv" "${SLURM_ARRAY_TASK_ID}")"
  CONTEXT_ARGS=(--contexts "${PROMPT_ID}")
elif [[ "${ALLOW_ALL_PROMPTS:-0}" != "1" ]]; then
  echo "Refusing to score all ${PROMPT_COUNT} prompts outside a Slurm array." >&2
  echo "Submit with sbatch, or set ALLOW_ALL_PROMPTS=1 explicitly." >&2
  exit 2
fi

cd "${REPO_ROOT}"
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
  --target-vocab-size "${TARGET_VOCAB_SIZE}" \
  --local-files-only \
  --hf-offline \
  "${CONTEXT_ARGS[@]}" \
  "$@"
