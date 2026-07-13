# Cross-dataset Qwen scoring manifest

This directory contains two self-contained Qwen scoring inputs:

- `hu_rnx_no_frame_manifest.csv`: the completed Hu/R&X no-frame batch, with
  609 item-condition rows and 851 unique prompt-candidate pairs.
- `focus_hu_remaining_qwen_manifest.csv`: the remaining focus no-frame plus
  focus/Hu X-but-not-Y batch, with 1,269 observation-frame rows and 871 unique
  prompt-candidate pairs.

## What each condition means

| Dataset family | Condition | Rows | No-frame prompt source |
| --- | --- | ---: | --- |
| Hu et al. benchmark | `scalar_inference` | 309 | Neutral sentence slot from the Hu prompt table |
| Ronai & Xiang (2024) | `ESI` | 60 | Experiment 1 plain-answer context |
| Ronai & Xiang (2024) | `Eweak` | 60 | Experiment 2 weak-QUD context |
| Ronai & Xiang (2024) | `Estrong` | 60 | Experiment 2 strong-QUD context |
| Ronai & Xiang (2024) | `Eonly` | 60 | Matched ESI prompt, with `only` absent |
| Ronai & Xiang (2024) | `Eonlystrong` | 60 | Matched Estrong prompt, with `only` absent |

The validator requires ESI/Eonly and Estrong/Eonlystrong prompts and candidate
pairs to match exactly. It also rejects `only` in the R&X answer frame.

For the remaining-frame manifest:

| Dataset family | Frame | Rows | Candidates scored per row |
| --- | --- | ---: | --- |
| Novel focus | no-frame | 480 | Trigger and query |
| Novel focus | X-but-not-Y | 480 | Query only |
| Hu benchmark | X-but-not-Y | 309 | Query only |

The X-but-not-Y rows never score a trigger continuation and never produce an
alternative-structure contrast. R&X does not receive this frame.

## Rebuild versus score

The committed CSV is all the cluster scorer needs. Rebuilding it requires the
local sibling `experiment_ronai&xiang` repository and is normally unnecessary:

```bash
python focus_alt_exp_pipeline/code/build_cross_dataset_scoring_manifest.py
python focus_alt_exp_pipeline/code/build_remaining_scoring_manifest.py
```

After pulling this repository on the cluster, first validate the manifest and
model path without loading Qwen:

```bash
python focus_alt_exp_pipeline/code/score_qwen_scoring_manifest.py \
  --model-path /path/to/models--Qwen--Qwen2-7B \
  --dry-run
```

Then score on a GPU node:

```bash
python focus_alt_exp_pipeline/code/score_qwen_scoring_manifest.py \
  --model-path /path/to/models--Qwen--Qwen2-7B \
  --dtype bfloat16 \
  --device-map auto \
  --resume
```

The model path should be either a Hugging Face cache root with a usable
`refs/main`, or an exact `snapshots/<revision>` directory. If a cache contains
multiple snapshots and no usable main ref, the scorer stops and asks for the
exact snapshot instead of choosing one silently.

The default final output is
`focus_alt_exp_pipeline/model_scores/hu_rnx_no_frame_qwen_scores.csv`. During a
run, the scorer writes
`focus_alt_exp_pipeline/model_scores/hu_rnx_no_frame_qwen_scores.partial.csv`;
the final filename appears only after all 609 rows finish.

As a generic non-SLURM wrapper, set `QWEN_MODEL_PATH` and run:

```bash
QWEN_MODEL_PATH=/path/to/models--Qwen--Qwen2-7B \
  bash focus_alt_exp_pipeline/cluster/score_hu_rnx_no_frame_qwen.sh
```

This wrapper can be called from the university's existing scheduler job. It
does not encode cluster-specific account, partition, time, or memory settings.

## Remaining focus/Hu scoring run

After pulling the frame-aware manifest and scorer, validate the added batch:

```bash
python focus_alt_exp_pipeline/code/score_qwen_scoring_manifest.py \
  --manifest focus_alt_exp_pipeline/scoring_manifests/focus_hu_remaining_qwen_manifest.csv \
  --model-path /users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B \
  --output focus_alt_exp_pipeline/model_scores/focus_hu_remaining_qwen_scores.csv \
  --dry-run
```

Then run it on a GPU node:

```bash
QWEN_MODEL_PATH=/users/ljohnst7/data/ljohnst7/hf-cache/models--Qwen--Qwen2-7B \
  bash focus_alt_exp_pipeline/cluster/score_focus_hu_remaining_qwen.sh
```

The in-progress checkpoint is
`focus_alt_exp_pipeline/model_scores/focus_hu_remaining_qwen_scores.partial.csv`.
The final `focus_hu_remaining_qwen_scores.csv` appears only after all 1,269
manifest rows finish. Re-running the same command safely resumes an interrupted
job.

## Hu analysis-filter warning

All 309 Hu rows are intentionally scored. Their
`analysis_inclusion_status` values begin with `pending_hu_exact_filter_` because
the exact published Hu analysis subset is not yet frozen. Apply that filter in
the later canonical-data step, not by deleting scoring rows.
