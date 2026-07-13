# Cross-dataset Qwen scoring manifest

`hu_rnx_no_frame_manifest.csv` is the self-contained input for the new Qwen
scoring run. It has 609 item-condition rows and validates to 851 unique
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
pairs to match exactly. It also rejects `only` in the R&X answer frame. The
X-but-not-Y frame is outside this manifest.

## Rebuild versus score

The committed CSV is all the cluster scorer needs. Rebuilding it requires the
local sibling `experiment_ronai&xiang` repository and is normally unnecessary:

```bash
python focus_alt_exp_pipeline/code/build_cross_dataset_scoring_manifest.py
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

## Hu analysis-filter warning

All 309 Hu rows are intentionally scored. Their
`analysis_inclusion_status` values begin with `pending_hu_exact_filter_` because
the exact published Hu analysis subset is not yet frozen. Apply that filter in
the later canonical-data step, not by deleting scoring rows.
