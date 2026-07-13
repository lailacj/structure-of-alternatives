# Canonical data

These tables are the stable interface between dataset-specific human data,
standardized Qwen scoring, and cross-dataset alternative-structure evaluation.
Each row contains one scored item/context, its human query-exclusion rate, the
no-frame trigger and query scores, and the X-but-not-Y query score when that
frame is defined.

Rebuild all tables from the repository root with:

```bash
python focus_alt_exp_pipeline/code/build_cross_dataset_canonical_observations.py
```

The builder requires the sibling `experiment_ronai&xiang` project used to
construct the Qwen manifests.

## Outputs

| File | Rows | Human groups | Exact-count rows | Rate-only rows | X-but-not-Y |
| --- | ---: | ---: | ---: | ---: | ---: |
| `novel_focus_observations.csv` | 480 | 16 stories | 480 | 0 | 480 |
| `hu_2023_observations.csv` | 309 | 223 scales | 60 | 249 | 309 |
| `ronai_xiang_2024_observations.csv` | 300 | 60 items | 300 | 0 | 0 |
| `all_observations.csv` | 1,089 | 299 family-specific groups | 840 | 249 | 789 |

All score rows use Qwen2-7B revision
`453ed1575b739b5b03ce3758b23befdb0967f40e` and have complete model
provenance.

## Human-response grain and counts

`human_count_status=exact` means `human_yes` and `human_total` are observed
counts and `human_rate` is validated as their ratio. `rate_only` means the
source publishes a response rate but the copied source material does not
contain a defensible denominator; both count columns must then be empty. The
schema rejects pseudo-counts on rate-only rows.

Focus and Ronai & Xiang (2024) have exact counts throughout. Within the Hu
benchmark, the 60 Ronai & Xiang (2022) counts are recovered from trial-level
data (40 responses per item). Gotzner et al. (2018), Pankratz & van Tiel
(2021), and van Tiel et al. (2016) remain rate-only. These rows support
correlations and item-mean proper scores, but not response-level binomial log
likelihood unless exact counts are recovered from source data.

The Hu table contains 309 scored contexts but 223 human scale observations.
The difference is van Tiel et al.: each of 43 human scales is represented in
three non-neutral sentence templates. `group_id` is the scale, so all templates
for a scale stay together in grouped validation.

For comparison with Hu et al.'s original string-surprisal analysis, the
canonical table also freezes the literal subset created by their public
notebook's bare `dropna()` at commit
`50a7064290a841b81b2608524000b81a33ddc4b0`. The resulting analysis counts are
57 Ronai-Xiang scales, 50 Pankratz-van Tiel scales, 67 Gotzner scales, and 39
van Tiel scales. The van Tiel score is the mean of three template scores.

## Frames and surfaces

X-but-not-Y is present only for the Hu and novel-focus datasets. Ronai & Xiang
(2024) has five condition rows per item and no X-but-not-Y diagnostic.
Alternative-structure models use only the no-frame trigger and query scores.

Hu's no-frame and X-but-not-Y prompts sometimes require different inflected
surfaces, such as no-frame `completed` versus framed `did not complete`. The
canonical `trigger` and `query` columns store the no-frame surfaces; Hu-specific
`x_but_not_y_trigger` and `x_but_not_y_query` columns preserve the framed
surfaces explicitly.

## Hu inclusion status

All 309 scored Hu contexts remain in the canonical table so the raw evidence is
never discarded. `hu_original_analysis_included` selects the exact comparison
subset. The excluded rows are marked either as missing GPT-2 string surprisal
or, for the three templates of van Tiel's `unsettling/horrific` scale, as
excluded by the source notebook's broad `dropna()` because the unrelated LSA
field is empty.
