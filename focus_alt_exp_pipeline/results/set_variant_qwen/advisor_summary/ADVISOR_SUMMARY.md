# Full sampled-prefix results

These results use all 360 Qwen prompts (1,089 canonical rows; 993 analysis units).

## Main results

- Best proposed structure: **ordering** with balanced OOF mean log score **-1.622**.
- Fold-safe dataset base-rate score: **-0.634**.
- Difference between the best proposed structure and base rate: **-0.988** (negative is worse).
- Top-K selected **K=100 in 10 fold(s)**; selections at K=100 are at the maximum tested value.
- Top-p selected **p=0.6 in 10 fold(s)**.
- Novel-focus correlations range from approximately **0.46 to 0.59** across structures, while most scalar-dataset correlations are weak or inconsistent.
- Matched R&X prompts produce identical model predictions even though mean human exclusion increases in Eonly by 0.30 and Eonlystrong by 0.27.

## Interpretation

Ordering is the strongest of the proposed structures, but none beats a simple
fold-safe condition-specific base-rate predictor in proper log score. Extreme
0/1 Monte Carlo probabilities produce large penalties when the model disagrees
with non-extreme human rates. This suggests that calibration/noise and the
Top-K search range require discussion before the final analysis is frozen.

## Coverage

- 16 novel-focus contexts are represented.
- Hu g18 contains 4 distinct scored prompts in this analysis.
- All planned Qwen prompts are included.


## Within-context Spearman

Word rankings use six tested alternatives and summed neutral continuation log probabilities. Negation rankings use all 30 ordered trigger–query pairs. Ties receive average ranks; constant vectors are undefined. Means weight valid contexts equally; valid/total counts are reported. Boundary selection still uses training log score.

| model | mean_within_context_spearman | valid_contexts | total_contexts | measure | structure | variant |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2-7B | 0.7071428571428572 | 16 | 16 | word_ranking |  |  |
| Qwen2-7B | 0.5850552362852297 | 16 | 16 | negation | No linking structure | direct |
| Qwen2-7B | 0.5212825350937136 | 16 | 16 | negation | X but not Y | direct |
| Qwen2-7B | 0.5564875861968573 | 16 | 16 | negation | conjunction | top_k |
| Qwen2-7B | 0.5622656002104559 | 15 | 16 | negation | conjunction | top_p |
| Qwen2-7B | 0.5298767658369093 | 16 | 16 | negation | disjunction | top_k |
| Qwen2-7B | 0.5186599354607055 | 16 | 16 | negation | disjunction | top_p |
| Qwen2-7B | 0.5017579889485276 | 16 | 16 | negation | ordering | top_k |
| Qwen2-7B | 0.5017579889485276 | 16 | 16 | negation | ordering | top_p |
| Qwen2-7B | 0.5434125445613756 | 16 | 16 | negation | set | top_k |
| Qwen2-7B | 0.5495992196784905 | 15 | 16 | negation | set | top_p |
