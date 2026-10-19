# Full sampled-prefix results

These results use all 360 Qwen prompts (1,089 canonical rows; 993 analysis units).

## Main results

Reported log scores sum item scores without dividing by item counts. The original
training-fold dataset-balanced mean objective and saved boundaries are unchanged.
Sums across datasets weight analysis units equally and depend on dataset sizes.
These are sums of pair/scale-level scores, not participant-weighted log-likelihoods.

- Best proposed structure: **ordering** with summed OOF log score **-1536.465**.
- Fold-safe dataset base-rate score: **-654.471**.
- Difference between the best proposed structure and base rate: **-881.995** (negative is worse).
- Top-K selected **K=20 in 10 fold(s)**. All selected values are below the tested maximum K=100.
- Top-p selected **p=0.6 in 7 fold(s), p=0.7 in 3 fold(s)**.
- Novel-focus correlations range from approximately **0.47 to 0.54** across structures, while most scalar-dataset correlations are weak or inconsistent.
- Matched R&X prompts produce identical model predictions even though mean human exclusion increases in Eonly by 0.30 and Eonlystrong by 0.27.

## Interpretation

The strongest proposed structure is **ordering**. Its difference from
the fold-safe condition-specific base rate is **-881.995**.
Extreme
0/1 Monte Carlo probabilities produce large penalties when the model disagrees
with non-extreme human rates. Calibration/noise requires discussion before
the final analysis is frozen. All selected values are below the tested maximum K=100.

## Coverage

- 16 novel-focus contexts are represented.
- Hu g18 contains 4 distinct scored prompts in this analysis.
- All planned Qwen prompts are included.


## Spearman: Hu datasets and R&X conditions

Hu ranks scales within each dataset after template aggregation. R&X ranks 60 items within each condition. The units match Pearson; ties receive average ranks.

| Dataset | No linking structure | X but not Y | Set Top-K | Set Top-p | Ordering | Conjunction Top-K | Conjunction Top-p | Disjunction Top-K | Disjunction Top-p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| van Tiel et al. (2016) | 0.100 | 0.139 | 0.166 | 0.067 | -0.040 | 0.148 | 0.020 | -0.032 | -0.006 |
| Gotzner et al. (2018) | 0.081 | 0.043 | 0.092 | -0.080 | 0.076 | 0.091 | -0.078 | 0.078 | 0.075 |
| Pankratz & van Tiel (2021) | -0.004 | 0.216 | -0.033 | 0.003 | -0.118 | -0.034 | -0.002 | -0.116 | -0.132 |
| Ronai & Xiang (2022) | 0.136 | 0.371 | 0.221 | 0.104 | 0.153 | 0.247 | 0.091 | 0.158 | 0.202 |
| R&X ESI | 0.163 | NA | 0.245 | 0.107 | 0.175 | 0.271 | 0.111 | 0.177 | 0.219 |
| R&X Eweak | -0.130 | NA | -0.069 | -0.209 | -0.083 | -0.067 | -0.132 | -0.074 | -0.171 |
| R&X Estrong | -0.259 | NA | -0.150 | -0.372 | 0.113 | 0.094 | -0.050 | -0.134 | -0.350 |
| R&X Eonly | 0.116 | NA | 0.221 | 0.029 | 0.156 | 0.215 | 0.021 | 0.179 | 0.207 |
| R&X Eonlystrong | -0.223 | NA | -0.147 | -0.277 | -0.036 | -0.104 | -0.228 | -0.139 | -0.207 |

## Within-context Spearman

Word rankings use six tested alternatives and summed neutral continuation log probabilities. Negation rankings use all 30 ordered trigger–query pairs. Ties receive average ranks; constant vectors are undefined. Means weight valid contexts equally; valid/total counts are reported. Boundary selection still uses training log score.

| model | mean_within_context_spearman | valid_contexts | total_contexts | measure | structure | variant |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2-7B | 0.6984228473198827 | 16 | 16 | word_ranking |  |  |
| Qwen2-7B | 0.5837382395176375 | 16 | 16 | negation | No linking structure | direct |
| Qwen2-7B | 0.5212825350937136 | 16 | 16 | negation | X but not Y | direct |
| Qwen2-7B | 0.5690937938402522 | 16 | 16 | negation | conjunction | top_k |
| Qwen2-7B | 0.5646375198184855 | 15 | 16 | negation | conjunction | top_p |
| Qwen2-7B | 0.5284888621050545 | 16 | 16 | negation | disjunction | top_k |
| Qwen2-7B | 0.530611158370048 | 16 | 16 | negation | disjunction | top_p |
| Qwen2-7B | 0.5130660773065275 | 16 | 16 | negation | ordering | top_k |
| Qwen2-7B | 0.5130660773065275 | 16 | 16 | negation | ordering | top_p |
| Qwen2-7B | 0.5665701692690844 | 16 | 16 | negation | set | top_k |
| Qwen2-7B | 0.5534343833212425 | 15 | 16 | negation | set | top_p |
