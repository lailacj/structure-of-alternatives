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
