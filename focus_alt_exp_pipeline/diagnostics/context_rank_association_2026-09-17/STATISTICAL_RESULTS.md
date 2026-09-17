# Context-level word-ranking / exclusion association

Historical, exploratory fixed-prediction analysis. Not corrected-score results.

## Analysis specification

- Seed 20260917; 199,999 random permutations and 20,000 paired context bootstrap samples.
- Primary: across-context Spearman between viewer word-rank agreement and each predictor's exclusion agreement.
- Secondary sensitivity: replace the word measure with ranks recovered from the actual historical sampling scores; exclusion predictions unchanged.
- Two-sided permutation p = (1 + count(|R_perm| >= |R_observed|))/(B + 1), tolerance 1e-12. One vector is permuted across contexts, retaining ties. The test assumes context pairings are exchangeable under the null, conditional on the saved scores.
- Holm correction over all nine predictors within each word-score definition. The machine-readable table also includes a conservative joint 18-test adjustment.
- Bootstrap: resample paired context records with replacement and re-rank within each draw; report unadjusted 95% percentile intervals. No participant resampling, norming-rank resampling, Monte Carlo rerun, or cross-validation refit.
- 16 contexts, except Set Top-p and Conjunction Top-p: 15 because mask has constant predictions and undefined exclusion correlation. Undefined is never replaced by zero.
- Context coefficients rounded to 12 decimals before ranking to preserve mathematical ties despite floating-point arithmetic. Ordering is included once because its Top-K/Top-p predictions are identical.
- Pearson and leave-one-context-out correlations are descriptive sensitivity checks, not extra hypothesis tests. Common-15-context coefficients are supplied for equal-coverage comparisons.
- These are associations between two estimated statistics, not causal tests. The same 213 participants supplied judgments across the 16 contexts, and fitted predictions share training folds. Fixed-summary resampling does not propagate these dependencies or human measurement uncertainty. Six-word norming ranks are treated as fixed.

## Primary: viewer word measure

| Predictor | Contexts | Spearman R | 95% context-bootstrap interval | Permutation p | Holm p (9) | Leave-one-out range |
|---|---:|---:|---|---:|---:|---|
| No linking | 16 | 0.703 | [0.200, 0.934] | 0.00321 | 0.02568 | [0.638, 0.765] |
| X but not Y | 16 | 0.356 | [-0.249, 0.836] | 0.17596 | 0.17596 | [0.220, 0.599] |
| Ordering | 16 | 0.625 | [0.147, 0.894] | 0.01111 | 0.06663 | [0.550, 0.740] |
| Set Top-K | 16 | 0.540 | [-0.024, 0.878] | 0.03300 | 0.06663 | [0.439, 0.637] |
| Conjunction Top-K | 16 | 0.616 | [0.107, 0.861] | 0.01269 | 0.06663 | [0.543, 0.695] |
| Disjunction Top-K | 16 | 0.647 | [0.138, 0.908] | 0.00806 | 0.05645 | [0.570, 0.763] |
| Set Top-p | 15 | 0.639 | [0.059, 0.920] | 0.01195 | 0.06663 | [0.554, 0.711] |
| Conjunction Top-p | 15 | 0.601 | [0.015, 0.861] | 0.01932 | 0.06663 | [0.516, 0.668] |
| Disjunction Top-p | 16 | 0.710 | [0.248, 0.916] | 0.00275 | 0.02475 | [0.652, 0.783] |

## Sensitivity: sampling-score word measure

| Predictor | Contexts | Spearman R | 95% context-bootstrap interval | Permutation p | Holm p (9) | Leave-one-out range |
|---|---:|---:|---|---:|---:|---|
| No linking | 16 | 0.742 | [0.292, 0.949] | 0.00134 | 0.00536 | [0.690, 0.874] |
| X but not Y | 16 | 0.183 | [-0.415, 0.746] | 0.49382 | 0.49382 | [0.073, 0.396] |
| Ordering | 16 | 0.714 | [0.370, 0.881] | 0.00259 | 0.00536 | [0.660, 0.763] |
| Set Top-K | 16 | 0.733 | [0.328, 0.934] | 0.00169 | 0.00536 | [0.676, 0.816] |
| Conjunction Top-K | 16 | 0.851 | [0.590, 0.943] | 0.00006 | 0.00049 | [0.819, 0.881] |
| Disjunction Top-K | 16 | 0.758 | [0.386, 0.942] | 0.00100 | 0.00500 | [0.706, 0.813] |
| Set Top-p | 15 | 0.803 | [0.424, 0.964] | 0.00057 | 0.00402 | [0.757, 0.889] |
| Conjunction Top-p | 15 | 0.833 | [0.538, 0.945] | 0.00023 | 0.00184 | [0.795, 0.884] |
| Disjunction Top-p | 16 | 0.768 | [0.416, 0.924] | 0.00073 | 0.00441 | [0.726, 0.824] |

## Interpretation boundaries

Confidence intervals are marginal, unadjusted intervals, so they need not agree with the Holm-adjusted testing decisions. A significant association for one model and a nonsignificant association for another does not establish a difference between their associations. No such between-model difference test was performed.

The viewer-word analysis reproduces the user's observation. The sampling-score sensitivity is better aligned with the sampled models' input, but remains based on the historical scorer. Neither should be labeled as corrected-run evidence.

## Method references

- [Permutation-test conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html). The absolute-statistic two-sided convention used here is explicit; it differs from SciPy's twice-the-smaller-tail default.
- [Holm multiple-testing adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

## Reproduction

Run `analyze.py` with NumPy and pandas. Run `render.cjs` with the bundled Node executable to render the SVG files to PNG. All outputs are confined to this new directory. `provenance.json` records input hashes and verification checks.
