# Linking structures in the novel focus study

## Figure 1 Fridge probabilities and ranks

[PNG](fridge_probabilities_and_ranks.png) · [PDF](fridge_probabilities_and_ranks.pdf) · [SVG](fridge_probabilities_and_ranks.svg)

**Caption.** Model predictions and human exclusion judgments for the fridge context, comparing No linking, X but not Y, Ordering, and Set Top-K. Each column shows the same 30 ordered trigger–query pairs. The upper row plots probabilities, with common 0–1 scales and an additional x-axis zoom for No linking. The lower row plots ranks starting at zero, with higher exclusion assigned lower ranks and ties assigned their average position. Circle numerals count pairs sharing a coordinate; colored outlines and letters identify the same four examples throughout. Dashed lines show equal values or ranks, not regression fits. No linking and Set Top-K have identical rank vectors and therefore identical Spearman correlations, while their different probability spacing yields different Pearson correlations. The table reports mean proper log score and both correlations; higher values are better, and bold identifies the highest values among these four models.

**Reading the examples.** A is water → juice, B is juice → water, C is water → milk, and D is meat → milk. Human exclusion is 1.000 for both A and B, .944 for C, and .667 for D. Ordering predicts .148 for A and .852 for B, illustrating its constraint that reverse-pair probabilities sum to one. No linking and Set do not condition on the trigger; their probabilities and ranks for C and D are equal even though the human rates differ.

## Figure 2 Differences across the 16 contexts

[PNG](context_fit_differences.png) · [PDF](context_fit_differences.pdf) · [SVG](context_fit_differences.svg)

**Caption.** Differences between each selected linking model and No linking, calculated separately for all 16 novel-focus contexts. Every fit measure uses the same 30 equally weighted pairs per context. Positive (blue) cells favor linking; negative (red) cells favor No linking. The Pearson and Spearman panels share a symmetric color scale; log score has a separate scale in nats per pair. Labels give numerical differences; values very close to zero retain enough precision to distinguish a loss from a tie. Entries labeled “0 (tie)” agree within 10⁻¹². Counts beneath each panel give the number of contexts with higher, tied, and lower fit for each model. These are descriptive differences, without significance or noninferiority tests.

**Conclusion.** Ordering and Set Top-K each improve log score in 13 of 16 contexts. Set improves Pearson in 12 contexts, but improves Spearman in six, ties in four, and worsens in six. Ordering improves Pearson in eight contexts and Spearman in five. These context-level patterns support conditional benefits, rather than uniform superiority over No linking. This figure does not mix pooled Pearson with average within-context Spearman.

## Data and verification

- `plotted_pair_values.csv`: exact predictions, human rates/counts and ranks for the four models in all 16 contexts.
- `context_fit_values.csv`: all 64 context/model fit summaries.
- `context_fit_differences.csv`: the 144 differences displayed in Figure 2, with baseline and linked values.
- `comparison_counts.csv`: better/tied/worse counts.
- `provenance.json`: source hashes, parameters, highlighted pairs, and verification scope.

All 144 differences were recalculated from saved paired predictions and verified against the existing context-comparison export within 10⁻¹². Saved ranks were independently recalculated with average ties. The complete No linking and Set Top-K fridge rank vectors were verified to be identical. Log scores use the established 10⁻¹⁰ clipping convention. No refitting, resampling, parameter selection or human-response reweighting was performed. Set uses the saved held-out K=20 predictions. Small sampled probabilities and rank changes may depend on finite sampling; the figures do not quantify that uncertainty.

To regenerate from the repository root:

```bash
python focus_alt_exp_pipeline/code/plot_focus_linking_conclusions.py
```
