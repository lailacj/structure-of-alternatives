# Word-ranking agreement and exclusion agreement across contexts

September 17, 2026. Exploratory analysis of the historical sampled-prefix results, not the corrected-score run. Existing results, the public viewer, model code, and manuscript files were not changed.

## Main finding

The observation in the viewer is supported descriptively: contexts with better word-ranking agreement generally also have better exclusion agreement. The evidence becomes stronger when the word-ranking measure is calculated from the scores actually used by the sampled structures, rather than the viewer's separate neutral target-score artifact.

This is an association between two context-level performance measures. It is not the mean within-story Spearman reported in Section 5.2, and it does not establish that changing word-ranking quality causes a change in exclusion performance.

## Figures

- [Viewer-score scatterplots](scatter_viewer.png): the direct test of the pattern noticed in the viewer, nine predictors, one point per context.
- [Sampling-score scatterplots](scatter_sampling_scores.png): the same exclusion predictions, with the horizontal axis recomputed from the stored sampling-score export. This is the more relevant diagnostic for the sampled models' first stage.
- [Three-context diagnostic](exceptions.png): word ranks and query-level exclusion summaries for fridge, beach, and cold. This is a descriptive reduction of the item-level data, not the measure used in the association tests.

Each figure also has an editable SVG version. Scatterplots use identical axes, show all available contexts, and highlight the three cases selected before this investigation. No trend line or causal model is fitted. For Set Top-p and Conjunction Top-p, mask has undefined exclusion correlation and is not plotted.

## Tests and uncertainty

For each predictor, the analysis correlates word-ranking Spearman with exclusion Spearman across contexts. There are 16 observations, except the two Top-p cases with 15. The word measure compares the six tested words with human norming ranks; the exclusion measure compares predictions with human exclusion rates across 30 ordered pairs.

The implementation uses 199,999 two-sided random permutations of the context pairings, seed 20260917, and Holm adjustment across nine predictors. It also supplies a joint adjustment across the primary and sensitivity analyses (18 tests). Unadjusted 95% percentile intervals use 20,000 paired context bootstrap samples, with ranks recalculated inside every draw. Leave-one-context-out coefficients and a common-15-context check are included. Tests, confidence intervals, the full specification, and references are in [STATISTICAL_RESULTS.md](STATISTICAL_RESULTS.md).

**Viewer word measure:** No linking (R = .703, adjusted p = .02568) and Disjunction Top-p (R = .710, adjusted p = .02475) pass the .05 threshold after adjusting all nine comparisons. All other associations are positive, but do not pass that adjustment. X but not Y has R = .356, adjusted p = .17596. This does not imply absence of an association, or prove a difference between predictors.

**Sampling-score word measure:** Every sampled linking variant has a positive association, R = .714–.851, with nine-test adjusted p values between .000495 and .00536. All seven also pass the conservative joint 18-test adjustment (adjusted p <= .02844). Conjunction Top-K gives the largest point estimate, R = .851, with a marginal 95% context-bootstrap interval [.590, .943]. This is not evidence that Conjunction is the best exclusion model: it means its performance varies most consistently with this particular word-ranking measure in these data. X but not Y has R = .183, adjusted p = .49382 under this word measure.

All sampled-model associations stay positive when any one context is removed, under both definitions of word agreement. The sensitivity checks retain all contexts rather than discarding exceptions to improve the result.

## Exception investigation

### Fridge: broad success, but a limitation of relative ordering

The neutral context is a thirsty person returning from a walk, followed by "Sure, I have ...". The human norming ranks favor water, juice, and milk over yogurt, ketchup, and meat.

The separate viewer scores reproduce the six-word norming order perfectly (rho = 1.000). The actual sampling scores swap juice/milk and ketchup/meat, giving rho = .886. Nevertheless, the broad division between drinks and less relevant contents is preserved. Exclusion rank agreement is relatively high: Ordering .745, Conjunction Top-K .831, Disjunction Top-K .785, Set Top-K .784, and X but not Y .816.

The strong overall fit conceals an instructive pair:

| Trigger | Query | Human exclusion | Ordering | Set Top-K | Conjunction Top-K | Disjunction Top-K |
|---|---|---:|---:|---:|---:|---:|
| water | juice | 22/22 = 1.000 | .026 | .726 | .026 | .726 |
| juice | water | 6/6 = 1.000 | .974 | 1.000 | .974 | 1.000 |

Ordering necessarily favors one direction over the other: the two precedence probabilities sum to one. Human exclusion can be high in both directions. Thus, getting the contextual order largely right does not make an ordering-only exclusion rule sufficient. Conjunction inherits the low precedence probability for water-to-juice, whereas Disjunction permits membership to contribute even when precedence is unlikely. This comparison illustrates an existing model constraint; it is not a new fitted explanation or evidence that one rule always wins.

### Beach: the displayed rank score overstates the sampled input's agreement, and plausibility is not exclusion

The prompt concerns packing something to wear to a beach. Human norming order is bikini > shorts > sunglasses > jeans > jacket > coat. The separate target scores give rho = .943, but the sampling scores rank shorts > jeans > bikini > sunglasses > jacket > coat, giving rho = .714. The apparent first-stage/second-stage mismatch is therefore partly a consequence of looking at different historical score artifacts.

That does not account for the whole pattern. Human exclusion itself is not ordered exactly like norming plausibility. Averaging the five trigger-specific item rates for each query, bikini is excluded at .950, shorts .760, coat .537, jeans .510, jacket .369, and sunglasses .213. Coat is last in the norming order but is not last in exclusion. These are unweighted means over the five items, not pooled trial proportions; the latter are also supplied in the data table.

For example, after shorts, coat is excluded in 3/4 responses (.750), but Ordering predicts .008 and Set Top-K predicts .070. Conversely, after coat, sunglasses is excluded in 0/3 responses, but Ordering predicts .894. These small denominators mean the individual rates are noisy; they illustrate discrepancies, not reliable population estimates on their own.

One possible interpretation is that substitutability or semantic grouping matters in addition to generic contextual plausibility: clothing and accessories need not be treated as equivalent competitors. This is a hypothesis suggested by the examples, not an established cause. No semantic-category intervention or new human analysis was performed.

### Cold: poor neutral ranking, a possible grammatical mismatch, and good framed ranks on the wrong probability scale

The prompt describes arriving at a cold cabin and ends "Sure, they have a ...". Human norming order is heater > blankets > tea > microwave > sink > fridge. The sampling scores instead give heater > microwave > fridge > sink > tea > blankets (rho = .029); the viewer's separate word score is also near zero (rho = -.029).

Blankets is a clear mismatch. It is second in human norming but sixth among the tested words in the sampling scores, at full-vocabulary rank 5,416. After heater, humans exclude blankets in 15/18 responses (.833), but Set Top-K predicts .002 and Ordering predicts .000. At the same time, after blankets, Ordering predicts microwave exclusion at .996 despite 0/5 exclusion responses.

The literal continuation "a blankets" is grammatically mismatched. The human data retain the raw trigger "blankets", while singular triggers retain expressions such as "a heater". The neutral scoring template's fixed article is therefore a concrete representational concern. It could contribute to the low blanket score, but its causal contribution has not been isolated. Do not claim that article choice explains all of this context's errors without rescoring controlled, grammatical prompt variants.

X but not Y is an exception to the neutral-input pattern: its exclusion Spearman is .761. However, this is rank agreement, not good absolute probabilities. Its score for heater-to-blankets is only .000751 despite the .833 human exclusion rate. The frame ranks many items more appropriately while remaining poorly aligned with their numerical rates. It is not a successful calibration result or a demonstrated repair of the neutral generator.

## Limits and implications for writing

- This was prompted by inspecting the viewer. Label it exploratory, not preregistered confirmation.
- The two human measures differ: norming ranks versus exclusion responses. The predictor components nevertheless share words and model scores, and contextual difficulty could influence both. This is not an independent causal validation of the linking mechanism.
- Six-word ranks are discrete and noisy summaries of a much larger distribution. The analysis evaluates agreement on the tested six words, not the entire candidate vocabulary.
- The 213 exclusion participants each contributed a response in all 16 contexts. Context resampling treats the already-estimated coefficients as fixed and does not propagate shared-participant uncertainty, norming uncertainty, sampling noise, or uncertainty from fitting cross-validation boundaries. The permutation null also requires exchangeability of context pairings conditional on these scores. Interpret the tests within that scope.
- The corrected-score run has not been evaluated here. Both the source definition and all comparative claims should be checked again on its completed, shared artifact.
- Significant associations for sampled structures and a nonsignificant association for X but not Y do not constitute a test that their associations differ. Such a contrast would require a separate paired analysis.

## Suggested Results paragraph (historical exploratory version)

> We examined whether contexts with more human-like rankings of the tested alternatives also yielded more human-like exclusion predictions. Across contexts, word-ranking agreement was positively associated with exclusion agreement for every sampled linking structure. The association depended on which historical scoring artifact supplied the word ranks. Using the separate scores displayed in the viewer, correlations ranged from .540 to .710, and only Disjunction Top-p among the sampled structures survived correction for nine predictor comparisons. Using the scores from which the sampled structures were actually generated, correlations ranged from .714 to .851, and all seven sampled variants survived the same correction. The direction of these associations was preserved when individual contexts were omitted. Thus, the success of exclusion prediction covaried with agreement about the contextual ranking of alternatives, although good word-ranking agreement did not uniformly ensure good exclusion predictions.

For the final paper, prefer a single clearly defined word-score source consistent with the model's input. Retain the historical source comparison in the analysis record rather than silently replacing the viewer result with a more favorable coefficient.

## Files and reproducibility

- `association_tests.csv`: all primary and sensitivity coefficients, intervals, permutation p values, multiplicity adjustments, and leave-one-out ranges.
- `context_measures.csv`: the context-level observations behind the scatterplots and tests.
- `leave_one_context_out.csv`: every leave-one-out check.
- `all_96_word_scores.csv`: human ranks, viewer scores, sampling scores, and vocabulary positions for the tested alternatives.
- `exception_all_90_pairs.csv`: every trigger–query item for the three selected contexts, with human denominators and all nine predictions. No participant identifiers are exported.
- `exception_word_summary.csv`: the six-word and query-level summaries used in the exception figure; both item-mean and pooled-trial human rates are explicitly labeled.
- `exception_diagnostics.csv`: descriptive absolute errors and trigger-residual checks; these are not additional hypothesis tests or primary endpoints.
- `exception_prompts.json`: exact neutral prompts for the three contexts.
- `provenance.json`: source SHA-256 hashes, software versions, seed, and input-preservation checks.

All scientific figures are generated as SVG by `analyze.py`; `render.cjs` renders 2x-resolution PNGs using the bundled image renderer. The analysis rebuilds and checks the original story-level coefficients from human responses, source scores, and saved out-of-fold predictions. Sampling-word scores are recovered from the offline viewer's distribution export only after verifying that its source, prediction-grid, and out-of-fold hashes match the saved historical run.
