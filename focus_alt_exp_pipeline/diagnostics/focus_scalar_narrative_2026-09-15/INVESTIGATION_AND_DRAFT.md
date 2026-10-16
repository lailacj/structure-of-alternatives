# From contextual expectedness to exclusion: first investigation

September 15, 2026. Exploratory analysis of the saved September 8 sampled-prefix results, not a rerun. No existing scores, boundaries, results tables, or manuscript files have been changed. The input hashes match the distributions embedded in the current offline results viewer; all available saved Pearson correlations were reproduced from aligned item predictions.

## Working conclusion

The current linking structures track human exclusion more strongly in the focus experiment than in the scalar benchmarks. However, the explanation should **not** be that scalar next-word distributions simply lack useful rankings. Some scalar ranking information is present before the linking rules are applied. The better working question is:

> When does contextual expectedness provide the information needed for exclusion, and which aspects of that information survive the mapping into an exclusion probability?

We need to distinguish candidate ranking, cross-item correlation, and probability accuracy. We also need a matched-scoring comparison before attributing the reported gains exclusively to the linking structures.

## 1. What the saved result table establishes

Pearson correlations, using the same analysis units within each dataset:

| Dataset | Direct neutral | X but not Y | Set Top-K | Set Top-p |
| --- | ---: | ---: | ---: | ---: |
| Focus experiment | .363 | .263 | .594 | .499 |
| van Tiel et al. (2016) | .162 | .133 | .276 | .259 |
| Gotzner et al. (2018) | .159 | .248 | .278 | .297 |
| Pankratz & van Tiel (2021) | −.110 | .129 | −.162 | −.153 |
| Ronai & Xiang (2022) | .087 | .371 | .183 | .169 |

Set Top-K is the strongest focus model by Pearson correlation. Focus Ordering, Conjunction Top-K, and Disjunction Top-K also have positive correlations of .462, .560, and .524, respectively. This does not show that adding a trigger-sensitive ordering relation improves on Set: Set itself is the strongest of these point estimates.

X but not Y has the highest Pearson correlation among the nine reported models in two of the four scalar datasets assembled by Hu: Pankratz & van Tiel and Ronai & Xiang 2022. It is not the best on van Tiel or Gotzner. The five Ronai–Xiang 2024 conditions have no X-but-not-Y predictions in the current analysis, so this contrast cannot be extended to them.

## 2. The comparison is not yet a clean linking-rule ablation

The direct baseline uses saved target-continuation scores. The linking structures use separate full-vocabulary score arrays. All 947 unique prompt–target entries compared in the viewer differ between those artifacts by more than 0.0001 log-probability units. These differences are not merely normalization: within-prompt rankings can change.

In the fridge context, the direct target scores rank the tested words:

> water > juice > milk > yogurt > ketchup > meat

The array used to generate sampled sets instead ranks them:

> water > milk > juice > yogurt > meat > ketchup

These are deterministic score rankings, **not** individual sampled orderings. The latter cannot be reconstructed from aggregate membership probabilities.

The repository documents a pending correction to how whitespace at the prompt–candidate boundary is scored. Existing saved results have not been regenerated with that correction. This investigation verifies the artifact mismatch but does not establish that whitespace is its sole cause.

The input-matched focus comparison is:

| Predictor | Pearson r |
| --- | ---: |
| Reported direct baseline, separate target scores | .363 |
| Raw query probability from the sampling array | .209 |
| Full-support normalized query probability from that array | .401 |
| Log query probability from that array | .507 |
| Set Top-K from that array | .594 |

Thus Set's advantage is not eliminated by matching the input artifact, but the headline .363 → .594 comparison changes more than the linking rule. Log probability is an exploratory diagnostic of response-scale effects, not a valid exclusion probability and not an additional fitted model. The direct target scores themselves yield r = .602 after taking logs; this also cautions against interpreting probability-scale Pearson alone as evidence of a uniquely necessary mechanism.

An exploratory paired bootstrap, resampling the 16 focus stories and keeping the OOF predictions fixed, gives a Set-minus-normalized-input difference of .193 (95% percentile interval [.080, .300]). The Set-minus-array-log-probability difference is .087 [−.002, .173]. These are not refit CV intervals, do not incorporate participant resampling or Monte Carlo reruns, and are not adjusted for exploratory comparisons.

## 3. The rank evidence supports a narrower, more interesting account

Three different comparisons should not all be called “the ranking.”

1. **Human plausibility ranks versus model ranks of six candidate words in each focus story.** Using the actual sampling-array scores, mean within-story Spearman is .679. Using the separate direct scores, it is .707. These human ranks come from the experiment's prior norming, not the exclusion responses.
2. **Model predictions versus exclusion rates across 30 ordered trigger–query pairs within each focus story.** Mean Spearman is .552 for the sampling-array input and .543 for Set Top-K. The reported direct baseline is .585. Thus Set improves pooled Pearson without improving within-story exclusion ranking. Constant Top-p predictions make one story undefined, so Top-p means use 15 rather than 16 stories.
3. **Model predictions versus exclusion rates across scalar items.** This compares different scales and prompts, not six alternatives sharing one context. It is a different inferential target from (1).

The crucial scalar example is Ronai & Xiang 2022:

| Predictor | Spearman rho |
| --- | ---: |
| Raw query probability from the sampling array | .474 |
| Full-support normalized query probability | .289 |
| Set Top-K | .298 |
| X but not Y, separate scoring artifact | .371 |

Raw probability and its logarithm have the same Spearman correlation. Their Pearson correlations differ sharply: .030 versus .420. A weak probability-scale Pearson correlation therefore does not imply that the scalar alternatives are ranked uninformatively.

Within a prompt, normalization preserves ranks. Across prompts, each prompt has a different normalizer, so normalization can change the ordering of item predictions. In this scalar dataset, some of the association with human exclusion is already attenuated at that step, before the sampled-set mapping. These historical scores require replication after the scoring correction, but they rule out the simple interpretation “there is no scalar ranking signal” for the current outputs. The R&X 2024 ESI condition likewise has raw-array Spearman .474, compared with .324 for Set Top-K.

## 4. Actual examples to build the narrative around

The fridge example was already part of the paper discussion, making it a useful running example rather than an item selected for maximal model fit. In the following table, human rates are for individual trigger–query pairs, not averages over different triggers. These examples illustrate mechanisms and failures; they do not replace aggregate evaluation.

| Context / uttered trigger | Query | Human exclusion | Array-normalized query probability | Set Top-K | Ordering | X but not Y |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fridge / milk | water | 5/6 = .833 | .149595 | 1.000 | .968 | .430 |
| Fridge / milk | yogurt | 1/5 = .200 | .000120 | .038 | .016 | .000305 |
| Ronai–Xiang 2022 / slowed | stopped | 33/40 = .825 | .000056 | .006 | .786 | .859 |
| Ronai–Xiang 2022 / allowed | obligatory | 33/40 = .825 | .000013 | .004 | .032 | .001552 |

For “slowed/stopped,” stopped is more likely than slowed in the sampling distribution, but both are far down the full candidate ranking: stopped ranks 1,706 and slowed 4,174. The saved Set probability is .006, equivalent to 3 memberships in 500 sampled Top-K sets. This is a case where relative ordering carries useful information that a small-prefix membership rule largely discards. It does not show that scalar ranking always fails; here Ordering succeeds much better than Set.

For “allowed/obligatory,” neither Set, Ordering, nor the contrastive frame captures the high human exclusion rate. This counterexample keeps us from presenting X but not Y as a general solution. Both scalar examples are exploratory selections used to contrast failure modes, not representative estimates of typical performance.

Do not infer specific individual sampled orderings from these counts. Use preserved or explicitly regenerated samples if Section 3 needs a sample-by-sample worked example.

## 5. Two further limits on the mechanistic story

**Trigger sensitivity.** Removing each focus story–query pair's mean isolates variation across the five different triggers for that query. Correlation with the remaining human variation is .134 for Ordering, .009 for Conjunction Top-K, and .156 for Disjunction Top-K. Set and the neutral baseline are constant within these groups, so their residual correlation is undefined. This descriptive check suggests that much of the overall association concerns *which query is under consideration*, with more limited evidence that the models capture the effect of the particular uttered trigger. It is not a significance test or a fitted residual model.

**The effect of only.** All 60 ESI/Eonly scale pairs share the same generation prompt, and all 60 Estrong/Eonlystrong pairs do too. Their Set predictions are identical. Mean human exclusion increases from .387 to .684 for ESI → Eonly and from .613 to .886 for Estrong → Eonlystrong. Stripping only and then applying the same linking rule cannot capture these shifts. This is a representational limitation of the current model inputs/mapping, not evidence by itself that alternative generation is wrong.

Finally, correlation is not probability accuracy. Focus Set Top-K's mean log score is −1.569, compared with −.696 for the saved fold-safe dataset-base-rate predictor. Every currently reported focus linking structure has worse log score than that reference. The current results support an association with human judgments, not well-calibrated probabilities or a completed psychological model.

## Proposed Results narrative

Suggested progression: **focus association → inspect generation and mapping separately → scalar limits and frame contrast → explicit scope of the conclusion.** Keep evaluation details in Method and psychological interpretations in Discussion.

### Working draft: From contextual expectedness to exclusion

The linking structures were more strongly correlated with human exclusion judgments in the focus experiment than in the scalar benchmarks. In the focus experiment, Set Top-K produced the highest Pearson correlation (r = .594), followed by Conjunction Top-K (r = .560) and Disjunction Top-K (r = .524). The direct neutral and X-but-not-Y predictors yielded lower correlations of .363 and .263, respectively. This pattern is consistent with the hypothesis that contextual expectations provide useful information about which focus alternatives listeners exclude.

To understand this association, we examined the distributions from which the linking structures were constructed. Across the sixteen focus contexts, the model's ranking of the six tested alternatives correlated with the independently normed human plausibility ranking (mean Spearman rho = .679). Alternatives that were more expected in context also tended to receive higher exclusion rates. However, Set did not improve the ranking of exclusion judgments within each context: mean within-context Spearman correlation was .552 for its input distribution and .543 for Set Top-K. Its stronger Pearson correlation therefore should not be interpreted as evidence that it recovers a better ordering of the alternatives. Rather, the result motivates a distinction between identifying contextually appropriate alternatives and mapping their expectedness onto exclusion predictions.

The scalar results reveal a different limitation of this mapping. X but not Y had the highest Pearson correlation in the Ronai and Xiang (2022) and Pankratz and van Tiel (2021) datasets, but not in the van Tiel et al. (2016) or Gotzner et al. (2018) datasets. Moreover, the weaker scalar performance of the linking structures did not uniformly reflect an absence of information in the input distribution. For Ronai and Xiang (2022), raw query probabilities from the sampling distribution correlated with human exclusion at Spearman rho = .474. This association decreased to .289 after normalization over the candidate vocabulary and was .298 for Set Top-K. Thus a distribution can contain information about variation in scalar exclusion without that information being preserved by the current mapping into sampled-set membership.

Taken together, these findings motivate evaluating alternative generation and alternative exclusion separately. The focus results are consistent with contextual expectedness contributing to exclusion judgments, while the scalar results show that a particular transformation of expectedness need not generalize across datasets. The relative success of the X-but-not-Y frame in some scalar datasets motivates testing what additional information is supplied by an explicit utterance and contrast. It does not yet establish that scalar and focus alternatives require different underlying mechanisms.

**Drafting status:** All numerical sentences above are provisional until a common scoring convention is applied and the analysis is rerun. The direct-versus-linking comparison must explicitly disclose the current artifact mismatch if these versions are circulated. The rank analyses were exploratory. The stronger mechanistic and psychological claims belong in Discussion as hypotheses, not established results.

## Next checks, in order

1. Generate corrected candidate scores in a **new** results directory and ensure neutral direct baselines and all linking rules use the same prompt–candidate score artifact. Preserve historical results. Audit the framed scorer's convention too.
2. Expand the Top-K grid above 100. Every current fold chose the tested maximum. Continue selecting K and p separately by balanced training-fold Set log likelihood in grouped ten-fold CV, then reuse each selected boundary for Conjunction and Disjunction without refitting.
3. Repeat the rank/scale decomposition and the input-matched baseline comparison. Any fitted calibration control must be trained inside the CV folds. Report correlations alongside proper log scores and a fold-safe base-rate reference.
4. After that replication, build a paper figure with an aggregate matched-input contrast and a small set of clearly labeled mechanistic examples. Separate within-context focus comparisons from across-scale scalar comparisons, and retain counterexamples.

Reproduce this diagnostic with the bundled Python runtime and `analyze.py` in this directory. The script needs NumPy and pandas only. It produces CSVs, an SVG diagnostic, and a hash manifest. The PNG is a raster rendering of that SVG; it is a working figure, not a submission-ready result.
