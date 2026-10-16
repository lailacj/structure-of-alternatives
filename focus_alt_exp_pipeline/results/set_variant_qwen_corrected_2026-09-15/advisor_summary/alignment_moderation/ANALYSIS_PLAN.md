# Analysis plan for word ranking agreement as a moderator

## Hypothesis and scope

Model-predicted exclusion probability has a stronger positive association with human exclusion responses when human and model word rankings agree more closely. Fit separate models for No linking, X but not Y, Ordering, and Set Top-K, using the corrected novel-focus run. Retain all 16 contexts and all 3,408 binary responses. This plan is recorded before inspecting regression coefficients.

## Specification

Binomial logistic mixed-effects regression, with fixed effects for model-predicted exclusion probability, context-level word-ranking Spearman, and their interaction. Use the probability as the predictor, not its logit or next-word log probability. For numerical conditioning, center and divide probability by its SD across the 480 distinct pairs for each model. Center alignment at its equal-context mean without dividing by its SD. This is an affine reparameterization of the requested predictor and moderator; their functional form is unchanged.

Primary formula in lme4:

```r
neg ~ prediction_z * alignment_c +
      (1 + prediction_z || participant) +
      (1 + prediction_z || context) +
      (1 | item)
```

Participant IDs each occur once per context. Item is the ordered trigger–query pair nested within context. Uncorrelated random intercepts and prediction slopes allow participant and context differences in response tendency and prediction association. Item intercepts account for repeated judgments of the same pair. Context random slopes are especially relevant because alignment varies only across 16 contexts. The primary analysis conditions on the saved word ranks, finite-sample structure predictions, and held-out boundary choices.

## Estimation and diagnostics

Use lme4 glmer with binomial logit link and Laplace approximation. Fit full and no-interaction models with identical random effects. Report interaction estimates, approximate Wald 95% intervals, a two-sided likelihood-ratio test of the interaction, and Holm-adjusted p values across the four models. The directional hypothesis requires a positive interaction; a negative estimate does not support it even if its two-sided test is small. Wald intervals are individual-model intervals, not simultaneous intervals.

Record optimizer codes, warnings, scaled gradients, Hessian eigenvalues, and random-effect variances. If a fit is singular, report the boundary estimate explicitly. Retry numerical convergence problems with an alternative optimizer; do not choose a model according to its p value. If random-effect components are estimated at zero, a documented simplification may be examined as a sensitivity analysis; do not silently replace the primary model.

Report simple prediction slopes at the observed alignment quartiles and a plot of the slope as a continuous function of alignment. Limit plotted alignment to the observed range. These slopes are on the human-response log-odds scale. A stronger slope is not by itself better calibration, a higher Pearson/Spearman correlation, or better out-of-sample prediction.

## Limits

There are only 16 independent context-level alignment values. The analysis is exploratory and motivated by prior inspection of these data. The interaction is an association, not a causal effect. Human-response variability is modeled, while uncertainty in reference word rankings, finite model sampling and CV selection is not propagated. The separate model fits reuse the same human responses; differences in p values are not tests of differences between models. A refitted logistic relationship evaluates association after statistical fitting, not the original unadjusted log score of a linking model.
