"""Summarize participant-level moderation fits from the corrected novel-focus run."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/set_variant_qwen_corrected_2026-09-15/advisor_summary/alignment_moderation'
MODELS = {'no_linking': 'No linking', 'x_but_not_y': 'X but not Y',
          'ordering': 'Ordering', 'set_top_k': 'Set Top-K'}
COLORS = {'no_linking': '#65758B', 'x_but_not_y': '#C57429',
          'ordering': '#2768AC', 'set_top_k': '#21866A'}


def save(fig, stem):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    results = pd.concat([pd.read_csv(OUT / model / 'interaction.csv') for model in MODELS], ignore_index=True)
    order = np.argsort(results.lrt_p.to_numpy())
    adjusted = np.empty(4)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(results.lrt_p.to_numpy()[order] * np.arange(4, 0, -1)))
    results['holm_p'] = adjusted
    results.to_csv(OUT / 'interaction_summary.csv', index=False)
    loo = pd.concat([pd.read_csv(OUT / model / 'leave_one_context_out.csv') for model in MODELS], ignore_index=True)
    assert len(loo) == 64 and loo.interaction.notna().all()
    assert (loo.optimizer_code.astype(str) == '0').all()
    assert loo.messages.fillna('').str.replace('boundary (singular) fit: see help(\'isSingular\')', '', regex=False).eq('').all()
    assert loo.loc[~loo.singular, 'scaled_gradient'].le(.002).all()
    loo.to_csv(OUT / 'leave_one_context_out.csv', index=False)
    boundaries = pd.concat([pd.read_csv(OUT / model / 'boundary_sensitivity.csv')
                            for model in ['ordering', 'set_top_k']], ignore_index=True)
    boundaries.to_csv(OUT / 'boundary_sensitivity.csv', index=False)
    slopes = pd.concat([pd.read_csv(OUT / model / 'simple_slopes.csv').assign(model=model)
                        for model in MODELS], ignore_index=True)
    slopes.to_csv(OUT / 'simple_slopes.csv', index=False)
    alignment = pd.read_csv(OUT / 'context_alignment.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    fig.subplots_adjust(left=.19, right=.82, top=.76, bottom=.25)
    for i, r in results.iterrows():
        ax.errorbar(.1*r.interaction, 3-i, xerr=[[.1*(r.interaction-r.ci_low)], [.1*(r.ci_high-r.interaction)]],
                    color=COLORS[r.model], fmt='o', capsize=5, markersize=8, linewidth=2)
        ax.text(1.05, 3-i, f'{r.holm_p:.4f}', transform=ax.get_yaxis_transform(), va='center')
    ax.text(1.05, 3.65, 'Holm p', transform=ax.get_yaxis_transform(), weight='bold')
    ax.axvline(0, color='#555555', linewidth=1, linestyle='--')
    ax.set(yticks=range(4), yticklabels=list(MODELS.values())[::-1], ylim=(-.65, 3.65),
           xlabel='Change in prediction slope for +0.1 word-ranking Spearman\n(human-response log-odds per 1 SD model probability)')
    ax.grid(axis='x', alpha=.15)
    fig.text(.06, .94, 'Does word-ranking agreement strengthen the prediction–response relationship?', fontsize=15, weight='bold')
    fig.text(.06, .875, 'Ordering supports a positive interaction; estimates for the other models remain uncertain.', fontsize=12)
    fig.text(.06, .055, 'Dots: interaction estimates. Bars: individual 95% Wald intervals. Holm p: likelihood-ratio tests across four models.\n3,408 responses · 213 participant IDs · 16 contexts. Ordering and Set Top-K have boundary random-effect estimates.', fontsize=9, color='#444444')
    save(fig, 'interaction_estimates')

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), sharex=True)
    for ax, (model, label) in zip(axes.flat, MODELS.items()):
        curve = pd.read_csv(OUT / model / 'slope_curve.csv')
        ax.plot(curve.alignment, curve.slope, color=COLORS[model], linewidth=2.5)
        ax.fill_between(curve.alignment, curve.ci_low, curve.ci_high, color=COLORS[model], alpha=.18)
        ax.plot(alignment.alignment, np.full(len(alignment), .035), '|', color='#555555',
                transform=ax.get_xaxis_transform(), markersize=8)
        ax.axhline(0, color='#777777', linewidth=.8, linestyle='--')
        ax.set_title(label, weight='bold', loc='left')
        ax.set_xlim(alignment.alignment.min()-.025, 1.025)
        ax.grid(alpha=.15)
    fig.suptitle('How the fitted prediction slope changes with word-ranking agreement', fontsize=15, weight='bold', y=.98)
    fig.supxlabel('Whole-context word-ranking Spearman', y=.10)
    fig.supylabel('Human-response log-odds slope per 1 SD model probability', x=.015)
    fig.text(.09, .026, 'Bands: pointwise 95% Wald intervals. Rugs: observed context alignment (ties overlap). Panel y-axis scales differ.\nA steeper positive slope indicates stronger association; it does not measure calibration or out-of-sample accuracy.', fontsize=9)
    fig.tight_layout(rect=(.03, .13, 1, .94))
    save(fig, 'slopes_by_alignment')

    table = '\n'.join(f'| {MODELS[r.model]} | {r.interaction:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.lrt_p:.5f} | {r.holm_p:.5f} |'
                      for r in results.itertuples())
    sensitivity_rows = []
    for model, g in loo.groupby('model', sort=False):
        low = g.loc[g.interaction.idxmin()]
        high = g.loc[g.interaction.idxmax()]
        sensitivity_rows.append(f'| {MODELS[model]} | {low.interaction:.3f} (omit {low.omitted_context}) | {high.interaction:.3f} (omit {high.omitted_context}) | {int((g.ci_low > 0).sum())}/16 |')
    ordering = slopes[slopes.model.eq('ordering')].iloc[[0, 2]]
    low, high = ordering.iloc[0], ordering.iloc[1]
    boundary_text = '\n'.join(f'- **{MODELS[r.model]}:** interaction {r.interaction:.3f}, 95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}], unadjusted likelihood-ratio p={r.lrt_p:.5f}.'
                              for r in boundaries.itertuples())
    report = f'''# Does word-ranking agreement moderate model–human negation association?

## Conclusion

The exploratory analysis supports the proposed positive interaction for **Ordering**. Its model-predicted exclusion probability has a stronger positive association with human exclusion responses in contexts with higher human–model word-ranking agreement. No linking and Set Top-K have positive but uncertain interaction estimates. X but not Y has a negative but uncertain estimate. The results therefore do not establish the hypothesis across all four models.

## Data and specification

This analysis uses only the novel-focus study in the corrected September 15 result bundle: **3,408 binary responses, 213 participant IDs, 16 contexts, and 480 ordered trigger–query pairs**. All responses and contexts are retained. The `id` column is treated as participant: each ID occurs once in every context. Source counts and human response rates were verified against the saved pair-level results.

For each model separately, the dependent variable is the participant's exclusion response (`neg`, 0/1). The predictor is the saved model-predicted exclusion probability. The moderator is the context's Spearman correlation between the six reference human word ranks and the corrected neutral Qwen word ranks. Reference human ranks come from `trigger_relevance`; they are not ranks computed from the exclusion outcomes. The same word-ranking agreement measure is used for all four models, including X but not Y.

The binomial logistic mixed-effects model is:

```r
neg ~ prediction_z * alignment_c +
      (1 + prediction_z || participant) +
      (1 + prediction_z || context) +
      (1 | item)
```

`item` identifies an ordered trigger–query pair within a context. Participant and context random intercepts and uncorrelated prediction slopes allow response tendencies and associations to vary; pair intercepts account for repeated responses to a pair. Probability is centered and divided by its SD across the 480 pairs, separately for each model. Alignment is centered at {alignment.alignment.mean():.6f}. This scaling preserves the requested linear interaction; the predictor is **probability**, not its logit or the language model's next-word logits. See `scaling.csv` for all constants.

Unlike the earlier equally weighted correlations across pair-level response rates, this analysis uses individual responses. Pairs with more participant responses contribute more observations. Saved held-out predictions and boundary choices are preserved; the new regression itself is fit to all these responses.

## Primary results

Interaction coefficients below express the change in the human-response log-odds slope per **one SD of model probability and a +1 increase in word-ranking Spearman**. Positive values match the directional hypothesis. Confidence intervals are individual approximate Wald intervals. P values use a two-sided likelihood-ratio comparison with the same random effects and no interaction; Holm adjustment covers the four tests.

| Model | Interaction | 95% CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
{table}

For Ordering, at the 25th percentile of alignment (ρ={low.alignment:.3f}), the fitted probability slope is {low.slope:.2f} log-odds per SD (95% CI {low.ci_low:.2f}–{low.ci_high:.2f}). At the 75th percentile (ρ={high.alignment:.3f}), it is {high.slope:.2f} ({high.ci_low:.2f}–{high.ci_high:.2f}). These are fitted slopes at the observed alignment quartiles, not separate regressions on high- and low-agreement subsets.

The figure rescales interaction estimates to a **+0.1** increase in agreement for readability.

![Interaction estimates](interaction_estimates.png)

![Prediction slopes across agreement](slopes_by_alignment.png)

## Fit checks and sensitivity

All primary optimizers returned status 0. No linking and X but not Y have nonsingular primary fits with satisfactory gradient checks. Ordering's full fit estimates both participant and context prediction-slope variances at zero; Set Top-K's full fit estimates the context prediction-slope variance near zero. These are singular (boundary) fits. Alternative optimizers reproduced their likelihoods. Derivative diagnostics were unavailable for the boundary fits in this lme4 version, so optimizer agreement is reported separately from derivative-based checks. See each model's `diagnostics.json`.

Removing only the zero-variance slope components from each full model, and fitting its no-interaction comparator with the identical simplified random effects, gives:

{boundary_text}

These are sensitivity analyses, not replacements selected by significance. Singular fits can complicate inference; see the official [lme4 singularity documentation](https://lme4.github.io/lme4/reference/isSingular.html) and [convergence guidance](https://lme4.github.io/lme4/reference/convergence.html).

After inspecting primary fits, we also omitted each context in turn and refit the primary full formula, retaining the original scaling constants. This checks whether a single context determines the direction of the interaction. It conditions on the original saved predictions; it does not rerun boundary selection or constitute held-out validation.

| Model | Smallest interaction | Largest interaction | Individual 95% CI above zero |
|---|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

Intervals in this diagnostic are unadjusted and correlated across omissions; their counts are descriptive, not additional hypothesis tests. All selected omission fits returned optimizer status 0 with no remaining non-boundary convergence messages. The No linking fit omitting cut required restarting optimization from an earlier solution to pass gradient checks; all attempts are retained. Full coefficients, flags and optimizer diagnostics are saved in `leave_one_context_out.csv` and each model's diagnostic files.

## What this establishes, and what remains open

- **Supported interpretation:** for Ordering, higher whole-context word-ranking agreement is associated with a stronger positive relationship between predicted exclusion probability and human exclusion responses under this specification.
- **Other models:** the intervals do not establish the direction of moderation. An uncertain interaction does not establish no interaction, and it does not mean the model fails to predict exclusion.
- **Only 16 contexts:** the moderator varies across 16 contexts, even though the dataset contains 3,408 responses. Random context slopes allow between-context slope variation, but the Ordering estimate is on a variance boundary. The asymptotic intervals and p values should be treated as exploratory evidence and checked in additional contexts.
- **Meaning of “better”:** the interaction tests a conditional log-odds association after fitting a regression. It does not directly test higher Pearson/Spearman correlation, better calibration, or better prediction on new responses. Those require separate outcome measures and validation.
- **No causal claim:** agreement was not experimentally manipulated. Other context properties may explain its relationship with model fit. Uncertainty in reference ranks, finite model sampling, and boundary selection is not propagated.
- **No ranking of models from these p values:** all four regressions use the same responses, and their probability SDs differ. A significant interaction in one model and an uncertain interaction in another is not a formal test of their difference.

## Suggested wording for advisors

> In an exploratory participant-level logistic mixed-effects analysis of the novel-focus study, the association between Ordering's exclusion predictions and human exclusion responses became stronger as human–model word-ranking agreement increased. The interaction remained positive when each context was omitted in turn. The corresponding interactions for No linking, X but not Y, and Set Top-K were uncertain. These findings support an alignment-dependent association for Ordering, but do not establish that this pattern holds across linking structures or that improved alignment causes better predictive accuracy.

## Reproduce and inspect

From the repository root, with Python's numpy, pandas and matplotlib and R's lme4 and jsonlite installed:

```sh
python focus_alt_exp_pipeline/code/prepare_alignment_moderation.py
Rscript focus_alt_exp_pipeline/code/fit_alignment_moderation.R focus_alt_exp_pipeline/results/set_variant_qwen_corrected_2026-09-15/advisor_summary/alignment_moderation no_linking x_but_not_y ordering set_top_k
Rscript focus_alt_exp_pipeline/code/fit_alignment_moderation.R focus_alt_exp_pipeline/results/set_variant_qwen_corrected_2026-09-15/advisor_summary/alignment_moderation no_linking x_but_not_y ordering set_top_k --sensitivity
python focus_alt_exp_pipeline/code/summarize_alignment_moderation.py
```

The preparation audit records input hashes and join checks. Per-model folders contain fitted R objects, fixed and random effects, summaries, optimizer diagnostics and R session information. The original analysis plan is in `ANALYSIS_PLAN.md`. Estimation uses [lme4 glmer](https://lme4.github.io/lme4/reference/glmer.html) with a logit link and Laplace approximation. This run used R 4.6.0 and lme4 2.0-6; R reported that the package binary was built under R 4.6.1. A custom library can be supplied through `FOCUS_R_LIBRARY`.
'''
    (OUT / 'README.md').write_text(report)
    print(results[['model', 'interaction', 'ci_low', 'ci_high', 'holm_p']].to_string(index=False))
    print('\nBoundary sensitivity:\n', boundaries.to_string(index=False))
    print('\nLeave-one-context-out:\n' + '\n'.join(sensitivity_rows))


if __name__ == '__main__':
    main()
