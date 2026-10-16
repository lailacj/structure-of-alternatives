# No linking, Set Top-K and human exclusion: probability lists

Plain three-column figures for all 16 novel-focus contexts. Experimental words
are bold. No colored cards, probability bars, or connecting lines are used.
Each figure is supplied as a vector PDF and a 200-dpi PNG.

| Context | PDF | Image |
|---|---|---|
| Bag | [PDF](bag_no_linking_set.pdf) | [PNG](bag_no_linking_set.png) |
| Bakery | [PDF](bakery_no_linking_set.pdf) | [PNG](bakery_no_linking_set.png) |
| Beach | [PDF](beach_no_linking_set.pdf) | [PNG](beach_no_linking_set.png) |
| Cold | [PDF](cold_no_linking_set.pdf) | [PNG](cold_no_linking_set.png) |
| Cut | [PDF](cut_no_linking_set.pdf) | [PNG](cut_no_linking_set.png) |
| Fridge | [PDF](fridge_no_linking_set.pdf) | [PNG](fridge_no_linking_set.png) |
| Gym | [PDF](gym_no_linking_set.pdf) | [PNG](gym_no_linking_set.png) |
| Hot | [PDF](hot_no_linking_set.pdf) | [PNG](hot_no_linking_set.png) |
| Mall | [PDF](mall_no_linking_set.pdf) | [PNG](mall_no_linking_set.png) |
| Mask | [PDF](mask_no_linking_set.pdf) | [PNG](mask_no_linking_set.png) |
| Meat | [PDF](meat_no_linking_set.pdf) | [PNG](meat_no_linking_set.png) |
| Restaurant | [PDF](restaurant_no_linking_set.pdf) | [PNG](restaurant_no_linking_set.png) |
| Salad | [PDF](salad_no_linking_set.pdf) | [PNG](salad_no_linking_set.png) |
| Science | [PDF](science_no_linking_set.pdf) | [PNG](science_no_linking_set.png) |
| Throw | [PDF](throw_no_linking_set.pdf) | [PNG](throw_no_linking_set.png) |
| Transport | [PDF](transport_no_linking_set.pdf) | [PNG](transport_no_linking_set.png) |

## What is displayed

The No linking column shows the top five continuations, all six experimental
words, and the first up to four intervening words between consecutive experimental
words. If fewer than four occur in a gap, all are shown. Ellipses explicitly mark
omitted stretches. Numbers are positions in the full 121,301-candidate vocabulary,
including one- and two-word continuations. Equal scores retain original vocabulary
order; positions are not average-tie ranks. Probabilities are raw exponentiated
continuation scores, not probabilities renormalized over the displayed candidates.

The Set Top-K column shows only the six experimental words, sorted by saved
exclusion probability. Ranks are among these six words; `=` marks equal values.
Tied rows are ordered by raw continuation probability, then word. The two columns
therefore show different candidate sets; an apparent change in list position
must not be interpreted as a change in full-vocabulary rank.

The Set values are exactly the saved held-out K=20 results from 500 sampled
orderings. Neither displayed model conditions on the spoken trigger. Zero Set
probability means 0/500 inclusions, not a proven true zero. No resampling or
refitting was performed. The exact neutral prompt is shown with a display ellipsis.

The human column displays the equal-weight mean of the five trigger-specific
human exclusion rates for each query word. For example, the displayed rate for
water in fridge averages the observed exclusion rates after juice, milk, yogurt,
ketchup and meat were spoken. Each trigger contributes one fifth, regardless
of its number of participant responses, matching the equal-pair weighting used
in the earlier fit summaries. This is a descriptive context-level summary, not
a judgment after a particular utterance. The five underlying rates can differ.
The human column is independently sorted among the six experimental words;
ties use minimum ranks and alphabetical display order. These six averaged rates
do not replace the 30 pair-level observations used in the reported correlations.

## Data and verification

- `continuation_windows.json`: small export of only the needed candidate scores
  from the original corrected arrays on Oscar; includes array and vocabulary hashes,
  selection rules and per-gap availability. Full arrays remain on Oscar.
- `plotted_probabilities.csv`: every displayed candidate/probability, with full
  vocabulary position for No linking and six-word rank for Set.
- `experimental_probabilities.csv`: all 192 experimental word/model values,
  including the unchanged values from the original six-word figures.
- `human_exclusion_rates.csv`: all 96 plotted human means, five-trigger counts,
  participant response counts, underlying rate ranges and ranks. Pooled response
  rates are included for auditing; the plots use the equal-trigger means.
- `provenance.json`: input/script hashes, exact prompts and display conventions.
- `verification.json`: numerical, text and PDF rendering checks.

The plot builder verifies all six experimental probabilities against saved results,
Set parity with held-out K=20 predictions, prompt and array hashes, the top-five
selection, and every intervening-word selection. The export reads the existing
arrays without changing them. Original scientific results are unchanged.
Human pair rates and counts are independently verified against the raw binary
participant responses before computing the 96 displayed means.

## Reproduce

The saved small export allows local regeneration without accessing Oscar:

```sh
python focus_alt_exp_pipeline/code/plot_focus_probability_columns.py
```

To refresh the small export where the original arrays reside:

```sh
python focus_alt_exp_pipeline/code/export_focus_column_continuations.py --arrays /path/to/qwen_set_variant_log_probs_single_space --sources /path/to/corrected/inputs/source_rows.csv > continuation_windows.json
```

Place that JSON in this folder before rebuilding. Python dependencies: numpy,
pandas and matplotlib. PDF rendering checks use Poppler.
