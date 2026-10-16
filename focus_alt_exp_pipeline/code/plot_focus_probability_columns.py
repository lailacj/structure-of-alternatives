"""Draw No linking, Set Top-K, and human exclusion columns for 16 contexts.

Show selected full-vocabulary continuations beside six saved Set predictions.
No inference, parameter selection, or new Monte Carlo sampling is performed.
"""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
RESULT = PIPELINE / 'results/set_variant_qwen_corrected_2026-09-15'
OUT = RESULT / 'advisor_summary/no_linking_set_columns'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_label(value):
    # Avoid printing a small positive continuation probability as zero.
    if value == 0:
        return '0.000'
    if value < .0001:
        return f'{value:.2e}'
    if value < .01:
        return f'{value:.6f}'
    return f'{value:.4f}'


def prepare():
    sources = [RESULT / 'spearman/word_paired_ranks.csv',
               RESULT / 'spearman/negation_paired_ranks.csv',
               RESULT / 'inputs/source_rows.csv',
               RESULT / 'cv_results/oof_predictions.csv',
               PIPELINE / 'results_viewer/index.html']
    words, pairs, source, oof = [pd.read_csv(p) for p in sources[:4]]
    viewer = json.loads(re.search(r'<script id="results-data" type="application/json">(.*?)</script>',
                                 sources[4].read_text(), re.S).group(1))
    for p in sources[2:4]:
        assert sha(p) in {r['sha256'] for r in viewer['provenance'] if Path(r['path']).name == p.name}
    human_path = PIPELINE / 'human_exp_data/sca_dataframe.csv'
    responses = pd.read_csv(human_path)
    observed = responses.groupby(['story', 'cleaned_trigger', 'cleaned_query']).neg.agg(['mean', 'size', 'sum'])
    assert len(responses) == 3408 and len(observed) == 480 and responses.neg.isin([0, 1]).all()
    all_rows, human_rows, prompts = [], [], {}
    for context, w in words.groupby('context', sort=True):
        assert len(w) == 6 and w.word.nunique() == 6 and set(w.human_rank) == set(range(6))
        s = source[source.dataset_family.eq('novel_focus') & source.context_id.eq(context)]
        assert len(s) == 30 and s.generation_prompt.nunique() == 1
        prompt = next(p for p in viewer['prompts'] if p['frame'] == 'Neutral' and p['contexts'] == [context])
        assert prompt['text'] == s.generation_prompt.iloc[0]
        assert set(s.neutral_score_array_sha256) == {r['sha256'] for r in viewer['provenance']
                                                  if Path(r['path']).name == prompt['id'] + '.log_probs.npy'}
        prompts[context] = prompt['text']
        base = pairs[pairs.context.eq(context) & pairs.structure.eq('No linking structure') & pairs.variant.eq('direct')]
        checked = observed.loc[context].reindex(pd.MultiIndex.from_frame(base[['trigger', 'query']]))
        assert np.allclose(checked['mean'], base.human_rate, rtol=0, atol=1e-12)
        assert np.array_equal(checked['size'], base.human_total)
        for word, trials in base.groupby('query'):
            assert len(trials) == 5 and trials.trigger.nunique() == 5 and word not in set(trials.trigger)
            count = int(trials.human_total.sum())
            excluded = int(round((trials.human_rate * trials.human_total).sum()))
            human_rows.append(dict(context=context, word=word, probability=trials.human_rate.mean(),
                                   trigger_count=5, response_count=count, exclusion_count=excluded,
                                   pooled_response_rate=excluded/count,
                                   trigger_rate_min=trials.human_rate.min(), trigger_rate_max=trials.human_rate.max()))
        for model, structure, variant in [('No linking', 'No linking structure', 'direct'),
                                           ('Set Top-K', 'set', 'top_k')]:
            g = pairs[pairs.context.eq(context) & pairs.structure.eq(structure) & pairs.variant.eq(variant)]
            assert len(g) == 30 and set(g['query']) == set(w.word)
            assert g.groupby('query').size().eq(5).all()
            assert g.groupby('query').model_value.nunique().eq(1).all()
            values = g.groupby('query').model_value.first()
            d = w[['word', 'human_rank', 'model_value']].copy()
            d['probability'] = d.word.map(values)
            d['raw_probability'] = np.exp(d.model_value)
            assert d.probability.between(0, 1).all()
            if model == 'No linking':
                assert np.allclose(d.probability, d.raw_probability, rtol=0, atol=1e-12)
            else:
                assert g.boundary.eq(20).all()
                heldout = oof[oof.analysis_unit_id.isin(g.item_id) & oof.variant.eq('top_k')].set_index('analysis_unit_id')
                assert len(heldout) == 30 and heldout.boundary.eq(20).all()
                assert np.allclose(heldout.set_probability.reindex(g.item_id), g.model_value, atol=1e-12, rtol=0)
                assert np.allclose(d.probability * 500, np.round(d.probability * 500), atol=1e-10)
            d['rank_min'] = d.probability.rank(ascending=False, method='min').astype(int)
            d['rank_average'] = d.probability.rank(ascending=False, method='average')
            d['tied'] = d.probability.duplicated(keep=False)
            d = d.sort_values(['probability', 'raw_probability', 'word'], ascending=[False, False, True])
            d['display_row'] = np.arange(1, 7)
            d['model'] = model
            d['context'] = context
            d['reference_word_rank'] = d.human_rank.astype(int) + 1
            all_rows.append(d.drop(columns=['human_rank', 'model_value']))
    data = pd.concat(all_rows, ignore_index=True)
    assert data.context.nunique() == 16 and len(data) == 192
    windows_path = OUT / 'continuation_windows.json'
    windows = json.loads(windows_path.read_text())
    assert windows['source_rows_sha256'] == sha(sources[2])
    selected = {r['context']: r for r in windows['contexts']}
    assert set(selected) == set(prompts)
    for context, block in selected.items():
        rows = block['candidates']
        positions = [r['vocabulary_position'] for r in rows]
        assert positions == sorted(set(positions)) and set(range(1, 6)).issubset(positions)
        assert sum(r['experimental'] for r in rows) == 6
        s = source[source.dataset_family.eq('novel_focus') & source.context_id.eq(context)]
        assert set(s.neutral_score_array_sha256) == {block['score_array_sha256']}
        d = data[data.context.eq(context) & data.model.eq('No linking')].set_index('word')
        assert {r['word'] for r in rows if r['experimental']} == set(d.index)
        for row in rows:
            assert np.isclose(np.exp(row['log_probability']), row['probability'], atol=1e-15, rtol=1e-12)
            if row['experimental']:
                assert np.isclose(d.loc[row['word'], 'probability'], row['probability'], atol=1e-12, rtol=0)
        for gap in block['gaps']:
            wanted = list(range(gap['after_position']+1, min(gap['before_position'], gap['after_position']+5)))
            assert wanted == gap['shown_positions'] and set(wanted).issubset(positions)
    human = pd.DataFrame(human_rows)
    assert len(human) == 96 and human.probability.between(0, 1).all()
    # Collapse only floating-point noise when assigning tied human ranks.
    human['rank_value'] = human.probability.round(12)
    human['rank_min'] = human.groupby('context').rank_value.rank(ascending=False, method='min').astype(int)
    human['rank_average'] = human.groupby('context').rank_value.rank(ascending=False, method='average')
    human['tied'] = human.duplicated(['context', 'rank_value'], keep=False)
    human = human.sort_values(['context', 'rank_value', 'word'], ascending=[True, False, True])
    human['display_row'] = human.groupby('context').cumcount()+1
    human['model'] = 'Human exclusion'
    sources.extend([windows_path, human_path])
    return data, prompts, sources, selected, human


def draw(context, d, prompt, block, human):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'font.size': 11})
    fig = plt.figure(figsize=(15.5, 13.5), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.045, .965, context.capitalize(), fontsize=20, weight='bold', va='top')
    ax.text(.045, .928, textwrap.fill(prompt.rstrip() + '...', width=166),
            fontsize=10, va='top', linespacing=1.4)
    ax.text(.045, .848, 'No linking', fontsize=16, weight='bold')
    ax.text(.405, .848, 'Set Top-K (K = 20)', fontsize=16, weight='bold')
    ax.text(.725, .848, 'Human exclusion', fontsize=16, weight='bold')
    ax.text(.045, .825, 'Raw continuation probabilities', fontsize=10)
    ax.text(.405, .825, 'Exclusion probabilities: six experimental words', fontsize=10)
    ax.text(.725, .825, 'Mean across five spoken alternatives', fontsize=10)
    for x, rank_title, word_x, p_x in [(.045, 'Position', .10, .34), (.405, 'Rank', .45, .665), (.725, 'Rank', .765, .965)]:
        ax.text(x, .799, rank_title, fontsize=9)
        ax.text(word_x, .799, 'Word', fontsize=9)
        ax.text(p_x, .799, 'Mean rate' if x == .725 else 'Probability', fontsize=9, ha='right')
        ax.plot([x, p_x], [.789, .789], color='black', linewidth=.6)
    y, step, previous = .774, .018, 0
    for row in block['candidates']:
        pos = row['vocabulary_position']
        if pos > previous+1:
            ax.text(.10, y, '...', color='#777777', fontsize=10, va='center')
            y -= step
        experimental = row['experimental']
        weight = 'bold' if experimental else 'normal'
        color = 'black' if experimental else '#666666'
        size = 11.2 if experimental else 10.4
        ax.text(.045, y, f'{pos:,}', fontsize=10, weight=weight, color=color, va='center')
        ax.text(.10, y, row['word'], fontsize=size, weight=weight, color=color, va='center')
        ax.text(.34, y, probability_label(row['probability']), fontsize=10.2, weight=weight,
                color=color, ha='right', va='center', fontfamily='DejaVu Sans Mono')
        y -= step
        previous = pos
    assert y > .087, (context, y)
    for frame, x, wx, px in [(d[d.model.eq('Set Top-K')], .405, .45, .665), (human, .725, .765, .965)]:
        for i, row in enumerate(frame.sort_values('display_row').itertuples()):
            sy = .774-i*.048
            rank = str(row.rank_min)+('=' if row.tied else '')
            ax.text(x, sy, rank, fontsize=11, va='center')
            ax.text(wx, sy, row.word, fontsize=11.2, weight='bold', va='center')
            ax.text(px, sy, f'{row.probability:.3f}', fontsize=11, ha='right', va='center',
                    fontfamily='DejaVu Sans Mono')
    ax.text(.045, .071,
            'Bold words were tested in the experiment. No linking shows the top 5 continuations and up to 4 intervening words per experimental-word gap.\n'
            'No linking positions refer to the full vocabulary; ... marks omissions. Set and human columns are independently ranked among the 6 experimental words; = marks ties.\n'
            'Human rates average the five trigger-specific exclusion rates with equal weights. They summarize across spoken alternatives; the individual rates may differ.\n'
            'Saved corrected Qwen2-7B results; Set uses 500 samples. Zero means 0/500 inclusions. No new sampling.',
            fontsize=8.5, va='top', linespacing=1.5, color='#444444')
    for extension in ['png', 'pdf']:
        fig.savefig(OUT / f'{context}_no_linking_set.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, prompts, sources, selected, human = prepare()
    plotted = []
    for context, d in data.groupby('context', sort=True):
        block = selected[context]
        human_context = human[human.context.eq(context)]
        draw(context, d, prompts[context], block, human_context)
        for i, r in enumerate(block['candidates']):
            plotted.append(dict(context=context, model='No linking', display_row=i+1, **r))
        for r in d[d.model.eq('Set Top-K')].itertuples():
            plotted.append(dict(context=context, model=r.model, word=r.word, probability=r.probability,
                                rank_min=r.rank_min, rank_average=r.rank_average, tied=r.tied,
                                experimental=True, display_row=r.display_row))
        for r in human_context.itertuples():
            plotted.append(dict(context=context, model=r.model, word=r.word, probability=r.probability,
                                rank_min=r.rank_min, rank_average=r.rank_average, tied=r.tied,
                                experimental=True, display_row=r.display_row))
    data.to_csv(OUT / 'experimental_probabilities.csv', index=False)
    human.to_csv(OUT / 'human_exclusion_rates.csv', index=False)
    pd.DataFrame(plotted).to_csv(OUT / 'plotted_probabilities.csv', index=False)
    (OUT / 'provenance.json').write_text(json.dumps({
        'sources': [{'path': str(p.relative_to(PIPELINE)), 'sha256': sha(p)} for p in sources],
        'script_sha256': sha(Path(__file__)),
        'contexts': sorted(prompts), 'prompts': prompts, 'models': ['No linking', 'Set Top-K'],
        'experimental_words_per_context': 6, 'K': 20, 'original_num_reps': 500,
        'new_sampling': False, 'source_prediction_parity_atol': 1e-12,
        'no_linking_selection': 'Top 5 continuations plus all experimental words and first up to 4 intervening words per gap.',
        'no_linking_positions': 'Full 121301-candidate vocabulary sorted by descending score; ties retain original vocabulary order.',
        'set_rank_scope': 'Six experimental words only; ties share the minimum rank.',
        'set_tie_display_order': 'Descending raw continuation probability, then word.',
        'human_column': 'Equal-weight mean of the five distinct trigger-specific exclusion rates per query word.',
        'human_rank_scope': 'Six experimental words; probabilities rounded to 12 decimals only for assigning ties; alphabetical display within ties.',
        'human_source_validation': 'All 480 human pair rates/counts checked against 3408 binary participant responses.',
        'plotted_values': len(plotted),
    }, indent=2) + '\n')
    links = '\n'.join(f'| {c.capitalize()} | [PDF]({c}_no_linking_set.pdf) | [PNG]({c}_no_linking_set.png) |' for c in sorted(prompts))
    (OUT / 'README.md').write_text("""# No linking, Set Top-K and human exclusion: probability lists

Plain three-column figures for all 16 novel-focus contexts. Experimental words
are bold. No colored cards, probability bars, or connecting lines are used.
Each figure is supplied as a vector PDF and a 200-dpi PNG.

| Context | PDF | Image |
|---|---|---|
""" + links + """

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
""")
    print(f'Validated {len(data)} experimental values; plotted {len(plotted)} values; saved 16 PDFs and 16 PNGs to {OUT}')


if __name__ == '__main__':
    main()
