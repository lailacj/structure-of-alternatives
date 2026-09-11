**Mask Top-p diagnostic — 2026-09-11**

“Mask” occupies **65.2243% of the probability normalized over the scored candidate support**. At Top-p 0.6, a sampled prefix therefore ends exactly when “mask” appears. The six experimental alternatives have little chance of appearing before it. Ordering instead compares a query with its trigger: both bandana and wallet are much more probable than napkins, even though neither is competitive with mask. This explains the saved Set zeros alongside high Ordering predictions, including Ordering's overprediction for wallet.

All original inputs were preserved; no model settings were changed and Qwen was not loaded or rerun. The [saved mask grid extract](tables/saved_mask_grid_authoritative.csv) is authoritative. Fresh sampling below is a **diagnostic replication**, not an exact reproduction of the original grid.

Validation passed with no detected mismatch. The array is float32, shape `(121301,)`, with 121,301 finite scores. The vocabulary has 98,509 unigrams at offset 0 and 22,792 bigrams at offset 98,509; file lengths, progress counts, completed flags, metadata sizes, and contiguous offsets agree. All 121,301 normalized candidate strings are unique and nonempty. All six targets are present, finite, and strictly positive after normalization. Our mapping agrees exactly with the prediction implementation's loader. These checks establish alignment with the current manifest and vocabulary files; the saved metadata does not provide historical vocabulary content hashes. [Validation details](validation.json)

The metadata prompt matches all 30 mask source rows and the prompt manifest, and its SHA-256 prefix reproduces `prompt_7ebf909e4b947d98da36`. Its exact text is the following, with **one trailing space** after `my`:

> You and your friend Carly go grocery shopping together. As you're about to enter the supermarket, you reach into your pocket and realize, 'I forgot to bring my mask!' Carly looks through her bag and responds, 'Sure, I have my

We use only the full-support array for probabilities, converting its saved log scores to float64 exactly as [build_set_variant_prediction_grid.py](../../code/build_set_variant_prediction_grid.py) does:

```python
log_probs = np.asarray(saved_array, dtype=np.float64)
weights = np.exp(log_probs - log_probs.max())
probabilities = weights / weights.sum()
```

The implementation filters finite scores first; here all scores are finite, so this leaves the full support unchanged. `exp(log_probs)` gives the **unnormalized continuation probabilities**, whose sum is 0.0188088320. For mask, the saved log score is −4.400766373, its unnormalized continuation probability is 0.0122679345, and its normalized sampling probability is 0.652243290. These are different quantities. In particular, 65.22% describes the candidate-support sampling distribution, not the model's unnormalized continuation probability. The separate direct-baseline score columns in `source_rows.csv` were not used.

The next largest candidates are masks (3.6797%), face (2.8658%), surgical (1.1946%), disposable (1.1946%), blue (0.7713%), and double (0.7713%). Thus concentration on competing continuations is directly observed. Unigrams receive 99.8331% of normalized mass; all bigrams together receive 0.1669%. [Top 100 with full scores, sources, ranks, and cumulative mass](tables/top_100_candidates.csv); [mass plot](plots/probability_mass.png)

| Mass threshold | Highest-ranked candidates needed | Attained mass |
|---|---:|---:|
| 50% | 1 | 65.2243% |
| 60% | 1 | 65.2243% |
| 70% | 3 | 71.7698% |
| 80% | 18 | 80.0127% |
| 90% | 184 | 90.0101% |
| 95% | 824 | 95.0012% |

These are descending-rank statistics, not sampled-prefix lengths. Tied probabilities are ranked by ascending vocabulary index for deterministic reporting. All six alternatives come from `vocab_1gram.txt`; indices below are zero-based, ranks one-based. Their total normalized mass is just 0.0006046213 (0.0604621%). The [alternative CSV](tables/experimental_alternatives.csv) also supplies saved log scores, unnormalized continuation probabilities, exact source lines, and expected inclusion counts.

| Alternative | Vocabulary index | Rank | Normalized sampling probability | Cumulative mass through rank | Exact Top-p 0.6 inclusion probability |
|---|---:|---:|---:|---:|---:|
| bandana | 82753 | 134 | 0.0003010474 | 0.887305809 | 0.0004613439 |
| handkerchief | 12192 | 220 | 0.0001713529 | 0.906963125 | 0.0002626442 |
| napkins | 33223 | 9968 | 0.0000007368 | 0.995473696 | 0.0000011296 |
| gloves | 9948 | 486 | 0.0000667312 | 0.934133430 | 0.0001022999 |
| wallet | 19646 | 594 | 0.0000527887 | 0.940489557 | 0.0000809276 |
| candy | 10717 | 1862 | 0.0000119642 | 0.970675242 | 0.0000183429 |

The actual mechanism calls `rng.choice(..., replace=False, p=probabilities)` to retain 32,768 entries from each weighted ordering. It takes the shortest prefix whose cumulative **original** normalized probability reaches the boundary, including the crossing candidate. It does not sort candidates by probability. Required targets beyond the retained prefix receive a consistent relative ordering through exponential race clocks; those suffix positions preserve pairwise relations, not full-vocabulary absolute ranks. The diagnostic imports the implementation's `_sample_prefixes`, `_target_positions`, and `_probability_records` directly.

At 0.6, mask alone exceeds the boundary, while all other candidates combined have mass 0.347756710, below the boundary. Consequently, for any other candidate Y,

\[
P(Y\text{ included at }0.6)=P(Y\text{ precedes mask})
=\frac{w_Y}{w_Y+w_{\rm mask}}.
\]

Bandana's expected inclusion count in 500 orderings is only 0.231; wallet's is 0.0405. The chance that at least one of the six precedes mask in one ordering is 0.000926129. The chance that **none of the six enters any of 500 independent prefixes is 0.629217**. Thus the saved zeros are unsurprising finite-sample outcomes, not true zero probabilities. The 30 items share six query inclusion events from one ordering bank; they are not 30 independent zero findings.

The diagnostic uses 500 orderings, seed 7, PCG64, NumPy 1.26.4, and the original retained prefix size 32,768. Every retained ordering reaches 0.95; the minimum retained mass is 0.999514230. Every sampled prefix was checked for shortest-prefix cutoff correctness. The original code uses one shared RNG across sorted prompts, so resetting seed 7 for mask alone produces a different sample.

| Top-p | Mean length | Median | 5th–95th percentiles | Min–max |
|---|---:|---:|---:|---:|
| 0.6 | 1.456 | 1 | 1–3 | 1–7 |
| 0.7 | 9.528 | 9 | 4–18 | 3–26 |
| 0.8 | 56.872 | 56 | 42–74.05 | 36–93 |
| 0.9 | 429.458 | 427.5 | 397–467 | 379–505 |
| 0.95 | 1553.878 | 1555 | 1499.95–1604 | 1460–1635 |

Percentiles use NumPy's default linear interpolation. At 0.6, lengths 1 through 7 occurred 339, 112, 37, 9, 1, 1, and 1 times. [Full length distributions](tables/prefix_length_distribution.csv); [all 500 lengths per boundary](tables/prefix_lengths_all_500.csv); [plots](plots/sampled_prefixes.png)

Each cell below is **diagnostic inclusion frequency / authoritative saved Set probability**. Diagnostic frequencies use denominator 500.

| Alternative | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---:|---:|---:|---:|---:|
| bandana | 0 / 0 | .004 / .004 | .060 / .060 | .586 / .628 | .996 / .996 |
| handkerchief | 0 / 0 | .006 / .006 | .026 / .046 | .432 / .412 | .960 / .972 |
| napkins | 0 / 0 | 0 / 0 | 0 / 0 | 0 / .004 | .010 / .014 |
| gloves | 0 / 0 | .002 / .002 | .012 / .008 | .200 / .190 | .762 / .740 |
| wallet | .002 / 0 | .004 / .002 | .008 / .012 | .148 / .138 | .666 / .652 |
| candy | 0 / 0 | .002 / 0 | .004 / 0 | .052 / .034 | .240 / .238 |

Wallet entered one diagnostic 0.6 prefix, illustrating why fresh Monte Carlo results must not replace the saved zeros. The most frequent 0.6 occupants were mask 500/500, masks 21/500, face 12/500, blue and double each 10/500, surgical 9/500, and disposable 8/500. Occupant frequencies need not sum to one because prefixes can contain several candidates. [Complete occupant table](tables/p06_prefix_occupants.csv)

Representative 0.6 prefixes below give `(original probability; cumulative mass)`, rounded to six decimals. Ordering IDs are zero-based; examples are the first observed prefixes of lengths 1–4, with an additional length-5 example in the [CSV](tables/representative_p06_prefixes.csv).

- Ordering 0: mask (.652243; .652243).
- Ordering 7: blue (.007713; .007713) → mask (.652243; .659956).
- Ordering 16: clean (.002352; .002352) → tablecloth (.000040; .002393) → mask (.652243; .654636).
- Ordering 90: masks (.036797; .036797) → colorful (.000065; .036862) → face (.028658; .065519) → mask (.652243; .717763).

For the worked examples, X is the trigger napkins and Y the query. Ordering predicts exclusion when Y precedes X. Array-derived shifted weights are `w_napkins = 1.12964693273e-6`, `w_bandana = 4.61556842726e-4`, and `w_wallet = 8.09341254679e-5`. Weighted sampling without replacement gives

\[
P(Y\text{ precedes napkins})=\frac{w_Y}{w_{\rm napkins}+w_Y}.
\]

| Example | Human exclusion | Theoretical Ordering | Saved Ordering | Diagnostic Ordering | Saved Set at 0.6 |
|---|---:|---:|---:|---:|---:|
| napkins → bandana | 5/5 | 0.997558505 | 0.996 | 0.998 | 0 |
| napkins → wallet | 0/4 | 0.986234523 | 0.988 | 0.982 | 0 |

Bandana has 408.6 times napkins' weight; wallet has 71.65 times napkins' weight. Both can follow mask (and miss the selected prefix) while preceding napkins. Their theoretical pairwise probabilities closely match the saved Monte Carlo estimates. Ordering's success for bandana does not imply that relative probability captures exclusion generally: it assigns wallet nearly certain exclusion despite 0/4 observed exclusions. It has no requirement that the query clear an absolute inclusion threshold or satisfy the contextual functional relation. This is an observed prediction mismatch; the four responses do not establish a population exclusion probability of exactly zero. [All 30 pairwise calculations](tables/pairwise_ordering_comparison.csv); [worked-example plot](plots/worked_examples.png)

The [original grid](../../results/set_variant_qwen/prediction_grid.csv) confirms zero Set and Conjunction on all 30 mask items at 0.6, with Disjunction equal to Ordering. This follows from `Conjunction = in_set & ordering` and `Disjunction = in_set | ordering`. [evaluate_set_variant_grid.py](../../code/evaluate_set_variant_grid.py) selects boundaries by mean Set log score within each training dataset, then averages across datasets, excluding the held-out fold. The [saved selections](../../results/set_variant_qwen/cv_results/fold_selections.csv) choose 0.6 in all ten folds; mask belongs to fold 9. These larger-boundary diagnostics show sensitivity only. They neither select a new boundary from mask outcomes nor establish a held-out improvement.

The verified explanation is **concentration on mask plus very small target probabilities plus the selected cutoff**. Missing vocabulary entries, nonfinite scores, underflow, incorrect offsets, and a sorted-nucleus substitution do not explain these results. No inconsistency was found in the prediction or cutoff mechanism for this prompt.

Vocabulary representation remains relevant to interpretation. Entries are exact candidate strings, not semantic classes: mask, masks, face, and phrases such as handkerchief mask are separate events. The scorer sums subtoken log probabilities without an end-of-word condition; unigram and bigram continuation events can overlap. Normalizing these scores creates a distribution over candidate entries, not a partition of all possible complete answers. The local tokenizer splits bandana into 2 tokens, handkerchief into 3, napkins into 2, and gloves, wallet, candy into 1 each. Token length and representation may affect scores, but this audit does not isolate their causal contributions. [Related strings](tables/related_candidate_strings.csv); [tokenization table](tables/tokenizer_boundary_audit.csv)

There is also a **verified current-code boundary construction issue with an unresolved numerical effect**. [precompute_qwen_vocab_log_probs.py](../../code/precompute_qwen_vocab_log_probs.py), `_prepare_prompt_state` (line 287), retains one trailing space, while `_score_continuation_log_prob` (line 337) separately tokenizes a leading-space candidate. The metadata's local tokenizer confirms their concatenated IDs decode to `Sure, I have my  mask` with two spaces, unlike ordinary single-space concatenation. This was checked using tokenizer files only. It warrants a separate scoring audit; the saved metadata does not pin the historical scorer source, and no inference was run to measure how another boundary construction would change scores. It therefore cannot be claimed as the established cause of the concentration. The saved array remains usable for this requested diagnosis of the existing grid. [Tokenizer audit](tokenizer_audit.json)

Exact score inputs are [prompt_7ebf909e4b947d98da36.log_probs.npy](/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs/prompt_7ebf909e4b947d98da36.log_probs.npy), [prompt_7ebf909e4b947d98da36.meta.json](/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs/prompt_7ebf909e4b947d98da36.meta.json), [prompt_7ebf909e4b947d98da36.progress.json](/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs/prompt_7ebf909e4b947d98da36.progress.json), and [vocab_manifest.json](/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs/vocab_manifest.json). Ordered vocabulary inputs are [vocab_1gram.txt](/users/ljohnst7/data/ljohnst7/ngrams/set_variant_qwen_support/vocab_1gram.txt), followed by [vocab_2gram.txt](/users/ljohnst7/data/ljohnst7/ngrams/set_variant_qwen_support/vocab_2gram.txt). Prompt/item/count provenance comes from [source_rows.csv](../../scoring_manifests/set_variant_qwen/source_rows.csv) and [prompts.csv](../../scoring_manifests/set_variant_qwen/prompts.csv). The [postprocessing wrapper](../../cluster/run_set_variant_postprocessing.sh) supplies the original sampling parameters. [input_provenance.json](input_provenance.json) records exact paths and SHA-256 hashes for all inputs, implementation files, and the local tokenizer; every input hash was unchanged after the run.

To regenerate the numerical diagnostics, CSVs, and PNG/PDF plots from the repository root:

```bash
/users/ljohnst7/data/ljohnst7/oscar_jobs/venv/llm_nextword_env/bin/python \
  focus_alt_exp_pipeline/diagnostics/mask_top_p_2026-09-11/diagnose_mask.py
```

Use `--output-dir /path/to/new/diagnostic-directory` to retain this diagnostic too. Dependencies and numerical summary are recorded in [summary.json](summary.json). The [script](diagnose_mask.py) contains the validation and sampling checks; this report supplies the interpretation. Plots are available as both PNG and PDF in `plots/`.
