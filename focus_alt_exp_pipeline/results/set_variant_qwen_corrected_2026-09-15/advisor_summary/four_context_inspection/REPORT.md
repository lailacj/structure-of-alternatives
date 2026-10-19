# Inspecting fridge throw bag and meat

This inspection uses the corrected Qwen2-7B bundle and all 30 ordered trigger–query pairs in each context. `trigger → query` means that the trigger was spoken and the query is the alternative whose exclusion is judged. Human rates aggregate binary responses; the figures show response counts. All fit measures here use the same 30 pairs per context, with equal pair weights. Spearman uses average ranks for ties. Results are descriptive point estimates.

## Files

- [Fridge figure](fridge_inspection.png) · [PDF](fridge_inspection.pdf)
- [Throw figure](throw_inspection.png) · [PDF](throw_inspection.pdf)
- [Bag figure](bag_inspection.png) · [PDF](bag_inspection.pdf)
- [Meat figure](meat_inspection.png) · [PDF](meat_inspection.pdf)
- [All 120 trial comparisons](all_120_trial_comparisons.csv), including exact probabilities, ranks, absolute errors and proper log scores.
- [Context fit measures](context_metrics.csv), [tested word rankings](tested_word_rankings.csv), and [saved leading continuations](saved_candidate_continuations.csv).
- [Source hashes and rank checks](audit.json).

Each figure shows the exact scored prompt, the ten leading saved candidate strings, the six tested alternatives, every human/model prediction, and Set-minus-human residuals. Continuation percentages are sampling weights normalized across 121,301 scored strings, including overlapping one- and two-word candidates. They are not a softmax over only the six experimental alternatives. The `Raw P` column and no-linking predictions use exponentiated whole-continuation log probabilities; these differ from normalized sampling weights. The word reference ranks come from the experiment's `trigger_relevance` field and are separate from ranks of human exclusion rates.

## Fit measures

Higher values are better for all three measures. Context log scores sum all 30 pair scores without dividing by 30; individual-pair examples retain their original scores.

| Context | Model | Pearson r | Spearman ρ | Summed log score |
|---|---|---:|---:|---:|
| Fridge | No linking | 0.5573 | 0.8595 | −89.1236 |
| Fridge | X but not Y | 0.5592 | 0.8162 | −51.0263 |
| Fridge | Ordering | 0.6944 | 0.7690 | −25.1960 |
| Fridge | Set Top-K | 0.8269 | 0.8595 | −30.2143 |
| Throw | No linking | 0.5519 | 0.6834 | −91.1496 |
| Throw | X but not Y | 0.4172 | 0.3232 | −169.9507 |
| Throw | Ordering | 0.7285 | 0.7199 | −33.7612 |
| Throw | Set Top-K | 0.5802 | 0.8549 | −69.4636 |
| Bag | No linking | 0.4999 | 0.8744 | −123.2617 |
| Bag | X but not Y | 0.3798 | 0.7464 | −112.0299 |
| Bag | Ordering | 0.6111 | 0.7052 | −25.5076 |
| Bag | Set Top-K | 0.5358 | 0.8605 | −46.7252 |
| Meat | No linking | 0.4519 | 0.1717 | −86.6556 |
| Meat | X but not Y | 0.3090 | 0.3752 | −86.7926 |
| Meat | Ordering | 0.0679 | 0.1851 | −48.8699 |
| Meat | Set Top-K | 0.3670 | −0.0313 | −43.1548 |

## Fridge

**Context:** A thirsty person is offered something from a fridge. Among the six tested words, both the human reference and Qwen rank water > juice > milk > yogurt > ketchup > meat. The full candidate list begins with “some”, “a”, and “water”; raw noun continuation probabilities consequently need not be large even for plausible answers.

**Why Spearman stays the same:** No linking and Set Top-K have exactly the same rank vector across all 30 pairs, including ties. Both depend only on the query in this comparison. Set changes water from .055845 to 1.000, juice from .010330 to .762, and milk from .002165 to .260. This nonlinear change in spacing raises Pearson from .5573 to .8269 without changing Spearman (.8595). A mere positive linear rescaling would not change either correlation.

**Success:** juice → water has human exclusion 6/6 = 1.000. Set predicts 1.000, versus .055845 for no linking and .852 for Ordering.

**Remaining failure:** water → milk has human exclusion 17/18 = .944. Set predicts .260 and Ordering .032, versus .002165 for no linking. Improving on the baseline does not make these predictions accurate. For meat → yogurt, humans give 10/16 = .625 while Set predicts .006; query-only predictions cannot capture variation across triggers.

**Ordering limitation:** water → juice has human exclusion 22/22 = 1.000, and juice → water has 6/6 = 1.000. Ordering predicts .148 and .852. Each sampled strict ordering places one word before the other, so the two directional probabilities must sum to one. Set permits high predictions in both directions (.762 and 1.000).

## Throw

**Context:** Someone proposes playing catch and asks what is available in a garage. The leading saved continuations are baseball, ball, tennis, basketball, and football. Qwen therefore favors several plausible play-catch continuations outside the six tested alternatives.

**Rank changes:** Among tested words, Qwen ranks football > beach ball > shovel > boomerang > lawnmower > tires. Saved Set probabilities instead rank football (.972) > beach ball (.082) > boomerang (.004) > shovel = lawnmower = tires (all zero). This reversal and the new ties accompany a Spearman gain from .6834 to .8549. Pearson improves only from .5519 to .5802.

**Success:** beach ball → football has human exclusion 5/5 = 1.000. Set predicts .972, Ordering .980, and no linking .067458.

**Remaining failure:** football → beach ball has human exclusion 14/16 = .875. Set predicts .082 and Ordering .020, versus .001687 for no linking. Football → boomerang is also underpredicted: human 13/21 = .619, Set .004, Ordering .002. High Spearman can coexist with large probability errors because it only measures relative ranks.

**Sampling qualification:** Set probabilities are empirical inclusion frequencies from 500 sampled orderings. The .004 boomerang estimate represents two inclusions and the zero estimates represent none. The rank gain should be checked across seeds or with more samples before interpreting it as a stable effect of the linking rule. No resampling was performed in this inspection.

X but not Y is worse than no linking on all three measures in this context.

## Bag

**Context:** Someone needs something to write with, and a friend looks in a handbag. Among tested words, Qwen ranks pen > crayons > chalk > makeup > headphones > cash. The reference human word ranking differs only in the last two positions. The full candidate list begins with “a”, “some”, and “something”; pen is sixth.

**Metric disagreement:** Set raises Pearson from .4999 to .5358 and improves log score substantially, but lowers Spearman from .8744 to .8605. Compared with no linking, the saved Set ranking only swaps cash and headphones: cash receives .002 and headphones zero. This corresponds to one versus zero inclusions in 500 samples. Although that swap matches the reference word ranking, it worsens the correlation with the separate human exclusion judgments. Word ranking and negation ranking are different targets.

**Success:** cash → pen has human exclusion 5/5 = 1.000. No linking predicts .019163; Set predicts .990 and Ordering 1.000.

**Remaining failure:** pen → crayons has human exclusion 15/17 = .882, but Set predicts .068 and Ordering .018. Pen → chalk has human exclusion 18/21 = .857, but Set predicts .014 and Ordering .004. The model recognizes pen as a strong answer but gives too little exclusion probability to other writing implements after pen is spoken.

**Interpretation:** The high human judgments in both directions among pen, crayons, and chalk are consistent with a group of contextually relevant alternatives. This motivates examining set-based explanations, although the current Set Top-K estimates still severely underpredict crayons and chalk. This is an interpretation of these judgments, not a fitted comparison of category-based theories.

## Meat

**Context:** A person explicitly requests meat in ramen. Among tested words, Qwen ranks chicken > tofu > rice > turkey > mushrooms > crab. The reference human order is chicken > turkey > crab > tofu > mushrooms > rice. The full distribution begins with “meat”, “meat in”, and “chicken”; the problem is therefore not a complete failure to recognize the request. It is visible in the relative scores of particular tested alternatives.

**Success:** crab → chicken has human exclusion 5/5 = 1.000. Set predicts .896, versus .082286 for no linking. Such gains help Set improve summed log score from −86.6556 to −43.1548.

**Failures:** chicken → turkey has human exclusion 20/21 = .952, but Set predicts .036 and Ordering .016. Chicken → crab has human exclusion 13/17 = .765, but Set predicts .006 and Ordering zero. Conversely, chicken → tofu has human exclusion 0/5 = 0; Set increases the prediction from .010461 to .254, making it worse.

**Metric disagreement:** Set lowers Pearson from .4519 to .3670 and Spearman from .1717 to −.0313, even while improving log score. It improves absolute probability error for 22/30 pairs and worsens it for 8/30. Neither a majority of improved trials nor an improved log score guarantees a higher correlation.

The saved Set ranking also swaps turkey and mushrooms relative to raw word scores (.036 versus .058). This is another empirical rank difference whose sampling stability remains untested.

## What these cases establish

1. Equal Spearman can have a direct explanation: identical ranks and ties, despite nonlinear changes in numerical spacing. Fridge is an exact example.
2. Unequal Spearman can reflect changed ties and rare-event inclusion estimates. Throw and bag require a sampling-stability check before assigning a theoretical interpretation to small rank changes.
3. Ordering has a structural limitation when observed exclusion is high in both directions. These four contexts all contain such pairs. The observed human proportions are noisy estimates, but the model's sum-to-one constraint is exact.
4. Linking often improves probability predictions without resolving all errors. The next-word scores and the rule mapping them to exclusion probabilities can each contribute to failures.
5. No linking remains competitive on particular measures. It has the highest negation Spearman in bag and ties Set in fridge; it has the highest Pearson in meat. X but not Y worsens all measures in throw.

A useful next diagnostic is to check the stability of Set ranks across sampling runs, while preserving the current results, then compare mutual exclusion among the three contextually relevant words in each context. Uncertainty estimates should account for human response counts and participant dependence where available.

## Reproduction

From the repository root:

```bash
python focus_alt_exp_pipeline/code/inspect_focus_contexts.py
```

The script verifies that the local viewer's source, out-of-fold prediction, grid, and per-prompt array hashes match the corrected run before using the exported candidate lists. It does not access Oscar, change predictions, or select new parameters. This report interprets the generated exports; source hashes and exact prompts are in `audit.json`.
