# Absolute-threshold development results

These outputs are a novel-focus integration test for the new absolute
expectedness threshold model. They are not final cross-dataset or paper results.

Run from the repository root with:

```bash
python focus_alt_exp_pipeline/code/evaluate_absolute_threshold.py
```

The evaluator:

- holds out one story/context at a time (16 folds)
- fits one threshold on the other 15 contexts using the Set model's log score
- reuses that threshold for Set, Conjunction, and Disjunction
- evaluates Ordering without a threshold parameter
- computes one held-out probability per item
- reports Pearson and Spearman correlations over 480 item types
- reports log scores over all 3,408 human responses
- compares each structure with a training-fold intercept-only baseline

## Unit-scale development run

The default Gumbel scale is 1. The held-out results are:

| Structure | Pearson r | Spearman rho | Mean log score | Delta vs. intercept |
| --- | ---: | ---: | ---: | ---: |
| Set | 0.550 | 0.592 | -1.325 | -0.627 |
| Ordering | 0.476 | 0.514 | -1.377 | -0.679 |
| Conjunction | 0.542 | 0.594 | -1.423 | -0.725 |
| Disjunction | 0.531 | 0.593 | -1.321 | -0.623 |

The intercept mean log score is -0.698. All four unit-scale models are therefore
overconfident despite positive item-level correlations. No threshold fit reached
its search boundary; the problem is the probability scale, not failed threshold
optimization.

An exploratory fixed-scale check found that scales around 3-7 produce much
better calibration; for example, scale 5 gives Set Pearson r = 0.595 and mean
log score = -0.612. Scale 5 is not adopted as a final value. If the scale becomes
a fitted parameter, it must be learned inside each training fold alongside the
threshold and then tested on held-out groups.

The committed development outputs in this directory predate the standardized
Qwen rescore and should not be cited as final results. The current canonical
input now carries verified Qwen revision
`453ed1575b739b5b03ce3758b23befdb0967f40e`; final results still require the
cross-dataset evaluation rule and joint threshold/scale fitting to be frozen.
