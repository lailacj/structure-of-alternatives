# Archived absolute-threshold/Gumbel analysis

This directory preserves the August 2026 absolute-expectedness-threshold model,
its Gumbel-noise calibration, tests, development tables, configuration, and
reproducibility snapshot. It is no longer part of the active analysis.

The active replacement is the sampled-prefix set analysis:

- Top-K: the first K words of each sampled ordering;
- Top-p: the shortest sampled-ordering prefix whose original normalized
  candidate probabilities reach p;
- K and p selected separately in grouped training folds by balanced Set log
  likelihood and then evaluated out of fold.
