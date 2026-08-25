# Cross-dataset correlation table

This directory contains the machine-readable correlation table and its visual
render. Both are regenerated from:

`focus_alt_exp_pipeline/results/big_table_development/big_table_correlations.csv`

Build once:

```bash
./refresh_table.sh
```

Keep the outputs synchronized while results are changing:

```bash
./refresh_table.sh --watch
```

The watcher rebuilds both outputs whenever the canonical correlation CSV
changes. Stop it with `Ctrl-C`.

The `Set` column uses the global shared learned-cutoff model. In each grouped
cross-validation fold, one expectedness threshold and one Gumbel scale are fit
on the other nine folds by balanced item log score. A held-out query's Set
probability is the probability that its noisy latent Qwen expectedness exceeds
that fold's learned threshold. It is not the legacy top-K model.
