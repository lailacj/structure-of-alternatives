# Legacy cross-dataset correlation table

This directory preserves the pre-sampled-prefix correlation table and its
visual render. It is not the active cross-dataset analysis.

These files are regenerated from the older development artifact:

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

In this legacy table, the `Set` column uses the global shared learned-cutoff
model. In each grouped
cross-validation fold, one expectedness threshold and one Gumbel scale are fit
on the other nine folds by balanced item log score. A held-out query's Set
probability is the probability that its noisy latent Qwen expectedness exceeds
that fold's learned threshold. It is not the legacy top-K model.

The active analysis does not use that threshold/Gumbel definition. It samples
weighted orderings without replacement and evaluates separate Top-K and Top-p
prefix sets. Current full-coverage outputs are under:

- `focus_alt_exp_pipeline/results/set_variant_qwen/cv_results/`
- `focus_alt_exp_pipeline/results/set_variant_qwen/linking_structure_tables/`
- `focus_alt_exp_pipeline/results/set_variant_qwen/advisor_summary/`

Use `focus_alt_exp_pipeline/results/set_variant_qwen/linking_structure_tables/LINKING_STRUCTURE_TABLES.md`
for the current cross-dataset correlation and proper-log-score table.
