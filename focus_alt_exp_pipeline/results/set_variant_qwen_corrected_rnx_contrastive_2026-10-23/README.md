# Corrected results with R&X X-but-not-Y

This bundle adds 300 query-only contrastive predictions to the verified corrected parent. Neutral scores, sampled predictions, folds and selected boundaries are reused unchanged. Only framed source fields and copied prediction source-hash references change. See inputs/rnx_contrastive_provenance.json for parent hashes and the extension audit.

Only is omitted: ESI/Eonly and Estrong/Eonlystrong predictions are identical. rnx_contrastive/ contains descriptive matched-item condition changes.

Integration and verification completed October 26, 2026; the directory retains
the October 23 scoring-batch date (successful GPU job 7864638).

The two primary measures are **Spearman correlation** and **summed proper log
score**. Higher values are better for both. Use the following tables and figures
for current model comparisons:

- [Spearman by Hu dataset and R&X condition](linking_structure_tables/spearman_by_dataset_and_linking_structure.csv)
- [Focus within-context Spearman and context means](spearman/SPEARMAN.md)
- [Summed log scores by dataset](linking_structure_tables/log_scores_by_dataset_and_linking_structure.csv)
- [Spearman comparison figure](advisor_summary/four_model_comparison/four_models_spearman_rho.png)
- [Log-score comparison figure](advisor_summary/four_model_comparison/four_models_log_score.png)

Additional artifacts:

- [Offline results viewer](results_viewer.html)
- [Correlation and summed log-score tables](linking_structure_tables/LINKING_STRUCTURE_TABLES.md)
- [Four-model figures and their source values](advisor_summary/four_model_comparison/README.md)
- [Independent verification report](audit_2026-10-26/verification.json)

The default explorer at `../../results_viewer/index.html` also serves this bundle.
The original default export is preserved at
`../../results_viewer/archive/corrected_pre_rnx_2026-10-26.html`; its original
hash remains in the audit ledger, and verification checks the archived bytes.

The audit verified all 427 original file hashes, exact preservation of sampled
predictions and folds, and all 360 embedded neutral distributions. It independently
recomputed Pearson correlation, Spearman correlation and summed proper log score
for all five R&X conditions. Re-run it from any directory with Python and SciPy:

```bash
python /path/to/this/bundle/audit_2026-10-26/verify_results.py
```
