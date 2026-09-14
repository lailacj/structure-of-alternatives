# Cluster task: find and import X-but-not-Y top-50 scores

Check whether existing cluster artifacts contain vocabulary-wide continuation
scores or saved top-50 rankings for the X-but-not-Y prompts. Do not launch new
Qwen scoring or refit any analysis as part of this task.

1. Update the `structure-of-alternatives` checkout from `origin/main`, preserving
   local changes. Activate the existing analysis environment with NumPy.

2. Read `AGENTS.md` and `focus_alt_exp_pipeline/results_viewer/README.md`.
   The authoritative HTML is `focus_alt_exp_pipeline/results_viewer/index.html`.
   Its embedded `results-data` JSON contains `prompts`: select entries whose
   `frame` equals `X but not Y`. There are currently 390 unique framed prompts.
   Use each entry's exact `text`, `id`, and experimental `candidates` as the
   expected inventory. These are supported for the four Hu datasets and novel
   focus, not the five R&X conditions.

3. Search likely existing score/output directories under the project's cluster
   storage, especially `/users/ljohnst7/data/ljohnst7/ngrams`, and the existing
   scoring job scripts/manifests for framed, contrastive, or X-but-not-Y runs.
   Report the actual locations found. Do not assume a directory exists just
   because its name sounds appropriate.

4. Distinguish the artifact types:
   - `model_scores/focus_hu_remaining_qwen_scores.csv` and other experimental
     query-score CSVs contain scores for selected candidates only. They cannot
     establish the vocabulary's top 50.
   - Full-support arrays need their vocabulary order/manifest, exact prompt
     metadata, completed scoring status, and finite continuation log scores.
   - Saved top-50/top-100 tables may also be usable if they identify the exact
     prompt, candidate strings, ranking scope, scoring provenance, and scores.
     Determine whether they rank model tokens or scored word/phrase
     continuations. The viewer uses the latter. If a table lacks full-support
     normalization information, do not invent normalized probabilities.

5. Compare prompt text exactly, allowing only trailing prompt whitespace to
   differ. Do not join on trigger/query strings or context names alone. Report
   matched, missing, and ambiguous prompt counts out of 390. Inspect model
   identity, vocabulary support, and `scoring_boundary_version`. Report whether
   the scores precede the whitespace-boundary fix or use
   `single-space-exact-concat-v1`; do not call legacy scores corrected.

6. If all 390 prompts have compatible arrays, the exporter supports:

   ```bash
   python focus_alt_exp_pipeline/code/build_results_viewer.py \
     --framed-log-probs-dir /actual/path/to/framed_scores \
     --top-candidates 50
   ```

   This preserves the 360 neutral distributions already in the HTML. It matches
   framed `*.meta.json` files by prompt text, so the score filenames do not need
   the viewer's generated `framed_...` IDs. Each matching file needs sibling
   `.log_probs.npy` and `.progress.json` files and the directory's
   `vocab_manifest.json`. Metadata must include `prompt` and `target_vocab_size`;
   progress must mark the `1gram` and `2gram` sources done.

   If vocabulary paths moved, add `--framed-vocab-dir /actual/vocab/directory`.
   To explicitly load a prior populated HTML, add
   `--reuse-distributions-from /path/to/populated/index.html`.
   If also refreshing neutral arrays, pass the separate `--log-probs-dir` and,
   when needed, `--vocab-dir` options. Preserve the original score files.

7. If only partial coverage, ranked tables, a different array format, or ambiguous
   matches are available, report their schemas, examples, counts, and provenance
   before changing the importer. Do not fabricate missing scores, silently drop
   unmatched prompts, rename original artifacts to force a match, or start a
   new inference run.

8. After a compatible import, verify the builder reports 360/360 neutral and
   390/390 framed distributions, 993 analysis units, and 180 matching metric
   cells. Top 50 ranks and probabilities must come from the full support, not
   from renormalizing the displayed subset. Run:

   ```bash
   python -m unittest discover -s focus_alt_exp_pipeline/tests -p test_results_viewer.py
   # If Node.js 18+ is available:
   node --test focus_alt_exp_pipeline/results_viewer/viewer.test.cjs
   ```

Return the artifact inventory, coverage report, scoring-version findings, and
the rebuilt HTML's absolute path and size. Leave the result ready for review;
do not push or publish newly imported cluster results without an explicit
request to do so.
