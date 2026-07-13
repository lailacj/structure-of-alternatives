# Cross-dataset model scores

The cluster scorer writes `hu_rnx_no_frame_qwen_scores.csv` here after all 609
manifest rows finish. An interrupted run leaves a `.partial.csv` checkpoint
that can be continued with `--resume`.

The remaining-frame scorer writes `focus_hu_remaining_qwen_scores.csv` after
all 1,269 focus/Hu rows finish. Its no-frame rows contain both candidate scores;
its X-but-not-Y rows contain the query score only, with trigger and contrast
columns intentionally empty.

Every scored candidate records:

- exact continuation token IDs and tokenizer tokens
- per-token log probabilities
- summed continuation log probability
- mean per-token log probability
- exact model snapshot, tokenizer, dtype, library, manifest-hash, and timestamp
  provenance

Query-minus-trigger contrasts are populated only for no-frame rows, where both
candidates are scored.

The summed log probability is the probability of the complete candidate
surface and is the natural primary expectedness score. The mean per-token score
is retained as a length-normalized diagnostic so the choice can be assessed
before the paper analysis is frozen.
