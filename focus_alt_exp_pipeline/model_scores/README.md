# Cross-dataset model scores

The cluster scorer writes `hu_rnx_no_frame_qwen_scores.csv` here after all 609
manifest rows finish. An interrupted run leaves a `.partial.csv` checkpoint
that can be continued with `--resume`.

Each final row contains both trigger and query scores:

- exact continuation token IDs and tokenizer tokens
- per-token log probabilities
- summed continuation log probability
- mean per-token log probability
- query-minus-trigger contrasts
- exact model snapshot, tokenizer, dtype, library, manifest-hash, and timestamp
  provenance

The summed log probability is the probability of the complete candidate
surface and is the natural primary expectedness score. The mean per-token score
is retained as a length-normalized diagnostic so the choice can be assessed
before the paper analysis is frozen.
