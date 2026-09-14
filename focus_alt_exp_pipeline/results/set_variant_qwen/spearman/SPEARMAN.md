# Within-context Spearman

Word rankings use six tested alternatives and summed neutral continuation log probabilities. Negation rankings use all 30 ordered trigger–query pairs. Ties receive average ranks; constant vectors are undefined. Means weight valid contexts equally; valid/total counts are reported. Boundary selection still uses training log score.

| model | mean_within_context_spearman | valid_contexts | total_contexts | measure | structure | variant |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2-7B | 0.7071428571428572 | 16 | 16 | word_ranking |  |  |
| Qwen2-7B | 0.5850552362852297 | 16 | 16 | negation | No linking structure | direct |
| Qwen2-7B | 0.5212825350937136 | 16 | 16 | negation | X but not Y | direct |
| Qwen2-7B | 0.5564875861968573 | 16 | 16 | negation | conjunction | top_k |
| Qwen2-7B | 0.5622656002104559 | 15 | 16 | negation | conjunction | top_p |
| Qwen2-7B | 0.5298767658369093 | 16 | 16 | negation | disjunction | top_k |
| Qwen2-7B | 0.5186599354607055 | 16 | 16 | negation | disjunction | top_p |
| Qwen2-7B | 0.5017579889485276 | 16 | 16 | negation | ordering | top_k |
| Qwen2-7B | 0.5017579889485276 | 16 | 16 | negation | ordering | top_p |
| Qwen2-7B | 0.5434125445613756 | 16 | 16 | negation | set | top_k |
| Qwen2-7B | 0.5495992196784905 | 15 | 16 | negation | set | top_p |
