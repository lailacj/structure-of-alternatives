# Within-context Spearman

Word rankings use six tested alternatives and summed neutral continuation log probabilities. Negation rankings use all 30 ordered trigger–query pairs. Ties receive average ranks; constant vectors are undefined. Means weight valid contexts equally; valid/total counts are reported. Boundary selection still uses training log score.

| model | mean_within_context_spearman | valid_contexts | total_contexts | measure | structure | variant |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2-7B | 0.6984228473198827 | 16 | 16 | word_ranking |  |  |
| Qwen2-7B | 0.5837382395176375 | 16 | 16 | negation | No linking structure | direct |
| Qwen2-7B | 0.5212825350937136 | 16 | 16 | negation | X but not Y | direct |
| Qwen2-7B | 0.5690937938402522 | 16 | 16 | negation | conjunction | top_k |
| Qwen2-7B | 0.5646375198184855 | 15 | 16 | negation | conjunction | top_p |
| Qwen2-7B | 0.5284888621050545 | 16 | 16 | negation | disjunction | top_k |
| Qwen2-7B | 0.530611158370048 | 16 | 16 | negation | disjunction | top_p |
| Qwen2-7B | 0.5130660773065275 | 16 | 16 | negation | ordering | top_k |
| Qwen2-7B | 0.5130660773065275 | 16 | 16 | negation | ordering | top_p |
| Qwen2-7B | 0.5665701692690844 | 16 | 16 | negation | set | top_k |
| Qwen2-7B | 0.5534343833212425 | 15 | 16 | negation | set | top_p |
