# Structure of Alternatives

**Work in progress.** Not intended for public use.

Modeling and analysis of the generation of foucs alternatives. Our main claim is that a good model for human next word predicition is a good model for how humans generate foucs alternatives. 

There are some main projects in this repo. 
1) Correlating next word probabilities of different LLMs to the next word probabilites of humans (cloze probability). 
2) Modeling foucs alternative generation. 

Description of repo directories: 
- *archive*: Self explanatory. Includes code, data, figures, write ups, etc that I no longer use but don't want to delete. 
- *code*: All code. 
    - *bert_modeling*: Code for when BERT was the main and only language model we were using. (Not used now).
        - `bert_model.py`: Gets a next word distrubtion from BERT given a prompt. Samples sets and orderings from that distrubtion. Then runs the set, ordering, disjunction, conjunction models. Gets the log likelihood of a trial from the experimental data for each of the set, ordering, disjunction, conjunction models. Outputs result dataframes for each alternative models. 
        - `bert_top_words.py`: Get top next words from BERT given a prompt.
        - `bert_uniform_alternatives_model.py`: Use BERT's vocabulary to get a uniform distrubtion over next words. Run alternative models on this uniform distrubition. 
        - `cloze_prob_model.py`: Run the alternative models on the pesudo cloze probability from our inside the set task. This did not work.
        - `heat_map_plots.py`: Produce heat maps.
        - `ranking_correlations.py`: Calculate Spearman correlation between BERT next word distrubtion and the 6 words from the focus alternatives task. 
        - `results_analysis_aug25.Rmd`: R Markdown code. Used for plots and analysis on BERT output. 
        - `results_analysis.Rmd`: R Markdown code. The older version of `results_analysis_aug25.Rmd`
        - `scatterplots_log_likelihood_sepearman_comparsion.py`: Make scatter plots of pearman_drop vs mean log likelihood per context.
        - `tokenization_script.py`: Generate the BERT tokenization of a word. 
        - `word2Vec_top_words.py`: Word2Vec model. Finds the top N most similar words to a given word based on cosine similarity using the static BERT embedding space. 
        - `wordNet_top_words.py`: Word Net model. Finds the top N most similar words to a given word based on Resnik similarity. 
    - *llm*: Code used for comparing different LLMs to human cloze probability. 
        - *external_cloze_prob_dataset*: Code used for comparing different LLMs to the human cloze probability dataset of Peelle et al (2020). 
            - `llm_external_cloze_prob.py`: Get the probability of a next word for multiple LLMs against cloze dataset.
            - `organizing_data.py`: Reformat the data from Peelle et al (2020) as a long data csv file. 
            - `plots_analysis.py`: Getting plots.
            - `resumable_llm_external_cloze_prob.py`: A resumable of `llm_external_cloze_prob.py`.
        - *inside_the_set*: Code used for comparing different LLMs to the word frequencies from the human stimulus generation experiments on the inside the set task.
            - `correlate_half_cloze_prob.py`: Spilt the participant data in half and do a spearman correlation between the two halfs. Are participants correlated with each other?
            - `llm_clozprob_comparison.py`: Compute spearman correlation between human frequency probabilities and LLM next-word probabilities.
            - `llm_logits_target_words.py`: Get the next word probabilities of the words from our inside the set task for multiple LLMs given a context.
            - `llm_playground.py`: A playground script that I used to figure out how to get next word probabilities from LLMs. 
            - `plots.py`: Getting plots.
            - `split_half_human_llm_spearman.py`: Split half of the human word data and compute spearman correlations between each half and LLM next word probabilities. 
            - `word_freq_to_cloze_prob.py`: Get the pesudo cloze probability for the words in the inside the set task. 
        - `alternative_task_model.py`: Code to run the alternatives negation task on the top model from the next word probability correlations. Work in progress. 
    - *ollama*: Code I used when I was exploring how to use the Ollama framework on Oscar. Don't use this anymore.
    - *prompts*: Code to output LLM prompts in the format I want. 
        - `prompts_script_only.py`: Prompts with 'only' - "I only have "
        - `prompts_script_stim.py`: Prompts without 'only'. Used in the stimulus generation, inside the set experiments. "Sure, I have "
        - `prompts_trigger.py`: Include the trigger word in the prompt. Replace the last "only have " or "only has " clause in prompt with "only have/has {trigger} and [MASK].'"
    - `clean_queries.py`: When the query word has an "a" or "an" in front of it, remove this. 
- *data*: All the data. 
    - *BERT_Word2Vec_WordNet*
        - `BERT_top_words.csv`: The top next words from BERT given a context. The output of `bert_top_words.py`. 
        - `query_positions_results.csv`: The same thing as `bert_top_words.py` but with the ranking of the word included. 
        - `word2Vec_top_words.csv`: The output of `word2Vec_top_words.py`. 
        - `wordNet_top_words.csv`: The output of `wordNet_top_words.py`. 
    - *external_cloze_prob_dataset*
        - `cloze_data.csv`: Data from Peelle et al 2020. Output of `organizing_data.py`
        - `output.md`: Original data file from Peelle et al 2020. 
    - *inside_the_set*: Data for the inside the set task.
        - `Generative_Data_RAW.csv`: Data from the stimulus generation experiment. Inside the set case are rows where `positive = TRUE`.
        - `word_freq_to_cloze_prob.csv`: Output of `word_freq_to_cloze_prob.py`. 
    - *prompts*: All the different prompt files. The output of the prompts code.
    - `sca_dataframe_filtered.csv`: Filtered version of the data. 
    - `sca_dataframe.csv`: All the data from the alternative negation experiment.
- *figures*: A bunch of different plots and figures. 
- *presentations*: All the presentations I have given associated with this project. 
- *results* 
    - *BERT_results*: Results from when BERT was the LM we were primarily interested in. 
    - *fyp_results*: Results I used in my FYP. 
    - *llm* 
        - *external_cloze_prob_dataset*
            - `llm_next_word_from_external_cloze.csv`: The output of `llm_external_cloze_prob.py`. The next word probabilities of the words from Peelle et al, for all tested LLMs. 
            - `mean_spearman_by_llm.csv`: The mean spearman correlation for each LLM. 
            - `per_sentence_spearman.csv`: The spearman correlation for each sentence from Peelle et al, for each LLM. 
        - *inside_the_set*
            - `llm_human_half_spearman_per_split.csv`: Results of spliting half of participant data and spearman correlating to LLMs, for each context, and each split.
            - `llm_human_half_spearman_summary.csv`:  Mean spearman correlation between spliting half of participant data and LLMs; for each LLM.
            - `mean_spearman_by_split.csv`: Spliting half of participant data. Mean of spearman corraltions for each split.
            - `per_context_rank_agreement.csv`: Spearman correlation for each context and each LLM. 
            - `results_llm_next_word_probs.csv`: Main results! Output from `llm_logits_target_words.py`. Shows the next word probability for each LLM, each context, and each word. 
            - `split_half_spearman_by_context.csv`: Results of spliting half of participant data and correlating participants to participants. 
            - `split_half_spearman_summary.csv`: Spliting half of participant data. Mean of spearman corraltions for each context. Mean of `split_half_spearman_by_context.csv`. 
            - `summary_by_llm.csv`: Mean of spearman correlations for each LLM. Mean of `per_context_rank_agreement.csv`.
- *write_ups*: Writings I have done related to this project. 


        


