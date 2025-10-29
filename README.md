# Structure of Alternatives

**Work in progress.** Not intended for public use.

Modeling and analysis of the generation of foucs alternatives. Our main claim is that a good model for human next word predicition is a good model for how humans generate foucs alternatives. 

There are some main projects in this repo. 
1) Correlating next word probabilities of different LLMs to the next word probabilites of humans (cloze probability). 
2) Modeling foucs alternative generation. 

Description of repo directories: 
- `archive`: Self explanatory. Includes code, data, figures, write ups, etc that I no longer use but don't want to delete. 
    - `code`: All code. 
        - `bert_modeling`: Code for when BERT was the main and only langugae model we were using. (Not used now).
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
        - `llm`: Code used for comparing different LLMs to human cloze probability. 
            - `external_cloze_prob_dataset`: Code used for comparing different LLMs to the human cloze probability dataset of Peelle et al (2020). 
                - `llm_external_cloze_prob.py`: Get the probability of a next word for multiple LLMs against cloze dataset.
                - `organizing_data.py`: Reformat the data from Peelle et al (2020) as a long data csv file. 
                - `plots_analysis.py`: Getting plots.
                - `resumable_llm_external_cloze_prob.py`: A resumable of `llm_external_cloze_prob.py`.
            - 


