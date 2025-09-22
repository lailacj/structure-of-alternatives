from importlib.metadata import distribution
from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import random
import torch
import seaborn as sns

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")

# ------- Cleaning and Sorting the Experimental Data ------

# Function that removes 'a' or 'an' from the front of the items
def clean_word(word):
    word = word.lower()
    if word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word

df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')
df_experimental_data['cleaned_trigger'] = df_experimental_data['trigger'].apply(clean_word)
df_experimental_data['cleaned_query'] = df_experimental_data['query'].apply(clean_word)
df_experimental_data = df_experimental_data.sort_values(by='story')

# df_prompts = pd.read_csv("../data/prompts_with_trigger_AND.csv")
df_prompts = pd.read_csv("../data/prompts_BERT.csv")

# ------ Helper Functions -------

# Get the input prompt for BERT based on the context.
def get_prompt_for_context(df_prompts, context):
    matching_row = df_prompts[df_prompts['story'] == context]
    if not matching_row.empty:
        return matching_row['prompt'].values[0]
    else:
        return None

# ------ Get next token predictions from BERT ------

# Returns all next tokens and their probabilities.
def get_next_word_probability_distribution(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    mask_token_logits = logits[0, mask_token_index, :].squeeze()
    probs = torch.softmax(mask_token_logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(range(len(mask_token_logits)))

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, probs.tolist()))

# ------ Log Likelihood Calculation ------

def get_ordering(probability_distribution):
    tokens, probabilities = zip(*probability_distribution)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    ordered_tokens = np.random.choice(tokens, size=len(tokens), replace=False, p=probabilities)
    return ordered_tokens

def ordering_log_likelihood(samples, query, trigger, query_negated):
    count_query_above_trigger = 0

    for sample in samples:
        sampled_ordering = (sample[0]).tolist()

        # do not need this try / except block if there is a good check of query and trigger in the log_likelihoods function
        try:
            query_index = sampled_ordering.index(query)
        except ValueError:
            query_index = -1 
        try:
            trigger_index = sampled_ordering.index(trigger)
        except ValueError:
            trigger_index = -1 

        if query_index != -1 and trigger_index != -1 and query_index < trigger_index:
            count_query_above_trigger += 1

    empirical_probability = float(count_query_above_trigger) / float(len(samples))

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)

    return log_likelihood, empirical_probability, prob_query_obs

def set_log_likelihood(samples, query, query_negated):
    count_query_in_set = 0

    for sample in samples:
        sampled_set = sample[1]

        if query in sampled_set:
            count_query_in_set += 1

    empirical_probability = float(count_query_in_set) / float(len(samples))

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def conjunction_log_likelihood(samples, query, trigger, query_negated):
    count_conjunction = 0

    for sample in samples:
        sampled_ordering = (sample[0]).tolist()
        sampled_set = sample[1]

        if query in sampled_set:
            query_index = sampled_ordering.index(query)
            trigger_index = sampled_ordering.index(trigger)
            if query_index < trigger_index:
                count_conjunction += 1

    empirical_probability = float(count_conjunction) / float(len(samples))

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def disjunction_log_likelihood(samples, query, trigger, query_negated):
    count_disjunction = 0

    for sample in samples:
        sampled_ordering = (sample[0]).tolist()
        sampled_set = sample[1]

        query_index = sampled_ordering.index(query)
        trigger_index = sampled_ordering.index(trigger)
        
        if ((query in sampled_set) or (query_index < trigger_index)):
            count_disjunction += 1

    empirical_probability = float(count_disjunction) / float(len(samples))

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def always_negate_log_likelihood(query_negated):
    empirical_probability = 1

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs
    
def never_negate_log_likelihood(query_negated):
    empirical_probability = 0

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        log_likelihood = np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def log_likelihoods_by_context(experimental_data, num_reps=500, save_interval=5):
    ordering_data = []
    set_data = []
    conjunction_data = []
    disjunction_data = []
    always_negate_data = []
    never_negate_data = []
    queries_not_in_distribution = []

     # get the unique context in the experimental data
    contexts = experimental_data['story'].unique()
    context_samples = {}

    print("In log likelihoods function")

    # range(start, stop, step)
    # the start has to be at least three
    # the stop can be at most 30522 (size of bert vocab)
    for set_boundary in range(3, 300, save_interval): 
        print(f"Set Boundary: {set_boundary}")

        for context in contexts:
            prompt = get_prompt_for_context(df_prompts, context)
            distribution = get_next_word_probability_distribution(prompt)
            context_samples[context] = []

            for i in range(0, num_reps):
                sampled_ordering = get_ordering(distribution)
                sampled_set = sampled_ordering[:set_boundary]
                context_samples[context].append((sampled_ordering, sampled_set))
                print(f"Completed {set_boundary} {context} {i}")

        # Sort by probability (descending) and take the top 100
        top_100 = sorted(distribution, key=lambda x: x[1], reverse=True)[:100]
        # Format as "token1:0.1234; token2:0.0987; ..."
        formatted_top_100 = "; ".join(f"{token}:{prob:.4f}" for token, prob in top_100)

        # iterate through the experimental data
        for _, row in experimental_data.iterrows():

            # print(f"Row: {row}")

            context = row['story']
            query = row['cleaned_query']
            trigger = row['cleaned_trigger']
            query_negated = row['always_negate'] 

            # Skip to the next row if query or trigger is not in the predicted tokens of distribution
            if not any(predicted_token == query for predicted_token, _ in distribution):
                queries_not_in_distribution.append((context, query, trigger))
                continue

            if not any(predicted_token == trigger for predicted_token, _ in distribution):
                queries_not_in_distribution.append((context, query, trigger))
                continue

            # get the samples for the context
            samples = context_samples[context]
            ordering_log_likelihood_single_trial, ordering_empirical_p, ordering_prob_query_obs = ordering_log_likelihood(samples, query, trigger, query_negated)
            set_log_likelihood_single_trial, set_empirical_p, set_prob_query_obs = set_log_likelihood(samples, query, query_negated)
            conjunction_log_likelihood_single_trial, conjunction_empirical_p, conjunction_prob_query_obs = conjunction_log_likelihood(samples, query, trigger, query_negated)
            disjunction_log_likelihood_single_trial, disjunction_empirical_p, disjunction_prob_query_obs = disjunction_log_likelihood(samples, query, trigger, query_negated)
            always_negate_log_likelihood_single_trial, always_negate_empirical_p, always_negate_prob_query_obs = always_negate_log_likelihood(query_negated)
            never_negate_log_likelihood_single_trial, never_negate_empirical_p, never_negate_prob_query_obs = never_negate_log_likelihood(query_negated)

            ordering_data.append([set_boundary, num_reps, context, trigger, query, query_negated, ordering_log_likelihood_single_trial, ordering_empirical_p, ordering_prob_query_obs, formatted_top_100])
            set_data.append([set_boundary, num_reps, context, trigger, query, query_negated, set_log_likelihood_single_trial, set_empirical_p, set_prob_query_obs, formatted_top_100])
            conjunction_data.append([set_boundary, num_reps, context, trigger, query, query_negated, conjunction_log_likelihood_single_trial, conjunction_empirical_p, conjunction_prob_query_obs, formatted_top_100])
            disjunction_data.append([set_boundary, num_reps, context, trigger, query, query_negated, disjunction_log_likelihood_single_trial, disjunction_empirical_p, disjunction_prob_query_obs, formatted_top_100])
            always_negate_data.append([set_boundary, num_reps, context, trigger, query, query_negated, always_negate_log_likelihood_single_trial, always_negate_empirical_p, always_negate_prob_query_obs, formatted_top_100])
            never_negate_data.append([set_boundary, num_reps, context, trigger, query, query_negated, never_negate_log_likelihood_single_trial, never_negate_empirical_p, never_negate_prob_query_obs, formatted_top_100])

            # print(f"Context: {context}, Query: {query}, Trigger: {trigger}, Ordering LL: {ordering_log_likelihood_single_trial}, Set LL: {set_log_likelihood_single_trial}, Conjunction LL: {conjunction_log_likelihood_single_trial}, Disjunction LL: {disjunction_log_likelihood_single_trial}")

        # Convert lists to DataFrames
        set_df = pd.DataFrame(set_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        ordering_df = pd.DataFrame(ordering_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        conjunction_df = pd.DataFrame(conjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        disjunction_df = pd.DataFrame(disjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        always_negate_df = pd.DataFrame(always_negate_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        never_negate_df = pd.DataFrame(never_negate_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed', 'top_100_from_distribution'])
        queries_not_in_distribution_df = pd.DataFrame(queries_not_in_distribution, columns=['context', 'query', 'trigger'])

        # Save results to CSV incrementally
        set_df.to_csv('../results/set_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/set_results_data_is_always_negate_aug2025.csv'), index=False)
        ordering_df.to_csv('../results/ordering_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/ordering_results_data_is_always_negate_aug2025.csv'), index=False)
        conjunction_df.to_csv('../results/conjunction_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/conjunction_results_data_is_always_negate_aug2025.csv'), index=False)
        disjunction_df.to_csv('../results/disjunction_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/disjunction_results_data_is_always_negate_aug2025.csv'), index=False)
        always_negate_df.to_csv('../results/always_negate_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/always_negate_results_data_is_always_negate_aug2025.csv'), index=False)
        never_negate_df.to_csv('../results/never_negate_results_data_is_always_negate_aug2025.csv', mode='a', header=not pd.io.common.file_exists('../results/never_negate_results_data_is_always_negate_aug2025.csv'), index=False)
        queries_not_in_distribution_df.to_csv('../results/queries_not_in_distribution_july2025.csv', mode='a', header=not pd.io.common.file_exists('../results/queries_not_in_distribution_july2025.csv'), index=False)

        print(set_df.head())
        print(ordering_df.head())
        print(conjunction_df.head())
        print(disjunction_df.head())
        print(queries_not_in_distribution_df.head())

        # Clear lists to free memory
        set_data.clear()
        ordering_data.clear()
        conjunction_data.clear()
        disjunction_data.clear()
        always_negate_data.clear()
        never_negate_data.clear()
        queries_not_in_distribution.clear()

    return None
    # return (set_data, ordering_data, conjunction_data, disjunction_data, queries_not_in_distribution)

def log_likelihoods_by_context_trigger(
    experimental_data: pd.DataFrame,
    num_reps: int = 500,
    save_interval: int = 5,
    results_dir: str = '../results'
):
    prompt_name = "I only have [trigger] and [MASK]."

    # prepare output accumulators
    collectors = {
        'set':    [],
        'ordering': [],
        'conjunction': [],
        'disjunction': [],
        'always_negate': [],
        'never_negate': []
    }
    missing = []

    # build a fast lookup for prompts
    prompt_lookup = {
        (r.story, r.trigger): r.prompt_with_mask
        for r in df_prompts.itertuples(index=False)
    }

    # unique (story,trigger) pairs in your experimental set
    ctx_tr_pairs = experimental_data[['story','trigger']].drop_duplicates().values

    # main loop over different set_boundaries
    for set_boundary in range(3, 300, save_interval):
        print(f"=== set_boundary = {set_boundary} ===")

        # Step 1: for each (story,trigger) get distribution & pre-sample
        samples = {}
        for story, trigger in ctx_tr_pairs:
            key = (story, trigger)
            prompt = prompt_lookup.get(key)
            if prompt is None:
                raise KeyError(f"No prompt found for {key}")

            dist = get_next_word_probability_distribution(prompt)
            # store both the raw distribution and the drawn samples
            sims = []
            for i in range(num_reps):
                ordering = get_ordering(dist)
                sims.append((ordering, ordering[:set_boundary]))
            samples[key] = (dist, sims)

        # Step 2: iterate through each trial in your experimental_data
        for row in experimental_data.itertuples(index=False):
            story, query, trigger, cleaned_trigger, query_negated = (
                row.story,
                row.cleaned_query,
                row.trigger,
                row.cleaned_trigger,
                row.neg
            )
            key = (story, trigger)
            dist, sims = samples[key]

            # skip if query/trigger not in vocab
            if query not in {tok for tok,_ in dist} or cleaned_trigger not in {tok for tok,_ in dist}:
                missing.append(key)
                continue

            # compute each LL
            ll_ordering, empirical_p_ordering = ordering_log_likelihood(sims, query, cleaned_trigger, query_negated)
            ll_set, empirical_p_set = set_log_likelihood(sims, query, query_negated)
            ll_conj, empricial_p_conj = conjunction_log_likelihood(sims, query, cleaned_trigger, query_negated)
            ll_disj, empricial_p_disj = disjunction_log_likelihood(sims, query, cleaned_trigger, query_negated)
            ll_always_neg, empricial_p_always_neg = always_negate_log_likelihood(query_negated)
            ll_never_neg, empricial_p_never_neg = never_negate_log_likelihood(query_negated)

            base = [prompt_name, set_boundary, num_reps, story, trigger, query, query_negated]
            collectors['ordering'].append(base + [ll_ordering, empirical_p_ordering])
            collectors['set'].append       (base + [ll_set, empirical_p_set])
            collectors['conjunction'].append(base + [ll_conj, empricial_p_conj])
            collectors['disjunction'].append(base + [ll_disj, empricial_p_disj])
            collectors['always_negate'].append(base + [ll_always_neg, empricial_p_always_neg])
            collectors['never_negate'].append(base + [ll_never_neg, empricial_p_never_neg])

        # Step 3: dump to CSVs
        os.makedirs(results_dir, exist_ok=True)
        for name, rows in collectors.items():
            df = pd.DataFrame(rows, columns=[
                'prompt', 'set_boundary','num_reps','context','trigger','query','neg','log_likelihood', 'empirical_probability'
            ])
            path = f"{results_dir}/{name}_with_trigger_AND_results.csv"
            df.to_csv(path, mode='a',
                      header=not os.path.exists(path),
                      index=False)
            collectors[name].clear()

        # missing queries
        pd.DataFrame(missing, columns=['story','trigger']) \
          .to_csv(f"{results_dir}/queries_not_in_dist.csv",
                  mode='a', header=not os.path.exists(f"{results_dir}/queries_not_in_dist.csv"),
                  index=False)
        missing.clear()

    print("Done.")



# ------ Main ------

# print("Starting log likelihood calculations...")

# # add a column to df_experimental_data called 'always_negate' that is 1 for all rows
# df_experimental_data['always_negate'] = 1

# log_likelihoods_by_context(df_experimental_data)

# ------ Example Usage ------
# Uncomment the following lines to run an example usage of the log likelihood functions

# context = "cold"
# query = "tea"
# trigger = "sink"
# query_negated = 0

# context_samples = {}
# prompt = get_prompt_for_context(df_prompts, context)
# distribution = get_next_word_probability_distribution(prompt)
# context_samples[context] = []

# for i in range(0, 500):
#     sampled_ordering = get_ordering(distribution)
#     sampled_set = sampled_ordering[:15]
#     context_samples[context].append((sampled_ordering, sampled_set))

# samples = context_samples[context]

# ordering_log_likelihood_single_trial, ordering_empirical_p = ordering_log_likelihood(samples, query, trigger, query_negated)
# print(f"Ordering Log Likelihood: {ordering_log_likelihood_single_trial}, Empirical Probability: {ordering_empirical_p}")

# ordering_log_likelihood_single_trial, ordering_empirical_p = ordering_log_likelihood(samples, query, trigger, query_negated)
# set_log_likelihood_single_trial, set_empirical_p = set_log_likelihood(samples, query, query_negated)
# print(f"Ordering Log Likelihood: {ordering_log_likelihood_single_trial}, Empirical Probability: {ordering_empirical_p}")
# print(f"Set Log Likelihood: {set_log_likelihood_single_trial}, Empirical Probability: {set_empirical_p}")        

# print the first 10 samples in samples. but only the first 20 tokens in each sample
# print("First 10 samples:")
# for i, sample in enumerate(samples[:10]):
#     sampled_ordering, sampled_set = sample
#     print(f"Sample {i+1}:")
#     print(f"  Ordering: {sampled_ordering[:20]}")
#     print(f"  Set: {sampled_set[:20]}")
#     print()


# ------ Getting BERT next word probabilities for a certain context ------


all_results = []

for specific_context in df_prompts['story'].unique():
    prompt = get_prompt_for_context(df_prompts, specific_context)
    distribution = get_next_word_probability_distribution(prompt)

    # Sort distribution by probability (descending)
    distribution_sorted = sorted(distribution, key=lambda x: x[1], reverse=True)

    # Get queries for this context
    queries = df_experimental_data[df_experimental_data['story'] == specific_context]['cleaned_query'].unique()

    # Prepare results
    results = []
    for query in queries:
        positions = [i for i, (token, _) in enumerate(distribution_sorted) if token == query]
        if positions:
            pos = positions[0]
            prob = distribution_sorted[pos][1]
            results.append({'context': specific_context, 'query': query, 'position': pos, 'probability': prob})
        else:
            results.append({'context': specific_context, 'query': query, 'position': None, 'probability': None})

    # Add top 10 next words
    for i, (token, prob) in enumerate(distribution_sorted[:10]):
        results.append({'context': specific_context, 'query': token, 'position': i, 'probability': prob})

    # Convert to DataFrame
    query_positions_df = pd.DataFrame(results)

    # Keep unique queries
    query_positions_df = query_positions_df.drop_duplicates(subset=['query'])

    # Sort by position
    query_positions_df = query_positions_df.sort_values(by='position')

    all_results.append(query_positions_df)

# Combine all contexts
final_df = pd.concat(all_results, ignore_index=True)

# Save to CSV
final_df.to_csv("../data/query_positions_results.csv", index=False)

print("Saved results to query_positions_results.csv")


# ------ Not used code ------
# set_data, ordering_data, conjunction_data, disjunction_data, queries_not_in_distribution = log_likelihoods(df_experimental_data)

# set_df = pd.DataFrame(set_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
# ordering_df = pd.DataFrame(ordering_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
# conjunction_df = pd.DataFrame(conjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
# disjunction_df = pd.DataFrame(disjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])

# queries_not_in_distribution_df = pd.DataFrame(queries_not_in_distribution, columns=['context', 'query', 'trigger'])

# print(set_df.head())
# print(ordering_df.head())
# print(conjunction_df.head())
# print(disjunction_df.head())
# print(queries_not_in_distribution_df.head())

# # Save DataFrames to CSV files
# set_df.to_csv('../data/set_results.csv', index=False)
# ordering_df.to_csv('../data/ordering_results.csv', index=False)
# conjunction_df.to_csv('../data/conjunction_results.csv', index=False)
# disjunction_df.to_csv('../data/disjunction_results.csv', index=False)
# queries_not_in_distribution_df.to_csv('../data/queries_not_in_distribution.csv', index=False)


    # contexts = experimental_data['story'].unique()
    # # context_samples = {}

    # for context in contexts:
    #     prompt = get_prompt_for_context(df_prompts, context)
    #     distribution = get_next_word_probability_distribution(prompt)
    #     # context_samples[context] = []

    #     context_specific_exp_data = experimental_data[experimental_data['story'] == context]

    #     for subject in range(0, num_subjects):
    #         sampled_ordering = get_ordering(distribution)
    #         set_boundary = random.randint(0, len(distribution) - 1)
    #         sampled_set = sampled_ordering[:set_boundary]
    #         # context_samples[context].append((sampled_ordering, set_boundary))

    #         for _, row in context_specific_exp_data.iterrows():
    #             query = row['cleaned_query']
    #             trigger = row['cleaned_trigger']
    #             query_negated = row['neg'] 


