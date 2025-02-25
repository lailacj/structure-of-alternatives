from transformers import AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
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
        return np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        return np.log(1e-10)

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
        return np.log(prob_query_obs)
    else:
        # Handle the zero or negative case by assigning a small value
        return np.log(1e-10)

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
        return np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        return np.log(1e-10)

def disjunction_log_likelihood(samples, query, trigger, query_negated):
    count_disjunction = 0

    for sample in samples:
        sampled_ordering = (sample[0]).tolist()
        sampled_set = sample[1]

        query_index = sampled_ordering.index(query)
        trigger_index = sampled_ordering.index(trigger)

        if ((query in sampled_set) or (query_index > trigger_index)):
            count_disjunction += 1

    empirical_probability = float(count_disjunction) / float(len(samples))

    if query_negated == 1:
        prob_query_obs = empirical_probability
    else:
        prob_query_obs = 1 - empirical_probability

    if prob_query_obs > 0:
        return np.log(prob_query_obs)
    else:   
        # Handle the zero or negative case by assigning a small value
        return np.log(1e-10)

def log_likelihoods(experimental_data, num_reps=500, save_interval=1000):
    ordering_data = []
    set_data = []
    conjunction_data = []
    disjunction_data = []
    queries_not_in_distribution = []

     # get the unique context in the experimental data
    contexts = experimental_data['story'].unique()
    context_samples = {}

    # range(start, stop, step)
    # the start has to be at least three
    # the stop can be at most 30522 (size of bert vocab)
    for set_boundary in range(3, 30522, save_interval): 
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

        # iterate through the experimental data
        for _, row in experimental_data.iterrows():

            # print(f"Row: {row}")

            context = row['story']
            query = row['cleaned_query']
            trigger = row['cleaned_trigger']
            query_negated = row['neg'] 

            # Skip to the next row if query or trigger is not in the predicted tokens of distribution
            if not any(predicted_token == query for predicted_token, _ in distribution):
                queries_not_in_distribution.append((context, query, trigger))
                continue

            if not any(predicted_token == trigger for predicted_token, _ in distribution):
                queries_not_in_distribution.append((context, query, trigger))
                continue

            # get the samples for the context
            samples = context_samples[context]
            ordering_log_likelihood_single_trial = ordering_log_likelihood(samples, query, trigger, query_negated)
            set_log_likelihood_single_trial = set_log_likelihood(samples, query, query_negated)
            conjunction_log_likelihood_single_trial = conjunction_log_likelihood(samples, query, trigger, query_negated)
            disjunction_log_likelihood_single_trial = disjunction_log_likelihood(samples, query, trigger, query_negated)

            ordering_data.append([set_boundary, num_reps, context, trigger, query, ordering_log_likelihood_single_trial])
            set_data.append([set_boundary, num_reps, context, trigger, query, set_log_likelihood_single_trial])
            conjunction_data.append([set_boundary, num_reps, context, trigger, query, conjunction_log_likelihood_single_trial])
            disjunction_data.append([set_boundary, num_reps, context, trigger, query, disjunction_log_likelihood_single_trial])

            # print(f"Context: {context}, Query: {query}, Trigger: {trigger}, Ordering LL: {ordering_log_likelihood_single_trial}, Set LL: {set_log_likelihood_single_trial}, Conjunction LL: {conjunction_log_likelihood_single_trial}, Disjunction LL: {disjunction_log_likelihood_single_trial}")

        # Convert lists to DataFrames
        set_df = pd.DataFrame(set_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
        ordering_df = pd.DataFrame(ordering_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
        conjunction_df = pd.DataFrame(conjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
        disjunction_df = pd.DataFrame(disjunction_data, columns=['set_boundary', 'num_reps', 'context', 'trigger', 'query', 'log_likelihood'])
        queries_not_in_distribution_df = pd.DataFrame(queries_not_in_distribution, columns=['context', 'query', 'trigger'])

        # Save results to CSV incrementally
        set_df.to_csv('../data/set_results.csv', mode='a', header=not pd.io.common.file_exists('../data/set_results.csv'), index=False)
        ordering_df.to_csv('../data/ordering_results.csv', mode='a', header=not pd.io.common.file_exists('../data/ordering_results.csv'), index=False)
        conjunction_df.to_csv('../data/conjunction_results.csv', mode='a', header=not pd.io.common.file_exists('../data/conjunction_results.csv'), index=False)
        disjunction_df.to_csv('../data/disjunction_results.csv', mode='a', header=not pd.io.common.file_exists('../data/disjunction_results.csv'), index=False)
        queries_not_in_distribution_df.to_csv('../data/queries_not_in_distribution.csv', mode='a', header=not pd.io.common.file_exists('../data/queries_not_in_distribution.csv'), index=False)

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
        queries_not_in_distribution.clear()

    return None
    # return (set_data, ordering_data, conjunction_data, disjunction_data, queries_not_in_distribution)

# ------ Main ------

log_likelihoods(df_experimental_data)


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


