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

# ------- Get the BERT Vocabulary (filtered for only full words) ------
def get_bert_vocab():
    # Get the full vocabulary
    vocab = tokenizer.get_vocab()

    # # Filter the vocabulary for full words
    # filtered_vocab = [token for token in vocab.keys() if 
    #             not token.startswith('[') and 
    #             not token.startswith('<') and 
    #             not token.startswith('##') and
    #             token.isalpha()]  # Ensures the token contains only alphabetic characters

    # Remove special tokens: Such as [CLS], [SEP], [MASK], [PAD], [UNK].
    filtered_vocab = [token for token in vocab.keys() if not token.startswith('[') and not token.startswith('<')]

    # Further filter to remove any other special tokens if needed
    special_tokens = set(tokenizer.all_special_tokens)
    filtered_vocab = [token for token in filtered_vocab if token not in special_tokens]

    return filtered_vocab

# ------- Uniform BERT Set Model ------

def get_set(vocab, set_size):
    return random.sample(vocab, set_size)

def log_likelihood_sampling_sets(df_experimental_data, num_sets):
    current_context = None
    set_log_likelihood = 0
    context_likelihoods = []
    current_context_likelihood = 0
    queries_not_in_distribution = []

    for _, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 

        if context != current_context:
            context_likelihoods.append((current_context, current_context_likelihood, set(queries_not_in_distribution)))
            print(f"Context: {current_context}, \nLog Likelihood: {current_context_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
            current_context_likelihood = 0
            queries_not_in_distribution = []

            distribution = get_bert_vocab()
            current_context = context

            # Sample a bunch of sets from the BERT distribution and store them in all_sampled_sets
            all_sampled_sets = []
            for _ in range(num_sets):
                all_sampled_sets = get_set(distribution, set_size=random.randint(2, len(distribution)))
                all_sampled_sets.append(all_sampled_sets)

        # Skip to the next row if query is not in the predicted tokens of distribution
        # if not any(predicted_token == query for predicted_token in distribution):
        #     queries_not_in_distribution.append(query)
        #     continue

        # Count how many of the sampled sets contain the query 
        count = sum(1 for sampled_set in all_sampled_sets if query in sampled_set)

        # The the proporation of smapled sets that contain the query 
        empirical_probability = count / num_sets

        # Compute likelihoods
        if query_negated == 1:
            prob_query_obs = empirical_probability

        else:
            prob_query_obs = 1 - empirical_probability

        if prob_query_obs > 0:
            set_log_likelihood += np.log(prob_query_obs)
            current_context_likelihood += np.log(prob_query_obs)
        else:   
            # Handle the zero or negative case by assigning a small value
            set_log_likelihood += np.log(1e-10)
            current_context_likelihood += np.log(1e-10)
            # set_log_likelihood += 0
            # current_context_likelihood += 0

        if row.equals(df_experimental_data.iloc[-1]):
            context_likelihoods.append((current_context, current_context_likelihood, set(queries_not_in_distribution)))
            print(f"Context: {current_context}, \nLog Likelihood: {current_context_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n\n")

    # print(set(queries_not_in_distribution))

    return(set_log_likelihood, context_likelihoods)
    # return(set_log_likelihood)

# ------- Uniform BERT Ordering Model ------

def get_ordering(vocab):
    return random.sample(vocab, len(vocab))

def log_likelihood_sampling_ordering(df_experimental_data, num_samples):
    current_context = None
    ordering_log_likelihood = 0
    current_context_likelihood = 0
    context_likelihoods = []
    queries_not_in_distribution = []

    for _, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 
        trigger = row['cleaned_trigger']

        if context != current_context:
            context_likelihoods.append((current_context, current_context_likelihood, set(queries_not_in_distribution)))
            print(f"Context: {current_context}, \nLog Likelihood: {current_context_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
            current_context_likelihood = 0
            queries_not_in_distribution = []

            distribution = get_bert_vocab()
            current_context = context

            all_sampled_orderings = []
            for _ in range(num_samples):
                sampled_ordering = get_ordering(distribution)
                all_sampled_orderings.append(sampled_ordering)
        
        # Skip to the next row if query or trigger is not in the predicted tokens of distribution
        # if not any(predicted_token == query for predicted_token in distribution):
        #     queries_not_in_distribution.append(query)
        #     continue

        # if not any(predicted_token == trigger for predicted_token in distribution):
        #     queries_not_in_distribution.append(trigger)
        #     continue

        count_query_above_trigger = 0
        for ordering in all_sampled_orderings:
            query_index = np.where(ordering == query)[0]
            if query_index.size > 0:
                query_index = query_index[0]
            else:
                query_index = 0

            trigger_index = np.where(ordering == trigger)[0]
            if trigger_index.size > 0:
                trigger_index = trigger_index[0]
            else:
                trigger_index = 0
        
            if query_index < trigger_index:
                count_query_above_trigger += 1

        empirical_probability = count_query_above_trigger / len(all_sampled_orderings)
        # print("Query: " + str(query) + " - Empirical: " + str(empirical_probability))

        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = empirical_probability
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = 1 - empirical_probability

        if prob_query_obs > 0:
            ordering_log_likelihood += np.log(prob_query_obs)
            current_context_likelihood += np.log(prob_query_obs)
        else:   
            # Handle the zero or negative case by assigning a small value
            ordering_log_likelihood += np.log(1e-10)
            current_context_likelihood += np.log(1e-10)

    if row.equals(df_experimental_data.iloc[-1]):
        context_likelihoods.append((current_context, current_context_likelihood, set(queries_not_in_distribution)))
        print(f"Context: {current_context}, \nLog Likelihood: {current_context_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n\n")

    return ordering_log_likelihood, context_likelihoods

# ------- Main ------

total_log_likelihood, context_likelihoods = log_likelihood_sampling_sets(df_experimental_data, num_sets=1000)
# total_log_likelihood, context_likelihoods = log_likelihood_sampling_ordering(df_experimental_data, num_samples=1000)

# ------- Save the Results ------

# Import results data 
df_results = pd.read_csv('../data/all_items_results_fyp.csv')

# Create a dictionary from context_likelihoods for easy lookup
context_likelihoods_dict = {context: likelihood for context, likelihood, _ in context_likelihoods}

# Map the likelihoods to the df_results based on the context
df_results['uniform_set_likelihoods'] = df_results['context'].map(context_likelihoods_dict)

# Save the updated dataframe 
df_results.to_csv('../data/all_items_results_fyp.csv', index=False)