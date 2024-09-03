from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pandas as pd
import pdb
import random
import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')

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

# ------ Getting next word predictions from BERT ------

# Return the top k next tokens and their probabilities.
# NOT BEING USED!
def get_top_k_predictions(text, top_k):
    inputs = tokenizer(text, return_tensors="pt")

    # Get logits 
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    mask_token_logits = logits[0, mask_token_index, :].squeeze()
    top_k_indices = torch.topk(mask_token_logits, top_k).indices
    top_k_logits = mask_token_logits[top_k_indices]
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, top_k_probs.tolist()))

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

# ------ Set Model with BERT (Version 1 - NOT BEING USED) ------
# The probability a query is in a set is equal to the probability of query being the next word. 

def prob_query_in_set(probability_distribution, query):
    try:
        for token, probability in probability_distribution:
            if token == query:
                return probability
        
        # If the token is not found, raise an error
        raise ValueError(f"Query '{query}' is not in BERT probability distribution")
    
    except ValueError as e:
        print(e)
        raise 

# Log likelihood computation
def log_likelihood_set(df_experimental_data, df_prompts):
    current_context = None
    set_log_likelihood = 0

    for index, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 

        # Get the logits (probability_distribution) for every context. 
        if context != current_context:
            prompt = get_prompt_for_context(df_prompts, context)
            probability_distribution = get_next_word_probability_distribution(prompt)
            current_context = context

        prob_set = prob_query_in_set(probability_distribution, query)

        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = prob_set * 1 + (1-prob_set) * 0
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = prob_set * 0 + (1-prob_set) * 1

        if prob_query_obs == 0:
            set_log_likelihood += 0 
        else:   
            set_log_likelihood += np.log(prob_query_obs)

    return(set_log_likelihood)

# ------ Set Model with BERT (Version 2) ------
# Here we empricially find the proability that a query and trigger are in the same set

# Sample a set based on the distribution of tokens outputted by BERT
def get_set(probability_distribution, set_size):
    words, probabilities = zip(*probability_distribution)
    
    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    sampled_indices = np.random.choice(len(words), size=set_size, replace=False, p=probabilities)
    sampled_set = [words[i] for i in sampled_indices]
    
    return sampled_set

def log_likelihood_sampling_sets(df_experimental_data, df_prompts, num_sets):
    current_context = None
    set_log_likelihood = 0

    for _, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 
        trigger = row['cleaned_trigger']

        if context != current_context:
            # Get the BERT input prompt and BERT output distribution for each context
            prompt = get_prompt_for_context(df_prompts, context)
            distribution = get_next_word_probability_distribution(prompt)
            current_context = context

            # Sample a bunch of sets from the BERT distribution and store them in all_sampled_sets
            all_sampled_sets = []
            for _ in range(num_sets):
                all_sampled_sets = get_set(distribution, set_size=random.randint(1, len(distribution)))
                all_sampled_sets.append(all_sampled_sets)

        # Count how many of the sampled sets contain the query and trigger
        count = sum(1 for sampled_set in all_sampled_sets if query in sampled_set and trigger in sampled_set)

        # The the proporation of smapled sets that contain the query and trigger
        empirical_probability = count / num_sets

        # Compute likelihoods
        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = empirical_probability * 1 + (1-empirical_probability) * 0
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = empirical_probability * 0 + (1-empirical_probability) * 1

        if prob_query_obs == 0:
            set_log_likelihood += 0 
        else:   
            set_log_likelihood += np.log(prob_query_obs)

    return(set_log_likelihood)

# ------- Empirical Experiment (Sets) --------

# Would the probability that a query gets negated by sampling a bunch of sets equal the 
# proability that BERT outputs? 

def sampling_sets_empirical_exp(df_experimental_data, df_prompts, runs):
    current_context = None
    results = []

    for index, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']

        # For each unique story, get the distribution of logits from BERT and sample sets
        if context != current_context:
            # Get the distribution of all logits from BERT
            prompt = get_prompt_for_context(df_prompts, context)
            distribution = get_next_word_probability_distribution(prompt)
            current_context = context

            # Sample a bunch of sets and store them in all_sampled_sets
            all_sampled_sets = []
            for _ in range(runs):
                all_sampled_sets = get_set(distribution, set_size=10)
                all_sampled_sets.append(all_sampled_sets)

        # Count how many of the sampled sets contain the query
        count = sum(1 for sampled_set in all_sampled_sets if query in sampled_set)

        # CHANGE! TRIGGER AND QUERY need to be in same set!! 

        # Propotion of sampled sets that have the query in the set
        empirical_probability = count / runs
        print(f"Proportion of sampled sets containing '{query}': {empirical_probability}")

        bert_probability = prob_query_in_set(distribution, query)
        print(f"BERT probability of '{query}': {bert_probability}")

        results.append({
            'query': query,
            'empirical_probability': empirical_probability,
            'bert_probability': bert_probability
        })
    
    # Convert the results to a DataFrame
    df_empirical_exp_results = pd.DataFrame(results)

    return df_empirical_exp_results

# ------- Ordering Model with BERT --------

def get_ordering(probability_distribution):
    tokens, probabilities = zip(*probability_distribution)
    ordered_tokens = []
    
    # While loop that samples one token based on its probability then removes that token from
    # the tokens list then resamples a token and so on. 
    while tokens:
        # Sample one token based on its probability then append it to the ordered_tokens list
        sampled_token = random.choices(tokens, weights=probabilities, k=1)[0]
        ordered_tokens.append(sampled_token)
        
        # Remove the sampled token and its probability from the list to ensure we sample without replacment 
        index = tokens.index(sampled_token)
        tokens = tokens[:index] + tokens[index+1:]
        probabilities = probabilities[:index] + probabilities[index+1:]

    # Another way to do this is below but this might not be a perfect simulation of sampling without replacement.
    # sampled_tokens = random.choices(tokens, weights=probabilities, k=len(tokens))
    # ordered_tokens = list(dict.fromkeys(sampled_tokens))
    
    return ordered_tokens

def log_likelihood_ordering(df_experimental_data, df_prompts, num_samples):
    current_context = None
    ordering_log_likelihood = 0

    for _, row in df_experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 
        trigger = row['cleaned_trigger']

        if context != current_context:
            # Get the BERT input prompt and BERT output distribution for each context
            prompt = get_prompt_for_context(df_prompts, context)
            distribution = get_next_word_probability_distribution(prompt)
            current_context = context

            all_sampled_orderings = []
            for _ in range(num_samples):
                sampled_ordering = get_ordering(distribution)
                all_sampled_orderings.append(sampled_ordering)

        count_query_above_trigger = 0
        for ordering in all_sampled_orderings:
            query_index = ordering.index(query)
            trigger_index = ordering.index(trigger)
        
            if query_index < trigger_index:
                count_query_above_trigger += 1

        empirical_probability = count_query_above_trigger / len(all_sampled_orderings)

        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = empirical_probability * 1 + (1-empirical_probability) * 0
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = empirical_probability * 0 + (1-empirical_probability) * 1

        if prob_query_obs == 0:
            ordering_log_likelihood += 0 
        else:   
            ordering_log_likelihood += np.log(prob_query_obs)

    return ordering_log_likelihood



num_sets = 1000
likelihood = log_likelihood_sampling_sets(df_experimental_data, df_prompts, num_sets)
print(likelihood)

ordering_likelihood = log_likelihood_ordering(df_experimental_data, df_prompts, 1000)
print(ordering_likelihood)



# row_num = 300
# context = df_experimental_data.loc[row_num, 'story']
# query = df_experimental_data.loc[row_num, 'query']
# prompt = get_prompt_for_context(df_prompts, context)
# print("Prompt: " + str(prompt))
# probability_distribution = get_next_word_probability_distribution(prompt)

# words = df_experimental_data['cleaned_query'].tolist()
# tokens = {token for token, _ in probability_distribution}

# for word in words:
#     if word not in tokens:
#         print(f"Word '{word}' is not in the BERT vocabulary")

# sorted_distribution = sorted(probability_distribution, key=lambda x: x[1], reverse=True)

# # Get the top 100 elements
# top_100_elements = sorted_distribution[:500]

# # Print the top 100 elements
# for token, probability in top_100_elements:
#     print(f'Token: {token}, Probability: {probability}')

# prob_set = prob_query_in_set(probability_distribution, query)
# print("Prob Query - " + str(query) + ": " + str(prob_set))

# runs = 10000
# df_empirical_exp = sampling_sets_empirical_exp(df_experimental_data, df_prompts, runs)
# df_empirical_exp = df_empirical_exp.drop_duplicates()
# df_empirical_exp.to_csv('../data/empirical_exp_results_set_size=10_num_sampled_sets=' + str(runs) + '.csv', index=False)


# total_set_log_likelihood = log_likelihood_set(df_experimental_data, df_prompts)
# print("Total Log likelihood: " + str(total_set_log_likelihood))

#  pdb.set_trace()
