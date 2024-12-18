from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import random
import torch
import seaborn as sns

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

# ------ Set Model with BERT (Version 1) ------
# The probability a query is in a set is equal to the probability of query being the next word. 

def prob_query_in_set(probability_distribution, query):
    # try:
        # for token, probability in probability_distribution:
        #     if token == query:
        #         return probability 
        
        # If the token is not found, raise an error
        # raise ValueError(f"Query '{query}' is not in BERT probability distribution")
    
    # except ValueError as e:
        # print(e)
        # raise 

    for token, probability in probability_distribution:
        if token == query:
            return probability
    
    return 0

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

# ------ Set Model with BERT (Version 2 - Sampling) ------
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
    current_likelihood = 0
    disaggregated_likelihoods = []
    queries_not_in_distribution = []

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
                sampled_set = get_set(distribution, set_size=random.randint(2, len(distribution)))
                all_sampled_sets.append(sampled_set)

        # Skip to the next row if query is not in the predicted tokens of distribution
        if not any(predicted_token == query for predicted_token, _ in distribution):
            queries_not_in_distribution.append(query)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue

        # Count how many of the sampled sets contain the query 
        count = 0
        count = sum(1 for sampled_set in all_sampled_sets if query in sampled_set)

        # print(f"Total sampled sets: {len(all_sampled_sets)}") 
        # print(f"The query word '{query}' appears in {count} sampled sets.")

        # The the proporation of smapled sets that contain the query 
        # print(f"Count: {count}, Num Sets: {num_sets}")
        empirical_probability = float(count) / float(num_sets)

        # Compute likelihoods
        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            # prob_query_obs = empirical_probability * 1 + (1-empirical_probability) * 0
            prob_query_obs = empirical_probability

        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            # prob_query_obs = empirical_probability * 0 + (1-empirical_probability) * 1
            prob_query_obs = 1 - empirical_probability

        if prob_query_obs > 0:
            set_log_likelihood += np.log(prob_query_obs)
            current_likelihood = np.log(prob_query_obs)
        else:   
            # Handle the zero or negative case by assigning a small value
            set_log_likelihood += np.log(1e-10)
            current_likelihood = np.log(1e-10)

        disaggregated_likelihoods.append((current_context, query, trigger, empirical_probability, current_likelihood, set(queries_not_in_distribution)))
        print(f"Context: {current_context}, Query: {query}, Trigger: {trigger}, \nEmpirical Probability: {empirical_probability}, \nLog Likelihood: {current_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
        current_likelihood = 0
        queries_not_in_distribution = []

        # if row.equals(df_experimental_data.iloc[-1]):
        #     context_likelihoods.append((current_context, current_context_likelihood, set(queries_not_in_distribution)))
        #     print(f"Context: {current_context}, \nLog Likelihood: {current_context_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n\n")

    # print(set(queries_not_in_distribution))

    return(set_log_likelihood, disaggregated_likelihoods)
    # return(set_log_likelihood)

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
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    ordered_tokens = np.random.choice(tokens, size=len(tokens), replace=False, p=probabilities)
    return ordered_tokens

def log_likelihood_sampling_ordering(df_experimental_data, df_prompts, num_samples):
    current_context = None
    ordering_log_likelihood = 0
    current_likelihood = 0
    disaggregated_likelihoods = []
    queries_not_in_distribution = []

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
        
        # Skip to the next row if query or trigger is not in the predicted tokens of distribution
        if not any(predicted_token == query for predicted_token, _ in distribution):
            queries_not_in_distribution.append(query)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue

        if not any(predicted_token == trigger for predicted_token, _ in distribution):
            queries_not_in_distribution.append(trigger)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue

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

        empirical_probability = float(count_query_above_trigger) / float(num_samples)
        # print("Query: " + str(query) + " - Empirical: " + str(empirical_probability))

        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = empirical_probability
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = 1 - empirical_probability

        if prob_query_obs > 0:
            ordering_log_likelihood += np.log(prob_query_obs)
            current_likelihood = np.log(prob_query_obs)
        else:   
            # Handle the zero or negative case by assigning a small value
            ordering_log_likelihood += np.log(1e-10)
            current_likelihood = np.log(1e-10)

        disaggregated_likelihoods.append((current_context, query, trigger, empirical_probability, current_likelihood, set(queries_not_in_distribution)))
        print(f"Context: {current_context}, Query: {query}, Trigger: {trigger}, \nEmpirical Probability: {empirical_probability}, \nLog Likelihood: {current_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
        current_likelihood = 0
        queries_not_in_distribution = []

    return ordering_log_likelihood, disaggregated_likelihoods

# ------- Conjunction Model with BERT --------

def log_likelihood_sampling_conjunction(df_experimental_data, df_prompts, num_samples):
    current_context = None
    conjunction_log_likelihood = 0
    current_likelihood = 0
    disaggregated_likelihoods = []
    queries_not_in_distribution = []

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

            all_sampled_sets = []
            for _ in range(num_samples):
                sampled_set = get_set(distribution, set_size=random.randint(2, len(distribution)))
                all_sampled_sets.append(sampled_set)
        
        # Skip to the next row if query or trigger is not in the predicted tokens of distribution
        if not any(predicted_token == query for predicted_token, _ in distribution):
            queries_not_in_distribution.append(query)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue

        if not any(predicted_token == trigger for predicted_token, _ in distribution):
            queries_not_in_distribution.append(trigger)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue
        
        # If the probability a query is in a set is independent of the probability that the query is above the trigger in an ordering, 
        # then the probability that a query is in a set and above the trigger in an ordering is the product of the two probabilities.
        # probability of the intersection of in set and above trigger = probability of query in set * probability of query above trigger 

        # If the two events are not independent, then we have to find the conditional probability: 
        # query being above the trigger given that the query is in the set.

        # For now, I am going to assume independence.

        count_set = 0
        count_set = sum(1 for sampled_set in all_sampled_sets if query in sampled_set)
        empirical_probability_set = float(count_set) / float(num_samples)

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

        empirical_probability_ordering = float(count_query_above_trigger) / float(num_samples)

        empirical_probability_conjunction = empirical_probability_set * empirical_probability_ordering

        if query_negated == 1:
            prob_query_obs = empirical_probability_conjunction
        else:
            prob_query_obs = 1 - empirical_probability_conjunction

        if prob_query_obs > 0:
            conjunction_log_likelihood += np.log(prob_query_obs)
            current_likelihood = np.log(prob_query_obs)
        else:
            conjunction_log_likelihood += np.log(1e-10)
            current_likelihood = np.log(1e-10)
        
        disaggregated_likelihoods.append((current_context, query, trigger, empirical_probability_conjunction, current_likelihood, set(queries_not_in_distribution)))
        print(f"Context: {current_context}, Query: {query}, Trigger: {trigger}, \nEmpirical Probability: {empirical_probability_conjunction}, \nLog Likelihood: {current_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
        current_likelihood = 0
        queries_not_in_distribution = []

    return conjunction_log_likelihood, disaggregated_likelihoods

# ------- Disjunction Model with BERT --------

def log_likelihood_sampling_disjunction(df_experimental_data, df_prompts, num_samples):
    current_context = None
    disjunction_log_likelihood = 0
    current_likelihood = 0
    disaggregated_likelihoods = []
    queries_not_in_distribution = []

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

            all_sampled_sets = []
            for _ in range(num_samples):
                sampled_set = get_set(distribution, set_size=random.randint(2, len(distribution)))
                all_sampled_sets.append(sampled_set)
        
        # Skip to the next row if query or trigger is not in the predicted tokens of distribution
        if not any(predicted_token == query for predicted_token, _ in distribution):
            queries_not_in_distribution.append(query)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue

        if not any(predicted_token == trigger for predicted_token, _ in distribution):
            queries_not_in_distribution.append(trigger)
            disaggregated_likelihoods.append((current_context, query, trigger, None, None, set(queries_not_in_distribution)))
            continue 

        count_set = 0
        count_set = sum(1 for sampled_set in all_sampled_sets if query in sampled_set)
        empirical_probability_set = float(count_set) / float(num_samples)

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

        empirical_probability_ordering = float(count_query_above_trigger) / float(num_samples)

        # The probability of the union of in set or above trigger is the probability of in set plus the probability of above trigger minus the probability of the intersection.
        # Assuming independence.
        empirical_probability_disjunction = empirical_probability_set + empirical_probability_ordering - (empirical_probability_set * empirical_probability_ordering)

        if query_negated == 1:
            prob_query_obs = empirical_probability_disjunction
        else:
            prob_query_obs = 1 - empirical_probability_disjunction

        if prob_query_obs > 0:
            disjunction_log_likelihood += np.log(prob_query_obs)
            current_likelihood = np.log(prob_query_obs)
        else:
            disjunction_log_likelihood += np.log(1e-10)
            current_likelihood = np.log(1e-10)
        
        disaggregated_likelihoods.append((current_context, query, trigger, empirical_probability_disjunction, current_likelihood, set(queries_not_in_distribution)))
        print(f"Context: {current_context}, Query: {query}, Trigger: {trigger}, \nEmpirical Probability: {empirical_probability_disjunction}, \nLog Likelihood: {current_likelihood}, \nQueries not in distribution: {set(queries_not_in_distribution)}\n")
        current_likelihood = 0
        queries_not_in_distribution = []

    return disjunction_log_likelihood, disaggregated_likelihoods


# ------- Main -------

# total_set_log_likelihood_sampling, disaggregated_likelihoods = log_likelihood_sampling_sets(df_experimental_data, df_prompts, 1000)
# total_ordering_log_likelihood_sampling, disaggregated_likelihoods = log_likelihood_sampling_ordering(df_experimental_data, df_prompts, 1000)
# total_conjunction_log_likelihood_sampling, disaggregated_likelihoods = log_likelihood_sampling_conjunction(df_experimental_data, df_prompts, 1000)
total_disjunction_log_likelihood_sampling, disaggregated_likelihoods = log_likelihood_sampling_disjunction(df_experimental_data, df_prompts, 1000)

# print("Total Set Log likelihood (Sampling): " + str(total_set_log_likelihood_sampling))
# print("Total Ordering Log likelihood (Sampling): " + str(total_ordering_log_likelihood_sampling))
# print("Total Conjunction Log likelihood (Sampling): " + str(total_conjunction_log_likelihood_sampling))
print("Total Disjunction Log likelihood (Sampling): " + str(total_disjunction_log_likelihood_sampling))

# Create a DataFrame with the results
df_results = pd.DataFrame({
    "context": [item[0] for item in disaggregated_likelihoods],
    "query": [item[1] for item in disaggregated_likelihoods],
    "trigger": [item[2] for item in disaggregated_likelihoods],
    "empirical_probability": [item[3] for item in disaggregated_likelihoods],
    "log_likelihoods": [item[4] for item in disaggregated_likelihoods],
    "excluded_items": [item[5] if item[5] else None for item in disaggregated_likelihoods],
    "alternative_structure": "disjunction"
})

print(f"Num exp data points: {len(df_experimental_data)}")
print(f"Num results: {len(df_results)}")

# Save the DataFrame to a CSV file
df_results.to_csv('../data/disaggregated_results_disjunction.csv', index=False) 

# ------- Random Code -------

# num_sets = 100
# total_set_likelihood, context_set_likelihood = log_likelihood_sampling_sets(df_experimental_data, df_prompts, num_sets)
# print("Sampling Set Log Likelihood" + str(total_set_likelihood))
# context_set_likelihood = context_set_likelihood[1:]

# total_ordering_likelihood, context_ordering_likelihood = log_likelihood_sampling_ordering(df_experimental_data, df_prompts, num_sets)
# print("Sampling Orderings Log Likelihood" + str(total_ordering_likelihood))
# context_ordering_likelihood = context_ordering_likelihood[1:]

# set_context, set_likelihoods = zip(*context_set_likelihood)
# ordering_context, ordering_likelihoods = zip(*context_ordering_likelihood)

# # Remove the "mall" context from the data
# filtered_set_context = [context for context in set_context if context != "mall"]
# filtered_set_likelihoods = [likelihood for context, likelihood in zip(set_context, set_likelihoods) if context != "mall"]
# filtered_ordering_likelihoods = [likelihood for context, likelihood in zip(ordering_context, ordering_likelihoods) if context != "mall"]

# # Calculate the differences and sort by them
# differences = [abs(set_likelihood - ordering_likelihood) for set_likelihood, ordering_likelihood in zip(filtered_set_likelihoods, filtered_ordering_likelihoods)]
# sorted_data = sorted(zip(differences, filtered_set_context, filtered_set_likelihoods, filtered_ordering_likelihoods))
# sorted_differences, sorted_contexts, sorted_set_likelihoods, sorted_ordering_likelihoods = zip(*sorted_data)

# plt.figure(figsize=(10, 8))
# plt.scatter(sorted_contexts, sorted_set_likelihoods, color='blue', label='Set Log Likelihoods', s=100)
# plt.scatter(sorted_contexts, sorted_ordering_likelihoods, color='red', label='Ordering Log Likelihoods', s=100)
# plt.xlabel('Context', fontsize=16)
# plt.ylabel('Log Likelihoods', fontsize=16)
# plt.title('Log Likleihoods by Context for Set and Ordering Models', fontsize=16)
# plt.xticks(rotation=45, ha='right', fontsize=14)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.legend(loc='lower left', fontsize = 14)
# plt.tight_layout()
# plt.savefig('../figures/likelihoods_by_context_.png', dpi=1000)
# plt.show()



# #Plot the likelihoods for each context
# plt.figure(figsize=(13, 6))
# plt.plot(context, likelihoods, marker='o')
# # sns.barplot(x=context[:len(context)], y=likelihoods[:len(context)])
# plt.xlabel('Context')
# plt.ylabel('Log Likelihood')
# plt.title('Log Likelihood for Each Context (Empirical Ordering Model)')
# for i, prob in enumerate(likelihoods[:len(context)]):
#     plt.text(i, prob, f'{prob:.3f}', ha='center', va='bottom')

# plt.savefig('../figures/likelihoods_context_ordering_empirical.png', dpi=500)
# plt.show()







# ordering_likelihood = log_likelihood_sampling_ordering(df_experimental_data, df_prompts, 1000)
# print("Sampling Orderings Log Likelihood" + str(ordering_likelihood))

# set_likelihood = log_likelihood_set(df_experimental_data, df_prompts)
# print("Direct Set Log Likeihood: " + str(set_likelihood))



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



# text = "You and your friend Lee are registering for classes at a fitness center. You say to Lee, 'Let's sign up for some team sports.' Lee looks at the list of classes and responds, 'They only have [MASK].'"
# text = "You and your friend Chris walk by a bakery. You say, 'I want some dessert.' Chris looks inside the bakery and responds, 'They only have [MASK].'"
# # text = "You and your friend Sam go for a long walk together. After the walk, you go back to Sam's house. You say to Sam, 'I'm thirsty.' Sam opens the fridge and responds, 'I only have [MASK].'"

# distribution = get_next_word_probability_distribution(text)
# tokens, probabilities = zip(*distribution)
# tokens = np.array(tokens)
# probabilities = np.array(probabilities)

# # Sort the tokens and probabilities
# sorted_indices = np.argsort(probabilities)[::-1]
# tokens = tokens[sorted_indices]
# probabilities = probabilities[sorted_indices]

# # Plot the distribution (top 100 tokens)
# token_size = 15
# plt.figure(figsize=(30, 8))
# sns.barplot(x=tokens[:token_size], y=probabilities[:token_size])
# plt.ylabel('Probability', fontsize=20)
# plt.xlabel('Token', fontsize=20)
# plt.title("BERT's Next Word Probability Distribution\nGym Context, Top " + str(token_size) + " Tokens", fontsize=18)
# # y-axis goes from 0 to 1
# # plt.ylim(0, 0.2)

# # Put the probability on top of each bar
# # change the font size of the text above the bars

# for i, prob in enumerate(probabilities[:token_size]):
#     plt.text(i, prob, f'{prob:.3f}', ha='center', va='bottom', fontsize=15)

# # Make the font size larger for the entire plot and title
# plt.tick_params(axis='both', which='major', labelsize=18)

# # Save plot with a higher resolution
# plt.savefig('../figures/gym_context_top_15_tokens.png', dpi=1000)

# plt.show()

# query = "swimming"
# trigger = "basketball"

# # Sample a bunch of sets from the BERT distribution and store them in all_sampled_sets
# all_sampled_orderings = []
# for _ in range(10000):
#     all_sampled_orderings = get_ordering(distribution)
#     all_sampled_orderings = all_sampled_orderings.tolist()
#     all_sampled_orderings.append(all_sampled_orderings)

# count_query_above_trigger = 0
# for ordering in all_sampled_orderings:
#     query_index = np.where(ordering == query)[0]
#     if query_index.size > 0:
#         query_index = query_index[0]
#     else:
#         query_index = 0

#     trigger_index = np.where(ordering == trigger)[0]
#     if trigger_index.size > 0:
#         trigger_index = trigger_index[0]
#     else:
#         trigger_index = 0

#     if query_index < trigger_index:
#         count_query_above_trigger += 1

# empirical_probability = count_query_above_trigger / len(all_sampled_orderings)

# print(f"Empirical Probability: {empirical_probability}")

