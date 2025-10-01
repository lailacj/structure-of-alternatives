# Get log likelihood for each alternative model structure 
# using the cloze probabilities from the generation task.

import pandas as pd
import numpy as np
import pdb 

SET_BOUNDARY_RANGE = range(2, 25, 1)
NUM_SAMPLES = 100

# ------- Cleaning and Sorting the Experimental Data and Cloze Probability Data ------

# Function that removes 'a' or 'an' from the front of the items
def clean_word(word):
    word = word.lower()
    if word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word

# Load the experimental data
df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')
df_experimental_data['cleaned_trigger'] = df_experimental_data['trigger'].apply(clean_word)
df_experimental_data['cleaned_query'] = df_experimental_data['query'].apply(clean_word)
df_experimental_data = df_experimental_data.sort_values(by='story')

# Load the cloze probabilities data
df_cloze = pd.read_csv('../data/word_freq_and_cloze_prob.csv')

# Only use the contexts that are in both the experimental data and cloze data
common_contexts = set(df_experimental_data['story']).intersection(set(df_cloze['context']))
df_experimental_data = df_experimental_data[df_experimental_data['story'].isin(common_contexts)]
df_cloze = df_cloze[df_cloze['context'].isin(common_contexts)]

def sample_ordering(context:str, random_state) -> list[str]:
    """Sample an ordering of all words without replacement for a given context,
    weighted by cloze probability."""

    subset = df_cloze[df_cloze['context'] == context].copy()

    # Normalize cloze probabilities to sum to 1
    probs = subset['cloze_probability'].to_numpy(dtype=float)
    probs = probs / probs.sum()

    words = subset["word"].to_numpy()

    rng = np.random.default_rng(seed=random_state)
    idx = rng.choice(len(words), size=len(words), replace=False, p=probs)
    return words[idx].tolist()

def ordering_log_likelihood(samples, query, trigger, query_negated):
    count_query_above_trigger = 0

    for sample in samples:
        sampled_ordering = sample[0]

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
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def conjunction_log_likelihood(samples, query, trigger, query_negated):
    count_conjunction = 0

    for sample in samples:
        sampled_ordering = sample[0]
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
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def disjunction_log_likelihood(samples, query, trigger, query_negated):
    count_disjunction = 0

    for sample in samples:
        sampled_ordering = sample[0]
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
        log_likelihood = np.log(1e-10)
    
    return log_likelihood, empirical_probability, prob_query_obs

def results():
    ordering_data = []
    set_data = []
    conjunction_data = []
    disjunction_data = []
    always_negate_data = []
    never_negate_data = []

    contexts = df_experimental_data['story'].unique()
    context_samples = {}

    for set_boundary in SET_BOUNDARY_RANGE:

        # Get sampled orderings and sets for each context
        for context in contexts:
            context_samples[context] = []

            for i in range(0, NUM_SAMPLES):
                sampled_ordering = sample_ordering(context, random_state=i)
                sampled_set = sampled_ordering[:set_boundary]
                # pdb.set_trace()
                context_samples[context].append((sampled_ordering, sampled_set))

        # Iterate through the experimental data
        for _, row in df_experimental_data.iterrows():
            context = row['story']
            query = row['cleaned_query']
            trigger = row['cleaned_trigger']
            query_negated = row['neg']

            words_in_context = set(df_cloze.loc[df_cloze["context"] == context, "word"])
            
            if query not in words_in_context:
                continue  

            if trigger not in words_in_context:
                continue      

            # For each trial, compute log likelihoods across all samples
            samples = context_samples[context]
            ordering_log_likelihood_single_trial, ordering_empirical_p, ordering_prob_query_obs = ordering_log_likelihood(samples, query, trigger, query_negated)
            set_log_likelihood_single_trial, set_empirical_p, set_prob_query_obs = set_log_likelihood(samples, query, query_negated)
            conjunction_log_likelihood_single_trial, conjunction_empirical_p, conjunction_prob_query_obs = conjunction_log_likelihood(samples, query, trigger, query_negated)
            disjunction_log_likelihood_single_trial, disjunction_empirical_p, disjunction_prob_query_obs = disjunction_log_likelihood(samples, query, trigger, query_negated)
            always_negate_log_likelihood_single_trial, always_negate_empirical_p, always_negate_prob_query_obs = always_negate_log_likelihood(query_negated)
            never_negate_log_likelihood_single_trial, never_negate_empirical_p, never_negate_prob_query_obs = never_negate_log_likelihood(query_negated)

            ordering_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, ordering_log_likelihood_single_trial, ordering_empirical_p, ordering_prob_query_obs])
            set_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, set_log_likelihood_single_trial, set_empirical_p, set_prob_query_obs])
            conjunction_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, conjunction_log_likelihood_single_trial, conjunction_empirical_p, conjunction_prob_query_obs])
            disjunction_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, disjunction_log_likelihood_single_trial, disjunction_empirical_p, disjunction_prob_query_obs])
            always_negate_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, always_negate_log_likelihood_single_trial, always_negate_empirical_p, always_negate_prob_query_obs])
            never_negate_data.append([set_boundary, NUM_SAMPLES, context, trigger, query, query_negated, never_negate_log_likelihood_single_trial, never_negate_empirical_p, never_negate_prob_query_obs])

        # Convert lists to DataFrames
        set_df = pd.DataFrame(set_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])
        ordering_df = pd.DataFrame(ordering_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])
        conjunction_df = pd.DataFrame(conjunction_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])
        disjunction_df = pd.DataFrame(disjunction_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])
        always_negate_df = pd.DataFrame(always_negate_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])
        never_negate_df = pd.DataFrame(never_negate_data, columns=['set_boundary', 'num_samples', 'context', 'trigger', 'query', 'neg', 'log_likelihood', 'empirical_probability', 'probability_query_observed'])

        # Save results to CSV incrementally
        set_df.to_csv('../results/set_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/set_results_cloze_prob_sep2025.csv'), index=False)
        ordering_df.to_csv('../results/ordering_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/ordering_results_cloze_prob_sep2025.csv'), index=False)
        conjunction_df.to_csv('../results/conjunction_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/conjunction_results_cloze_prob_sep2025.csv'), index=False)
        disjunction_df.to_csv('../results/disjunction_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/disjunction_results_cloze_prob_sep2025.csv'), index=False)
        always_negate_df.to_csv('../results/always_negate_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/always_negate_results_cloze_prob_sep2025.csv'), index=False)
        never_negate_df.to_csv('../results/never_negate_results_cloze_prob_sep2025.csv', mode='a', header=not pd.io.common.file_exists('../results/never_negate_results_cloze_prob_sep2025.csv'), index=False)

        print(set_df.head())
        print(ordering_df.head())
        print(conjunction_df.head())
        print(disjunction_df.head())

        # Clear lists to free memory
        set_data.clear()
        ordering_data.clear()
        conjunction_data.clear()
        disjunction_data.clear()
        always_negate_data.clear()
        never_negate_data.clear()

    return None

print("Starting log likelihood calculations...")
results()