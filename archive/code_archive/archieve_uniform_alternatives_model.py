# ---- Imports and Set Up ----

import nltk # Natural Language Toolkit 
import numpy as np
import pandas as pd
import pdb
import random
from collections import Counter
from nltk.corpus import brown # Brown Corpus
from nltk.probability import FreqDist

nltk.download('brown') 
nltk.download('universal_tagset') # Part of speech tagging that NLTK uses

df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')

# ---- Corpus Data and Experimental Data Preparation ----

# Getting most frequent nouns
vocab_size = 100000
words = brown.tagged_words(tagset='universal')
nouns = [word for word, pos in words if pos == 'NOUN']
noun_freq = Counter(nouns).most_common(vocab_size)
most_common_nouns = [word for word, freq in noun_freq]

# Getting most frequent words
# brown_words = brown.words()
# freq_dist = FreqDist(brown_words)
# top_words = freq_dist.most_common(100000)

# Function that removes 'a' or 'an' from the front of the items
# Not being used in the rest of the code. I wrote it to see how many query words are in the corpus dataset.
def clean_word(word):
    word = word.lower()
    if word.startswith('a '):
        return word[2:]
    elif word.startswith('an '):
        return word[3:]
    else:
        return word

df_experimental_data['cleaned_trigger'] = df_experimental_data['trigger'].apply(clean_word)
df_experimental_data['cleaned_query'] = df_experimental_data['query'].apply(clean_word)
df_experimental_data = df_experimental_data.sort_values(by='story')

not_in_corpus = []
for index, row in df_experimental_data.iterrows():
    word = row['cleaned_query']
    context = row['story']
    if word not in most_common_nouns:
        not_in_corpus.append(word)

unique_words_not_in_top_words = set(not_in_corpus)

print("The following words are NOT in the list of most common nouns:")
for word in unique_words_not_in_top_words:
    print(word)
print(len(unique_words_not_in_top_words))

# Remove the words that are not in the corpus from the experimental data
print(df_experimental_data.shape)
df_experimental_data = df_experimental_data[~df_experimental_data['cleaned_query'].isin(unique_words_not_in_top_words)]
print(df_experimental_data.shape)

# Save the filtered data
# df_experimental_data_filtered.to_csv('../data/sca_dataframe_filtered.csv', index=False)

# ---- Uniform Set Model ----

def prob_query_in_set(set_size):
    """
    Returns the probability that a query is inside the set. Sets are uniformly 
    sampled (without replacement) from a vocabulary of size, vocab_size.
    """
    # Query being in the set is INDEPENDENT of set size
    if set_size == None:
        return (1 / vocab_size)
    # Query being in the set is DEPENDENT of set size
    else:
        return (set_size / vocab_size) 

def get_set(set_size):
    """
    Returns a uniformly sampled set of nouns.
    """
    return random.sample(most_common_nouns, set_size)

# def prob_query_negated_set(trigger, query, inside_set):
#     # Noise Model: Probability that a query gets negated equal to the probability that it is inside the set
#     # return prob_query_in_set()

#     # Deterministic Model: When the query is inside the same set as the trigger then the query will be 
#     # negated with a probability of 1; and 0 otherwise 

#     if query in inside_set and trigger in inside_set:
#         return 1 # Negate with probability 1
#     else:
#         return 0 # Negate with probability 0

def set_uniform_log_likelihood():
    set_log_likelihood = 0

    for index, row in df_experimental_data.iterrows():
        query_negated = row['neg'] 
        prob_set = prob_query_in_set(set_size=None)

        if query_negated == 1:
            # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
            prob_query_obs = prob_set * 1 + (1-prob_set) * 0
        else:
            # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
            prob_query_obs = prob_set * 0 + (1-prob_set) * 1
        
        # product for probabilities; sum for logs
        set_log_likelihood += np.log(prob_query_obs)

    return set_log_likelihood

# ---- Uniform Ordering Model ----

def prob_query_above_trigger():
    # Since all possible orderings are equally likely, and since there are only two possible positions
    # that a query and trigger can take in relation to another (query above trigger or query below trigger), 
    # and since the probability of these possibilities needs to add to 1; the probability of each is 1/2. 
    return (1/2)

def get_ordering():
    return random.sample(most_common_nouns, vocab_size)

# def prob_query_negated_order(trigger, query, ordering):
#     # Noise Model: Probability that a query gets negated equal to the probability that it is above the trigger
#     # return prob_query_above_trigger()

#     # Deterministic Model: When the query is above the trigger then the query will be negated with a probability 
#     # of 1; and 0 otherwise 

#     #  Check if both query and trigger are in the ordering
#     if query not in ordering or trigger not in ordering:
#         raise ValueError("Both query and trigger must be in the ordering")
    
#     # Get the index positions
#     query_index = ordering.index(query)
#     trigger_index = ordering.index(trigger)

#     if query_index < trigger_index:
#         return 1 # Negate with probability 1
#     else:
#         return 0 # Negate with probability 0

def ordering_uniform_likelihood():
    ordering_log_likelihood = 0

    for index, row in df_experimental_data.iterrows():
        query_negated = row['neg'] 
        prob_above = prob_query_above_trigger()

        if query_negated == 1:
            prob_query_obs = prob_above * 1 + (1-prob_above) * 0
        else:
            prob_query_obs = prob_above * 0 + (1-prob_above) * 1
        
        ordering_log_likelihood += np.log(prob_query_obs)

    return ordering_log_likelihood


print("Set Uniform Likelihood: " + str(set_uniform_log_likelihood()))
print("Ordering Uniform Likelihood: " + str(ordering_uniform_likelihood()))

# pdb.set_trace()
