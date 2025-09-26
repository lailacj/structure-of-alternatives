# Get log likelihood for each alternative model structure 
# using the cloze probabilities from the generation task.

import pandas as pd
import numpy as np

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

demo = sample_ordering("fridge", random_state=123)
print(demo)



# def results():
#     ordering_data = []
#     set_data = []
#     conjunction_data = []
#     disjunction_data = []
#     always_negate_data = []
#     never_negate_data = []

