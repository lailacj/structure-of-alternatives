import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import pandas as pd

nltk.download('wordnet')
nltk.download('wordnet_ic')

# Load Information Content (IC) file (from Brown corpus)
# Information Content is based on the corpus frequency of words, so we may want to use a different corpus
ic = wordnet_ic.ic('ic-brown.dat')

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

# Only include the rows with a unique trigger 
df_experimental_data = df_experimental_data.drop_duplicates(subset='cleaned_trigger')

def get_most_similar_words(word, top_n, pos=wn.NOUN):
    """
    Given a word, find and rank the most similar words using Resnik Similarity.
    - word: input word (e.g., 'juice')
    - top_n: number of top similar words to return
    - pos: part of speech (default: NOUN)
    """
    # Get all synsets of the input word
    synsets1 = wn.synsets(word, pos=pos)
    
    if not synsets1:
        return []

    # Use the first synset (most common meaning)
    synset1 = synsets1[0]

    # Get all noun synsets to compare
    all_noun_synsets = list(wn.all_synsets(pos))
    similarities = []

    for synset2 in all_noun_synsets:
        # Compute Resnik similarity
        try:
            score = synset1.res_similarity(synset2, ic)
            word_only = synset2.name().split('.')[0]
            similarities.append((word_only, score))
        except:
            continue  # Ignore errors where similarity cannot be computed

    # Sort by highest similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top N results
    return similarities[:top_n]


results_data = []

for _, row in df_experimental_data.iterrows():
    trigger = row['cleaned_trigger']
    similar_words = get_most_similar_words(trigger, top_n=100)

    for word, resnik_score in similar_words:
        results_data.append([row['story'], trigger, word, resnik_score])

df_results = pd.DataFrame(results_data, columns=['context', 'trigger', 'word', 'resnik_score'])
df_results.to_csv('../data/wordNet_top_words.csv', index=False)





