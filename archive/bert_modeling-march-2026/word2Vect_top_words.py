import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pdb

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

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

# Get the input embedding layer (static word embeddings)
input_embeddings = model.embeddings.word_embeddings
vocab = tokenizer.get_vocab()

# Convert vocab ID to word
id_to_word = {v: k for k, v in vocab.items()}

def is_valid_word(word):
    """Check if a token is a valid word."""
    return (
        word.isalpha() and          # Only alphabetic characters
        word.isascii() and          # Exclude non-ASCII characters (like π, γ)
        not word.startswith("##") and # Exclude subwords
        len(word) > 1               # Exclude single characters
    )

# Filter out non-word tokens (special tokens + subwords)
valid_word_ids = [word_id for word_id, word in id_to_word.items() if is_valid_word(word)]

# pdb.set_trace()

def get_bert_embedding(word):
    """Returns the static BERT input embedding for a given word."""
    token_id = tokenizer.convert_tokens_to_ids(word)
    if token_id == tokenizer.unk_token_id:
        print(f"'{word}' is not in BERT's vocabulary.")
        return None
    return input_embeddings.weight[token_id].detach().numpy()

def find_top_similar_words(target_word, top_n):
    """Finds the top N most similar words to a given word based on cosine similarity."""
    target_embedding = get_bert_embedding(target_word)
    if target_embedding is None:
        return None

    similarities = []
    target_embedding = target_embedding.reshape(1, -1)

    # Compute cosine similarity for each word in the vocabulary
    for word_id in valid_word_ids:
        word_embedding = input_embeddings.weight[word_id].detach().numpy().reshape(1, -1)
        sim = cosine_similarity(target_embedding, word_embedding)[0][0]
        similarities.append((id_to_word[word_id], sim))

    # Sort by similarity and get top N words
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

results_data = []

for _, row in tqdm(df_experimental_data.iterrows(), total=len(df_experimental_data), desc="Processing Words"):
    trigger = row['cleaned_trigger']
    top_words = find_top_similar_words(trigger, top_n=100)

    if top_words == None:
        continue

    for word, similarity in top_words:
        results_data.append([row['story'], trigger, word, similarity])

df_results = pd.DataFrame(results_data, columns=['context', 'trigger', 'word', 'cosine_similarity'])
df_results.to_csv('../data/word2Vec_top_words.csv', index=False)

