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

def log_likelihood(experimental_data, num_runs):
    current_context = None
    disjunction_log_likelihood = 0
    current_likelihood = 0
    disaggregated_likelihoods = []
    queries_not_in_distribution = []

    for _, row in experimental_data.iterrows():
        context = row['story']
        query = row['cleaned_query']
        query_negated = row['neg'] 
        trigger = row['cleaned_trigger']

        if context != current_context:
            # Get the BERT input prompt and BERT output distribution for each context
            prompt = get_prompt_for_context(df_prompts, context)
            distribution = get_next_word_probability_distribution(prompt)
            current_context = context

        # sample 