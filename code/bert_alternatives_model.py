from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pandas as pd
import pdb
import random
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')
df_experimental_data = df_experimental_data.sort_values(by='story')

df_prompts = pd.read_csv("../data/prompts_BERT.csv")

# ------ Getting next word predictions from BERT ------

def get_top_k_predictions(text, top_k):
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get logits 
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get index of [MASK] token
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Get logits of masked token
    mask_token_logits = logits[0, mask_token_index, :].squeeze()

    # Get the top k indices
    top_k_indices = torch.topk(mask_token_logits, top_k).indices

    # Filter logits to only the top k
    top_k_logits = mask_token_logits[top_k_indices]

    # Apply softmax to the top_k logits 
    top_k_probs = torch.softmax(top_k_logits, dim=-1)

    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, top_k_probs.tolist()))

def get_all_logits(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    mask_token_logits = logits[0, mask_token_index, :].squeeze()
    probs = torch.softmax(mask_token_logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(range(len(mask_token_logits)))

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, probs.tolist()))

# ------ Set Model with BERT ------

# The probability a query is in a set is equal to the probability of query being the next word. 
def prob_query_in_set(logits, query):
    for token, prob in logits:
        if token == query:
            return prob
    
    # Return zero if the token is not in logits
    return 0 

def get_set(logits, set_size=3):
    words, probabilities = zip(*logits)
    
    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    sampled_indices = np.random.choice(len(words), size=set_size, replace=False, p=probabilities)
    sampled_words = [words[i] for i in sampled_indices]
    
    return sampled_words

def get_prompt_for_context(df_prompts, context):
    matching_row = df_prompts[df_prompts['story'] == context]
    if not matching_row.empty:
        return matching_row['prompt'].values[0]
    else:
        return None

current_context = None
set_log_liklihood = 0
for index, row in df_experimental_data.iterrows():
    context = row['story']
    query = row['query']
    trigger = row['trigger']
    query_negated = row['neg'] 

    print("Context: " + str(context))

    # Sample inside_set for a given story. Resample for every new story. 
    if context != current_context:
        prompt = get_prompt_for_context(df_prompts, context)
        print("Promt: " + str(prompt))
        logits = get_top_k_predictions(prompt, top_k=100)
        inside_set = get_set(logits)
        print("Sampled Set: " + str(inside_set))
        current_context = context

    prob_set = prob_query_in_set(logits, query)
    print("Prob in set: " + str(prob_set))
    # prob_neg_given_set = prob_query_negated_set(trigger, query, inside_set)

    if query_negated == 1:
        # p(q neg) = p(q in set) * p(q neg | in set) + p(q not in set) * p(q neg | not set)
        prob_query_obs = prob_set * 1 + (1-prob_set) * 0
    else:
        # p(q not neg) = p(q in set) * p(q not neg | in set) + p(q not in set) * p(q not neg | not set)
        prob_query_obs = prob_set * 0 + (1-prob_set) * 1
    
    print("Prob Obsevered: " + str(prob_query_obs))

    if prob_query_obs == 0:
        set_log_liklihood = 0 
    else:
        set_log_liklihood = np.log(prob_query_obs)

    set_log_liklihood += set_log_liklihood

    print("Log Liklihood: " + str(set_log_liklihood) + "\n\n")

print("Set Log Liklihood: " + str(set_log_liklihood))


# Input text with [MASK] token
# text = "Sam opens the fridge and responds, 'I only have [MASK].'"
# top_k_size = 50
# top_k_predictions = get_top_k_predictions(text, top_k_size)
# # all_logits = get_all_logits(text)
# for token, prob in top_k_predictions:
#     print(f'{token}: {prob:.4f}')

# query = "water"
# prob_query_set = prob_query_in_set(top_k_predictions, query)
# print(f"The probability that the token '{query}' is in the sampled set is: {prob_query_set:.4f}" if prob_query_set else f"The token '{query}' is not in the top_k predictions.")

# set = get_set(top_k_predictions)
# print("Sampled set of words: ", set)

