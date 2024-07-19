from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pdb
import random
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

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

    top_k_indices = torch.topk(mask_token_logits, top_k).indices

    # Filter logits to only the top_k
    top_k_logits = mask_token_logits[top_k_indices]

    # Apply softmax to the top_k logits to get normalized probabilities
    top_k_probs = torch.softmax(top_k_logits, dim=-1)

    # Decode the top 10 predictions and their normalized probabilities
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

    # Return the results as a list of (token, probability) tuples
    return list(zip(predicted_tokens, top_k_probs.tolist()))


# ------ Set Model with BERT ------

# The probability a query is in a set is equal to the probability of query being the next word. 
def prob_query_in_set(top_k_predictions, query):
    for token, prob in top_k_predictions:
        if token == query:
            return prob
    
    # Return None if the token is not in top_k_prediction
    return None  

def get_set(top_k_predictions, set_size=3):
    words, probabilities = zip(*top_k_predictions)
    
    # Normalize probabilities to sum to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    sampled_indices = np.random.choice(len(words), size=set_size, replace=False, p=probabilities)
    sampled_words = [words[i] for i in sampled_indices]
    
    return sampled_words

# Input text with [MASK] token
text = "Sam opens the fridge and responds, 'I only have [MASK].'"
top_k_size = 10
top_k_predictions = get_top_k_predictions(text, top_k_size)
for token, prob in top_k_predictions:
    print(f'{token}: {prob:.4f}')

query = "water"
prob_query_set = prob_query_in_set(top_k_predictions, query)
print(f"The probability of the token '{query}' is: {prob_query_set:.4f}" if prob_query_set else f"The token '{query}' is not in the top_k predictions.")

set = get_set(top_k_predictions)
print("Sampled set of words: ", set)

