from transformers import BertTokenizer, BertForMaskedLM
import pdb
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# ------ Getting next work predictions from BERT ------

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

# Input text with [MASK] token
text = "Sam opens the fridge and responds, 'I only have [MASK].'"
top_k_size = 10
top_k_predictions = get_top_k_predictions(text, top_k_size)
for token, prob in top_k_predictions:
    print(f'{token}: {prob:.4f}')

# ------ Set Model with BERT ------



# probabilities = torch.softmax(mask_token_logits, dim=-1)

# # Get the top next word predicitions
# top_k = 50
# top_k_indices = torch.topk(probabilities, top_k).indices

# # Decode the top 10 predictions and their probabilities
# predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
# predicted_probs = probabilities[top_k_indices]

# # Print the results
# for token, prob in zip(predicted_tokens, predicted_probs):
#     print(f'{token}: {prob.item():.4f}')
