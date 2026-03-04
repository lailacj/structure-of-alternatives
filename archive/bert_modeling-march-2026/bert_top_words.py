import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
df_prompts = pd.read_csv("../data/prompts_BERT.csv")

# Get the input prompt for BERT based on the context.
def get_prompt_for_context(df_prompts, context):
    matching_row = df_prompts[df_prompts['story'] == context]
    if not matching_row.empty:
        return matching_row['prompt'].values[0]
    else:
        return None

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

top_words = []

for context in df_prompts['story']:
    bert_prompt = get_prompt_for_context(df_prompts, context)
    bert_next_word_probabilities = get_next_word_probability_distribution(bert_prompt)
    bert_next_word_probabilities = sorted(bert_next_word_probabilities, key=lambda x: x[1], reverse=True)

    for token, prob in bert_next_word_probabilities[:100]:
        top_words.append([context, token, prob])

df_top_words = pd.DataFrame(top_words, columns=['context', 'token', 'probability'])
df_top_words.to_csv("../data/BERT_top_words.csv", index=False)