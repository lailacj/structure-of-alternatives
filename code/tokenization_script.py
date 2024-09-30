# A script that check the BERT tokenization of words. 
from transformers import BertTokenizer
from transformers import AutoTokenizer
import pandas as pd

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")

df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')

def tokenize_words(words):
    tokenized_words = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenized_words.append((word, tokens, token_ids))
    return tokenized_words

words = df_experimental_data['query'].unique()
tokenized_output = tokenize_words(words)

for word, tokens, token_ids in tokenized_output:
    print(f'Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}')
