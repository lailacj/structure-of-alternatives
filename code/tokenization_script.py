# A script that check the BERT tokenization of words. 

from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df_experimental_data = pd.read_csv('../data/sca_dataframe.csv')

def tokenize_words(words):
    """
    Tokenize a list of words using BERT tokenizer.
    
    Parameters:
    words (list): List of words to tokenize.
    
    Returns:
    list of str: List of tokenized words.
    """

    tokenized_words = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        tokenized_words.append((word, tokens))
    return tokenized_words

words = df_experimental_data['query'].unique()
tokenized_output = tokenize_words(words)

# Print the tokenized output
for word, tokens in tokenized_output:
    print(f'Word: {word}, Tokens: {tokens}')
