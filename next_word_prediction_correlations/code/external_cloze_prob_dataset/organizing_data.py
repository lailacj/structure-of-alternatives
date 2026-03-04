# Cleans and organizes the cloze probability dataset into a CSV file.

# Removed quotes around entire sentences (but kept quotes that are part of the sentence).
# Replaced dashes in words with spaces (e.g., girly-girl → girly girl).
# Filtering out “No Response” and any words containing **.

import re
import pandas as pd

# Paths
input_file = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/external_cloze_prob_dataset/output.md"
output_file = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/external_cloze_prob_dataset/cloze_data.csv"

sent_numbers = []
sentences = []
words = []
probs = []

with open(input_file, "r") as f:
    lines = f.readlines()

current_sentence = None
current_number = None

for line in lines:
    line = line.strip()
    
    # Match numbered sentences
    match_sentence = re.match(r"^(\d+)\.\s+(.*)", line)
    if match_sentence:
        current_number = int(match_sentence.group(1))
        current_sentence = match_sentence.group(2)
        # Remove the blank placeholder like "__________."
        current_sentence = current_sentence.replace("__________.", "").strip()
        # Remove surrounding quotes if present
        if current_sentence.startswith('"') and current_sentence.endswith('"'):
            current_sentence = current_sentence[1:-1].strip()
        continue
    
    # Match word-prob lines
    match_word = re.match(r"^\*\s*(.+)\s*\(([\d.]+)\)", line)
    if match_word and current_sentence:
        word = match_word.group(1).strip()
        prob = float(match_word.group(2))
        
        # Filter: skip "No Response" and words with ** in them
        if word.lower() == "no response":
            continue
        if "**" in word:
            continue
        # Replace dash with space in words like girly-girl
        word = word.replace("-", " ")
        
        sent_numbers.append(current_number)
        sentences.append(current_sentence)
        words.append(word)
        probs.append(prob)

# Create DataFrame
df = pd.DataFrame({
    "sentence_number": sent_numbers,
    "sentence": sentences,
    "word": words,
    "cloze_prob": probs
})

# Strip leading/trailing quotes (double or single) if they wrap the whole sentence
def strip_outer_quotes(s):
    if not isinstance(s, str) or len(s) < 2:
        return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s

df["sentence"] = df["sentence"].apply(strip_outer_quotes)

# Save to CSV
df.to_csv(output_file, index=False)
output_file