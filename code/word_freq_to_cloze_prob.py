# Get the cloze probability for the words in the generation task. 

import pandas as pd
import ast

# Load word frequencies file
word_freq_df = pd.read_csv("../data/generation_task_word_frequencies.csv")

#number of participants in generation task
N_PARTICIPANTS = 52

rows = []

for col in word_freq_df.columns:
    if col.endswith("_pos"):
        context = col.replace("_pos", "")

        for cell in word_freq_df[col].dropna():
            # Parse string "(word, freq)" safely
            word, freq = ast.literal_eval(cell)
            cloze_prob = freq / N_PARTICIPANTS
            rows.append([context, word, freq, cloze_prob])

output_df = pd.DataFrame(rows, columns=["context", "word", "frequency", "cloze_probability"])

# Rename contexts 
rename_map = {
    "handbag": "bag",
    "corner": "salad",
    "fitness": "gym",
    "library": "science",
    "garage": "throw",
    "closet": "beach"
}
output_df["context"] = output_df["context"].replace(rename_map)

# Remove all rows with context = "play"
output_df = output_df[output_df["context"] != "play"]

output_path = "../data/word_freq_and_cloze_prob.csv"
output_df.to_csv(output_path, index=False)
