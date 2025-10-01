# Get the cloze probability for the words in the generation task. 

import pandas as pd
import ast

# ---- Config ----
N_PARTICIPANTS = 52
NEG_ALPHA = 1-(1/N_PARTICIPANTS)  # scale factor to keep neg probs strictly below the min positive prob
INPUT_PATH = "../data/generation_task_word_frequencies.csv"
OUTPUT_PATH = "../data/word_freq_and_cloze_prob.csv"

# ---- Load ----
word_freq_df = pd.read_csv(INPUT_PATH)

rows = []
context_min_pos_prob = {}  # context -> lowest positive cloze prob

# ---------- PASS 1: POS columns ----------
for col in word_freq_df.columns:
    if col.endswith("_pos"):
        context = col.replace("_pos", "")

        for cell in word_freq_df[col].dropna():
            # Parse string "(word, freq)" safely
            word, freq = ast.literal_eval(cell)
            cloze_prob = freq / N_PARTICIPANTS
            rows.append([context, word, freq, cloze_prob, "pos"])

            # track min positive cloze prob for this context
            if context not in context_min_pos_prob:
                context_min_pos_prob[context] = cloze_prob
            else:
                context_min_pos_prob[context] = min(context_min_pos_prob[context], cloze_prob)

# ---------- PASS 2: NEG columns ----------
for col in word_freq_df.columns:
    if col.endswith("_neg"):
        context = col.replace("_neg", "")

        # If no positive column was found for this context, skip safely
        # (or set a conservative default)
        if context not in context_min_pos_prob:
            # You could set a tiny cap if you want to include these:
            # context_min_pos_prob[context] = 1.0 / N_PARTICIPANTS
            # For now, skip to be explicit:
            continue

        # Collect all (word, freq) for this neg column
        neg_items = []
        for cell in word_freq_df[col].dropna():
            word, freq = ast.literal_eval(cell)
            neg_items.append((word, freq))

        if not neg_items:
            continue

        total_neg_freq = sum(f for _, f in neg_items)

        # If there’s no mass, assign zero
        if total_neg_freq == 0:
            for word, freq in neg_items:
                rows.append([context, word, freq, 0.0, "neg"])
            continue

        # cap under the min positive cloze prob
        cap = context_min_pos_prob[context] * NEG_ALPHA

        # Distribute proportionally so ordering is preserved
        for word, freq in neg_items:
            score = freq / total_neg_freq
            cloze_prob = score * cap
            # Make sure it's strictly below the min positive (guard floating point)
            cloze_prob = min(cloze_prob, context_min_pos_prob[context] - 1e-12)
            rows.append([context, word, freq, cloze_prob, "neg"])

# ---------- Build output ----------
output_df = pd.DataFrame(
    rows, columns=["context", "word", "frequency", "cloze_probability", "type"]
)

# Remove NEG duplicates if the same word exists in POS for the same context
output_df = output_df[
    ~(
        (output_df["type"] == "neg") &
        (output_df.set_index(["context", "word"]).index.isin(
            output_df[output_df["type"] == "pos"].set_index(["context", "word"]).index
        ))
    )
]

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

# Save
output_df.to_csv(OUTPUT_PATH, index=False)
