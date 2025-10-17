# Script to correlate half cloze probabilities with other half. 
# Get 100 draws from one half and 100 draws from the other half. 
# Each draw will give you a correlation. Plot these in a histogram, 
# then put the LLM correlation lines. 

# Re-run with confirmed filename and produce outputs.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/word_freq_and_cloze_prob.csv"
df = pd.read_csv(path)

# Filter to POS
pos_df = df[df["type"].astype(str).str.lower() == "pos"].copy()

# Find the cloze prob column
colname = None
for candidate in ["cloze_probability", "cloze_prob", "cloze", "clozeprob"]:
    if candidate in pos_df.columns:
        colname = candidate
        break
if colname is None:
    raise ValueError(f"None of the expected cloze probability columns found. Available columns:\n{list(pos_df.columns)}")

vals = pd.to_numeric(pos_df[colname], errors="coerce").dropna().to_numpy()

n = len(vals)
if n % 2 == 1:
    vals = vals[:-1]
    n = len(vals)

if n < 4:
    raise ValueError(f"Need at least 4 POS rows; found {n}.")

half = n // 2

rng = np.random.default_rng(12345)
num_splits = 1000
split_rs = []

for _ in range(num_splits):
    perm = rng.permutation(n)
    A = vals[perm[:half]]
    B = vals[perm[half:]]
    r = np.corrcoef(A, B)[0,1]
    split_rs.append(r)

import pandas as pd
split_rs = np.array(split_rs)
summary_df = pd.DataFrame({
    "num_splits":[num_splits],
    "n_pos_rows_used":[n],
    "half_size":[half],
    "mean_r":[split_rs.mean()],
    "std_r":[split_rs.std(ddof=1)],
    "ci_2_5":[np.percentile(split_rs, 2.5)],
    "ci_97_5":[np.percentile(split_rs, 97.5)],
})

print(summary_df)
print(summary_df["mean_r"])

# Save CSV
# corrs_csv_path = "/mnt/data/pos_cloze_split_half_correlations.csv"
# pd.DataFrame({"pearson_r": split_rs}).to_csv(corrs_csv_path, index=False)

# Plot histogram
plt.figure(figsize=(7,5))
plt.hist(split_rs, bins=100)
plt.title("Split-half self-correlation for cloze_probability\n1000 random splits")
plt.xlabel("Pearson r")
plt.ylabel("Count")
hist_path = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/cloze_split_half_hist.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=160)
plt.close()

# (hist_path, corrs_csv_path)

