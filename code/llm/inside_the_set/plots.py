import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# assume df is your DataFrame with columns: llm_name, context, spearman_rho

df = pd.read_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/split_half_spearman_summary.csv")

# Sort by mean_rho
df_sorted = df.sort_values("mean_rho", ascending=False)

# Plot
plt.figure(figsize=(10,6))
plt.bar(df_sorted["context"], df_sorted["mean_rho"], color="skyblue", edgecolor="black")

plt.xticks(rotation=45, ha="right")
plt.xlabel("Context")
plt.ylabel("Mean Spearman Rho")
plt.title("Mean Spearman Correlation by Context (sorted)")
plt.tight_layout()

# --- 3. Save and show ---
plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/participant_split_correlation_2.png", dpi=300)
plt.show()
