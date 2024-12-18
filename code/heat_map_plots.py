import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------- Load and Reorganize Data ------

alternative_strucutre = "disjunction"

df_experimental_data = pd.read_csv('../data/sca_dataframe_filtered.csv')
df_experimental_data = df_experimental_data.rename(columns={'story': 'context'})

columns_to_keep = ['context', 'query_relevance', 'trigger_relevance', 'cleaned_query', 'cleaned_trigger']
df_experimental_data = df_experimental_data[columns_to_keep]
df_experimental_data = df_experimental_data.rename(columns={'cleaned_query': 'query', 'cleaned_trigger': 'trigger'})

df_results = pd.read_csv('../data/disaggregated_results_' + alternative_strucutre + '.csv')

# Merge the two dataframes on context, query, and trigger
merged_data = pd.merge(
    df_experimental_data,
    df_results,
    on=["context", "query", "trigger"],
    how="inner"
)

# Sort by query_relevance and trigger_relevance
merged_data = merged_data.sort_values(by=["query_relevance", "trigger_relevance"])

# Ensure queries are ordered by query_relevance
merged_data["query"] = pd.Categorical(
    merged_data["query"],
    categories=merged_data.sort_values("query_relevance")["query"].unique(),
    ordered=True
)

# Ensure triggers are ordered by trigger_relevance
merged_data["trigger"] = pd.Categorical(
    merged_data["trigger"],
    categories=merged_data.sort_values("trigger_relevance")["trigger"].unique(),
    ordered=True
)

# ------- Heat Map Plots ------

# Get unique contexts
contexts = merged_data["context"].unique()

# Set up the subplot grid
n_contexts = len(contexts)
fig, axes = plt.subplots(
    nrows=(n_contexts + 3) // 4,  # Calculate rows to fit all plots
    ncols=4,                      # Maximum 3 columns
    figsize=(20, 5 * ((n_contexts + 3) // 4)),  # Adjust height dynamically
    constrained_layout=True
)

# Flatten axes array for easier indexing
axes = axes.flatten()

for i, context in enumerate(contexts):
    context_data = merged_data[merged_data["context"] == context]

    # Create pivot table
    heatmap_data = context_data.pivot_table(
        index="trigger",
        columns="query",
        values="empirical_probability",
        aggfunc="mean"
    )

    heatmap_data = heatmap_data.fillna("n/a")

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt="", cmap="coolwarm", cbar=True, vmin=0, vmax=1, linewidths=0.5, linecolor="white", mask=heatmap_data == "n/a", ax=axes[i])
    axes[i].set_facecolor('white')
    axes[i].set_title(context)
    axes[i].set_xlabel("Query (ordered by query relevance)")
    axes[i].set_ylabel("Trigger (ordered by trigger relevance)")

    # Hide any unused axes
for j in range(len(contexts), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Heatmaps for Empirical Probability for " + alternative_strucutre, fontsize=16, y=1.02)  

    # plt.title(context)
    # plt.xlabel("Query (ordered by query relevance)")
    # plt.ylabel("Trigger (ordered by trigger relevance)")
    # plt.tight_layout()

output_file = "../figures/heatmap_plot_" + alternative_strucutre + ".png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")  # High-quality output
plt.show()