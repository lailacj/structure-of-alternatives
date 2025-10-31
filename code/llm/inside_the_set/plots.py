import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/split_half_spearman_summary.csv")
df_sorted = df.sort_values("mean_rho", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=df_sorted,
    x="context",
    y="mean_rho",
    color="skyblue",
    edgecolor="black",   
    linewidth=1
)

plt.xticks(rotation=30, ha="right")
plt.xlabel("Context")
plt.ylabel("Mean Spearman ρ")
plt.title("Mean Spearman Correlation per Context")
plt.tight_layout()

plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/participant_split_correlation_2.png", dpi=160)

# df = pd.read_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/split_half_spearman_by_context.csv")

# # Compute mean correlation across contexts for each split
# mean_by_split = (
#     df.groupby("split_id", as_index=False)
#         .agg(mean_rho=("spearman_rho","mean"))
# )

# # Plot histogram
# plt.figure(figsize=(7,5))
# plt.hist(mean_by_split["mean_rho"].dropna(), bins=20, edgecolor="k")
# plt.title("Distribution of mean Spearman correlations\n(across contexts, per split)")
# plt.xlabel("Mean Spearman rho")
# plt.ylabel("Count")
# plt.tight_layout()

# plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/participant_split_correlation_1.png", dpi=160)
# mean_by_split.to_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/mean_spearman_by_split.csv", index=False)


# df = pd.read_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/per_context_rank_agreement.csv")

# means = df.groupby("llm_name", as_index=False)["spearman_rho"].mean().rename(columns={"spearman_rho": "mean_rho"})
# order_llm = means.sort_values("mean_rho", ascending=False)["llm_name"].tolist()

# df["llm_name"] = pd.Categorical(df["llm_name"], categories=order_llm, ordered=True)
# means["llm_name"] = pd.Categorical(means["llm_name"], categories=order_llm, ordered=True)

# plt.figure(figsize=(8, 5))
# sns.stripplot(
#     data=df,
#     x="llm_name",
#     y="spearman_rho",
#     hue="context",
#     jitter=False,
#     dodge=False,
#     alpha=1.0,
# )

# plt.scatter(
#     x=means["llm_name"].cat.codes,
#     y=means["mean_rho"],
#     color="black",
#     s=30,
#     label="Mean",
#     zorder=5
# )

# plt.xticks(range(len(order_llm)), order_llm, rotation=30, ha="right")
# plt.xlabel("LLM name")
# plt.ylabel("Spearman ρ")
# plt.title("Spearman correlation per LLM and context")
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

# plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/llm_inside_set_correlation_1.png", dpi=300)


# means_ctx = (
#     df.groupby("context", as_index=False)["spearman_rho"]
#       .mean()
#       .rename(columns={"spearman_rho": "mean_rho"})
# )
# ctx_order = means_ctx.sort_values("mean_rho", ascending=False)["context"].tolist()

# df["context"] = pd.Categorical(df["context"], categories=ctx_order, ordered=True)
# means_ctx["context"] = pd.Categorical(means_ctx["context"], categories=ctx_order, ordered=True)

# plt.figure(figsize=(10, 5))
# sns.stripplot(
#     data=df,
#     x="context",
#     y="spearman_rho",
#     hue="llm_name",
#     jitter=False,
#     dodge=False,
#     alpha=1.0
# )

# plt.scatter(
#     x=means_ctx["context"].cat.codes,
#     y=means_ctx["mean_rho"],
#     color="black",
#     s=30,
#     zorder=5,
#     label="Mean"
# )

# # 6) Tidy up
# plt.xticks(range(len(ctx_order)), ctx_order, rotation=30, ha="right")
# plt.xlabel("Context")
# plt.ylabel("Spearman ρ")
# plt.title("Spearman correlation by Context (colored by LLM)")
# plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

# plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/llm_inside_set_correlation_2.png", dpi=300)
