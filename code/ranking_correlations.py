# Build the two requested dataframes:
# 1) word_comparison_summary: per-context counts + Spearman (drop-missing), sorted by Spearman
# 2) word_comparison_detailed: ONLY human words, with columns
#    [context, cleaned_trigger, human_position, bert_position, bert_prob],
#    ordered by context (following summary order) and human_position ascending.
#
# Inputs expected at:
# - /mnt/data/query_positions_results.csv  (columns: context, query, position, probability)
# - /mnt/data/sca_dataframe.csv            (columns: context or story, cleaned_trigger, trigger_relevance)
#
# Outputs:
# - /mnt/data/word_comparison_summary.csv
# - /mnt/data/word_comparison_detailed.csv
#
# Notes:
# - human trigger_relevance is already the final rank: 0=best, 1=next, ...
# - "n_words_overlap" = number of overlapping words with BERT (non-null bert_position)
# - Spearman computed on DROP-MISSING only (overlap set)

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ---- Load
bert_path  = "../data/query_positions_results.csv"
human_path = "../data/sca_dataframe.csv"

bert_df  = pd.read_csv(bert_path)
human_df = pd.read_csv(human_path) 

# ---- Normalize columns
if "context" not in human_df.columns and "story" in human_df.columns:
    human_df = human_df.rename(columns={"story": "context"})

# columns sanity
expected_human_cols = ["context", "cleaned_trigger", "trigger_relevance"]
keep_human = [c for c in expected_human_cols if c in human_df.columns]
human_df = human_df[keep_human].drop_duplicates()

# enforce numeric for ranks
human_df["trigger_relevance"] = pd.to_numeric(human_df["trigger_relevance"], errors="coerce")
human_df = human_df.dropna(subset=["trigger_relevance"])  # must have a human rank

# Treat human rank directly
human_df["human_position"] = human_df["trigger_relevance"].astype(int)

# BERT
keep_bert = [c for c in ["context", "query", "position", "probability"] if c in bert_df.columns]
bert_df = bert_df[keep_bert].drop_duplicates()
bert_df["position"] = pd.to_numeric(bert_df["position"], errors="coerce")
bert_df["probability"] = pd.to_numeric(bert_df["probability"], errors="coerce")

# ---- LEFT JOIN: restrict to human words only
merged = pd.merge(
    human_df.rename(columns={"cleaned_trigger": "word"}),
    bert_df.rename(columns={"query": "word", "position": "bert_position", "probability": "bert_prob"}),
    on=["context", "word"],
    how="left"
)

# Ensure consistent dtypes
merged["bert_position"] = pd.to_numeric(merged["bert_position"], errors="coerce")
merged["bert_prob"] = pd.to_numeric(merged["bert_prob"], errors="coerce")

# ---- Compute per-context Spearman on overlap only
def _spearman_drop(group: pd.DataFrame) -> float:
    g = group.dropna(subset=["bert_position"])
    if len(g) < 2:
        return np.nan
    try:
        rho, _ = spearmanr(g["human_position"].astype(float), g["bert_position"].astype(float))
        return float(rho) if pd.notna(rho) else np.nan
    except Exception:
        return np.nan

summary = (
    merged
    .groupby("context", as_index=False)
    .agg(
        n_human_words=("word", "size"),
        n_words_overlap=("bert_position", lambda s: s.notna().sum())
    )
)

# add spearman_drop
spearman_vals = merged.groupby("context").apply(_spearman_drop).rename("spearman_drop").reset_index()
summary = summary.merge(spearman_vals, on="context", how="left")

# sort by spearman_drop descending (best first)
summary = summary.sort_values(["spearman_drop", "context"], ascending=[False, True]).reset_index(drop=True)

# ---- Build detailed dataframe with the requested columns and ordering
detailed = merged[["context", "word", "human_position", "bert_position", "bert_prob"]].copy()
detailed = detailed.rename(columns={"word": "cleaned_trigger"})

# order contexts by summary order, and within each context by human_position ascending
context_order = summary["context"].tolist()
detailed["context"] = pd.Categorical(detailed["context"], categories=context_order, ordered=True)
detailed = detailed.sort_values(["context", "human_position", "cleaned_trigger"]).reset_index(drop=True)

# ---- Save outputs
summary_out = "../results/word_comparison_summary.csv"
detailed_out = "../results/word_comparison_detailed.csv"
summary.to_csv(summary_out, index=False)
detailed.to_csv(detailed_out, index=False)

# ---- Display to user
summary_out, detailed_out