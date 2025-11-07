import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pdb
import seaborn as sns
from matplotlib.lines import Line2D

# ---------- CONFIG ----------
INPUT_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/external_cloze_prob_dataset/llm_next_word_from_external_cloze.csv"  
OUT_PER_SENT = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/external_cloze_prob_dataset/per_sentence_spearman.csv"
OUT_SUMMARY  = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/external_cloze_prob_dataset/mean_spearman_by_llm.csv"
OUT_PLOT     = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/llm/external_cloze_prob_dataset_mean_spearman_by_llm_dbbtalk2.png"

MIN_ITEMS_PER_SENTENCE = 3   # require at least this many word candidates in a sentence
# ----------------------------

def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho with guards for NaNs and constant vectors."""
    a = pd.to_numeric(x, errors="coerce").to_numpy()
    b = pd.to_numeric(y, errors="coerce").to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2 or b.size < 2:
        return np.nan
    # constant vectors -> undefined
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    rho, _ = spearmanr(a, b, nan_policy="raise")
    return float(rho)

def main():
#     df = pd.read_csv(INPUT_CSV)

#     # Compute per (llm_name, sentence) Spearman over words
#     rows = []
#     for (llm, sent), group in df.groupby(["llm_name", "sentence"], sort=False):
#         unique_llm_sent = group.copy()
#         unique_llm_sent = unique_llm_sent[
#             np.isfinite(pd.to_numeric(unique_llm_sent["human_cloze_prob"], errors="coerce")) & 
#             np.isfinite(pd.to_numeric(unique_llm_sent["llm_next_word_prob"], errors="coerce"))
#             ]

#         n_items = len(unique_llm_sent)
#         if n_items < MIN_ITEMS_PER_SENTENCE:
#             continue

#         rho = safe_spearman(unique_llm_sent["human_cloze_prob"], unique_llm_sent["llm_next_word_prob"])
#         rows.append({
#             "llm_name": llm,
#             "sentence": sent,
#             "n_items": n_items,
#             "spearman_rho": rho
#         })

#     per_sent = pd.DataFrame(rows)
#     per_sent.to_csv(OUT_PER_SENT, index=False)
#     print(f"Wrote per-sentence results: {OUT_PER_SENT} ({len(per_sent)} rows)")

#     # pdb.set_trace()

#     # Average Spearman per LLM (simple mean over sentences)
#     summary = (
#         per_sent.groupby("llm_name", as_index=False)
#         .agg(mean_spearman=("spearman_rho", "mean"),
#              sentences_evaluated=("spearman_rho", lambda s: s.notna().sum()))
#         .sort_values("mean_spearman", ascending=False)
#         .reset_index(drop=True)
#     )
    # summary.to_csv(OUT_SUMMARY, index=False)
    # print(f"Wrote summary: {OUT_SUMMARY}")

    per_sent = pd.read_csv(OUT_PER_SENT)
    summary  = pd.read_csv(OUT_SUMMARY)

    order_llm = summary["llm_name"].tolist()
    per_sent["llm_name"] = pd.Categorical(per_sent["llm_name"], categories=order_llm, ordered=True)
    summary["llm_name"]  = pd.Categorical(summary["llm_name"],  categories=order_llm, ordered=True)

    # --- plot ---
    plt.figure(figsize=(10, 6))

    # 1) Violin per model (distribution of per-sentence Spearman)
    sns.violinplot(
        data=per_sent,
        x="llm_name",
        y="spearman_rho",
        inner=None,   # show median & quartiles inside
        cut=0,              # don't extend past min/max
        scale="width",
        color="lightgray",  # violin fill
        linewidth=0,      # outline
        edgecolor="black"
    )

    # 2) Overlay black dot for the mean per model
    plt.scatter(
        x=summary["llm_name"].cat.codes,
        y=summary["mean_spearman"],
        color="black",
        s=80,              # bigger mean marker
        marker="D",         # diamond (pick any marker you like)
        zorder=3
    )

    # 2b) Annotate each mean with text above the dot
    for i, row in summary.iterrows():
        plt.text(
            x=row["llm_name"].cat.codes[0] if hasattr(row["llm_name"], "cat") else i,
            y=row["mean_spearman"] + 0.1,           # position slightly above the dot
            s=f"{row['mean_spearman']:.2f}",         # round to 2 decimals
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            fontweight="bold"
        )

    # 4) Legend (just the mean marker)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Mean', markerfacecolor='black', markersize=8)
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left")

    # 5) Axes & save
    plt.xticks(range(len(order_llm)), order_llm, rotation=30, ha="right", fontsize=14)
    plt.xlabel("LLM", fontsize=14)
    plt.ylabel("Spearman Correlation\n(per sentence)", fontsize=14)
    plt.title("Peelle et al (2020) Cloze Probability Dataset\nDistribution of the spearman correlations for each sentence by LLM\nMean spearman correlation indicated by black diamond", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    plt.show()

    print(f"Saved plot: {OUT_PLOT}")
    


    # # Plot mean Spearman by LLM (sorted)
    # plt.figure(figsize=(9, 5))
    # x = np.arange(len(summary))
    # y = summary["mean_spearman"].to_numpy()
    # plt.scatter(x, y)
    # plt.axhline(0.0, linestyle="--", linewidth=1)

    # plt.xticks(x, summary["llm_name"], rotation=30, ha="right")
    # plt.ylabel("Mean Spearman ρ (per-sentence correlations)")
    # plt.xlabel("LLM")
    # plt.title("Mean per-sentence Spearman by LLM (higher is better)")
    # plt.tight_layout()
    # plt.savefig(OUT_PLOT, dpi=300)
    # plt.show()
    # print(f"Saved plot: {OUT_PLOT}")

if __name__ == "__main__":
    main()
