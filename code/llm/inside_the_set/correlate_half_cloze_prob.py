import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pdb

CSV_PATH   = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/inside_the_set/Generative_Data_RAW.csv"
N_SPLITS   = 100
CLOZE_DEN  = 26
RANDOM_SEED = 42

# CONTEXTS = [
#     "fridge","handbag","mall","bakery","play","corner",
#     "restaurant","fitness","library","garage","closet"
# ]

CONTEXTS = ["library"]

def normalize_word(x):
    if pd.isna(x): return ""
    return str(x).strip().casefold()

def vec_spearman(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan
    if (a == a[0]).all() and (b == b[0]).all(): return np.nan, np.nan
    rho, p = spearmanr(a, b, nan_policy="raise")
    return float(rho), float(p)

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(CSV_PATH)

    # Filter: positive == TRUE
    pos_mask = df["positive"] == True
    df_pos = df[pos_mask].reset_index(drop=True)

    # Build per-context column groups (1..6); ignore *_time
    ctx_cols = {ctx: [f"{ctx}{i}" for i in range(1,7)] for ctx in CONTEXTS}
    n = len(df_pos)
    
    records = []
    for split_id in range(N_SPLITS):
        perm = rng.permutation(n)
        A_idx = perm[: n // 2]
        B_idx = perm[n // 2 :]

        A = df_pos.iloc[A_idx].reset_index(drop=True)
        B = df_pos.iloc[B_idx].reset_index(drop=True)

        for ctx in CONTEXTS:
            cols = ctx_cols[ctx]

            # collect all words across the 6 columns for this context
            words_A = (
                pd.Series(np.concatenate([A[c].astype(str).values for c in cols], axis=0), dtype="object")
                .map(lambda x: x if x != "nan" else np.nan)
                .dropna()
                .map(normalize_word)
            )
            words_B = (
                pd.Series(np.concatenate([B[c].astype(str).values for c in cols], axis=0), dtype="object")
                .map(lambda x: x if x != "nan" else np.nan)
                .dropna()
                .map(normalize_word)
            )

            words_A = words_A[words_A != ""]
            words_B = words_B[words_B != ""]

            cnt_A = words_A.value_counts()
            cnt_B = words_B.value_counts()

            vocab = sorted(set(cnt_A.index).union(cnt_B.index))

            # align and add-one to every type in the union
            cnt_A_s = cnt_A.reindex(vocab, fill_value=0).astype(float) + 1.0
            cnt_B_s = cnt_B.reindex(vocab, fill_value=0).astype(float) + 1.0

            # turn into probabilities (sum to 1). For Spearman, scaling doesn’t matter,
            # but zeros -> ones and renorm is the typical Laplace form.
            pA = cnt_A_s / CLOZE_DEN
            pB = cnt_B_s / CLOZE_DEN

            pdb.set_trace()

            rho, p = vec_spearman(pA.values, pB.values)

            #variance of the spearman correlation
            var_rho = np.var([vec_spearman(pA.values, pB.values)[0] for _ in range(100)])

            records.append({
                "split_id": split_id,
                "context": ctx,
                "spearman_rho": rho,
                "p_value": p,
                "variance_rho": var_rho
            })

    results = pd.DataFrame(records).sort_values(["split_id","context"]).reset_index(drop=True)

    # Optional summary per context
    summary = (
        results.groupby("context", as_index=False)
        .agg(mean_rho=("spearman_rho","mean"),
             sd_rho=("spearman_rho","std"),
             var_rho=("spearman_rho", "var"),
             valid_splits=("spearman_rho", lambda s: s.notna().sum()))
        .sort_values("context")
    )

    # print("\n=== Results (head) ===")
    # print(results.head(22).to_string(index=False))
    # print("\n=== Summary ===")
    # print(summary.to_string(index=False))

    # results.to_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/split_half_spearman_by_context.csv", index=False)
    # summary.to_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/split_half_spearman_summary.csv", index=False)
    # print("\nSaved: split_half_spearman_by_context.csv, split_half_spearman_summary.csv")
    
    # # After you’ve built the `results` DataFrame of per-split × context correlations:

    # # 1. Compute mean correlation across contexts for each split
    # mean_by_split = (
    #     results.groupby("split_id", as_index=False)
    #         .agg(mean_rho=("spearman_rho","mean"))
    # )

    # # 2. Plot histogram
    # plt.figure(figsize=(7,5))
    # plt.hist(mean_by_split["mean_rho"].dropna(), bins=20, edgecolor="k")
    # plt.title("Distribution of mean Spearman correlations\n(across contexts, per split)")
    # plt.xlabel("Mean Spearman rho")
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/figures/hist_mean_spearman_by_split.png", dpi=160)
    # plt.show()

    # # Optional: also save the table of mean correlations
    # mean_by_split.to_csv("/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/mean_spearman_by_split.csv", index=False)

if __name__ == "__main__":
    main()
