#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split-half Spearman correlations: Human cloze halves vs LLM next-word probabilities,
using a HUMAN->LLM context map supplied by the user. We invert it internally
to look up the human context for each LLM context.

Outputs:
  1) llm_human_half_spearman_per_split.csv
  2) llm_human_half_spearman_summary.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ----------------------
# File paths
# ----------------------
LLM_RESULTS_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/results_llm_next_word_probs.csv"
HUMAN_RAW_CSV   = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/data/Generative_Data_RAW.csv"

OUT_PER_SPLIT   = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/llm_human_half_spearman_per_split.csv"
OUT_SUMMARY     = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/llm_human_half_spearman_summary.csv"

# ----------------------
# Settings
# ----------------------
N_SPLITS     = 50
RANDOM_SEED  = 42
DROP_NAN_LLM = True
N_COLS_PER_CONTEXT = 6  # expects human csv to have <human_ctx>1..6

# USER MAP: left = human context label, right = LLM context label
CONTEXT_MAP_HUMAN_TO_LLM = {
    "handbag": "bag",
    "corner":  "salad",
    "fitness": "gym",
    "library": "science",
    "garage":  "throw",
    "closet":  "beach",
}
# We invert it so we can go llm_ctx -> human_ctx during correlation
CONTEXT_MAP_LLM_TO_HUMAN = {llm: human for human, llm in CONTEXT_MAP_HUMAN_TO_LLM.items()}

def normalize_word(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().casefold()

def vec_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman rho with guards."""
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    rho, _ = spearmanr(a, b, nan_policy="omit")
    return float(rho)

def build_human_probs(df_half: pd.DataFrame, human_ctx: str) -> pd.Series:
    """
    Normalized human cloze distribution for one half and human context.
    Denominator = (#rows in half) * N_COLS_PER_CONTEXT.
    """
    cols = [f"{human_ctx}{i}" for i in range(1, N_COLS_PER_CONTEXT + 1)]
    # Flatten, clean, drop empties
    words = (
        pd.Series(np.concatenate([df_half[c].astype(str).values for c in cols], axis=0), dtype="object")
        .map(lambda s: None if s == "nan" else s)
        .dropna()
        .map(normalize_word)
    )
    words = words[words != ""]
    if words.empty:
        return pd.Series(dtype=float)
    counts = words.value_counts()
    denom = float(len(df_half) * N_COLS_PER_CONTEXT)
    return (counts / denom).astype(float)

def main():
    # Load LLM results
    df_llm = pd.read_csv(LLM_RESULTS_CSV)
    req = {"llm_name", "context", "word", "llm_next_word_prob"}
    if not req.issubset(df_llm.columns):
        raise ValueError(f"LLM CSV must have columns: {req}")
    if DROP_NAN_LLM:
        df_llm = df_llm.dropna(subset=["llm_next_word_prob"]).copy()

    df_llm["word_norm"] = df_llm["word"].map(normalize_word)
    df_llm["context"]   = df_llm["context"].astype(str)

    # Load human data and filter positives
    df_h = pd.read_csv(HUMAN_RAW_CSV)
    if "positive" not in df_h.columns:
        raise ValueError("Human CSV must contain a 'positive' column (True/False).")
    pos_mask = df_h["positive"].astype(str).str.strip().str.lower().isin(["true","1","t","yes","y"])
    df_h = df_h[pos_mask].reset_index(drop=True)
    if df_h.empty:
        raise ValueError("No rows with positive==TRUE in human CSV.")

    # Verify human columns exist (for any human context we might map to)
    # Collect all possible human ctx names: identity + inverse mapping
    llm_contexts = sorted(df_llm["context"].unique().tolist())
    possible_human_ctxs = set(CONTEXT_MAP_LLM_TO_HUMAN.values())
    # If some LLM contexts are not in the mapping, assume same name in human CSV
    for llm_ctx in llm_contexts:
        if llm_ctx not in CONTEXT_MAP_LLM_TO_HUMAN:
            possible_human_ctxs.add(llm_ctx)

    missing_any = False
    for human_ctx in sorted(possible_human_ctxs):
        cols = [f"{human_ctx}{i}" for i in range(1, N_COLS_PER_CONTEXT + 1)]
        missing = [c for c in cols if c not in df_h.columns]
        if missing:
            print(f"[warn] Human CSV missing columns for human context '{human_ctx}': {missing}")
            missing_any = True
    if missing_any:
        print("[warn] Some human contexts are missing; corresponding correlations may be NaN.")

    # Pre-index LLM probabilities as Series per (llm, llm_context)
    llm_series = {}
    for (llm, ctx), g in df_llm.groupby(["llm_name", "context"], sort=False):
        s = (
            g.set_index("word_norm")["llm_next_word_prob"]
             .astype(float)
             .groupby(level=0).max()  # if duplicates, take max; change to mean if you prefer
        )
        llm_series[(llm, ctx)] = s

    # Split halves
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df_h)
    if n < 2:
        raise ValueError("Not enough human rows to split.")

    records = []
    for split_id in range(N_SPLITS):
        perm = rng.permutation(n)
        A_idx = perm[: n // 2]
        B_idx = perm[n // 2 :]
        halves = [("A", df_h.iloc[A_idx].reset_index(drop=True)),
                  ("B", df_h.iloc[B_idx].reset_index(drop=True))]

        for half_label, df_half in halves:
            # Cache human distributions by human_ctx to avoid recompute
            human_cache = {}

            # For each LLM×context pair, map to human context and compute Spearman
            for (llm, llm_ctx), s_llm in llm_series.items():
                human_ctx = CONTEXT_MAP_LLM_TO_HUMAN.get(llm_ctx, llm_ctx)  # invert map in use
                if human_ctx not in human_cache:
                    human_cache[human_ctx] = build_human_probs(df_half, human_ctx)

                s_h = human_cache[human_ctx]
                # Align to the LLM word set (fill missing human words with 0)
                s_h_aligned = s_h.reindex(s_llm.index, fill_value=0.0)

                if s_h_aligned.empty or s_llm.empty:
                    rho = np.nan
                else:
                    rho = vec_spearman(s_h_aligned.values, s_llm.values)

                records.append({
                    "split_id": split_id,
                    "half": half_label,
                    "llm_name": llm,
                    "llm_context": llm_ctx,
                    "human_context": human_ctx,
                    "n_words_llm": int(s_llm.size),
                    "n_words_human_nonzero": int((s_h_aligned > 0).sum()),
                    "spearman_rho": rho,
                })

    per_split = pd.DataFrame(records).sort_values(
        ["split_id","half","llm_name","llm_context"]
    ).reset_index(drop=True)
    per_split.to_csv(OUT_PER_SPLIT, index=False)

    summary = (
        per_split.groupby("llm_name", as_index=False)
        .agg(mean_spearman=("spearman_rho","mean"))
        .sort_values("mean_spearman", ascending=False)
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"[done] Wrote {len(per_split)} rows -> {OUT_PER_SPLIT}")
    print(f"[done] Wrote {len(summary)} rows  -> {OUT_SUMMARY}")
    print("\nMapping (HUMAN → LLM):", CONTEXT_MAP_HUMAN_TO_LLM)
    print("Used inverse (LLM → HUMAN):", CONTEXT_MAP_LLM_TO_HUMAN)
    print("\nLeaderboard (mean Spearman):\n", summary.to_string(index=False))

if __name__ == "__main__":
    main()
