# -*- coding: utf-8 -*-

"""
Compute rank-similarity between human cloze probabilities and LLM next-word probabilities.

Input CSV must have columns:
  llm_name, context, word, cloze_prob, llm_next_word_prob, llm_log_prob  (log not used here)

Outputs:
  - per_context_rank_agreement.csv  (one row per llm_name x context)
  - summary_by_llm.csv              (aggregate across contexts)
"""

import pandas as pd
import numpy as np
import pdb

# ---------- CONFIG ----------
INPUT_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/results_llm_next_word_probs.csv"
MIN_ITEMS = 3               # minimum candidates in a context to compute correlations
DROP_NAN_PROBS = False       
OUT_PER_CONTEXT = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/per_context_rank_agreement.csv"
OUT_SUMMARY = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/inside_the_set/summary_by_llm.csv"
# ---------------------------

def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """Pandas' corr(method='spearman') handles ties; returns np.nan if not enough variance."""
    try:
        return float(a.corr(b, method="spearman"))
    except Exception:
        return np.nan

def main():
    df = pd.read_csv(INPUT_CSV)

    # Compute per (llm_name, context)
    rows = []
    for (llm, ctx), group in df.groupby(["llm_name", "context"], sort=False):
        # Keep only finite values
        unique_llm_context = group[np.isfinite(group["cloze_prob"]) & np.isfinite(group["llm_next_word_prob"])].copy()
        
        n = len(unique_llm_context)
        if n < MIN_ITEMS:
            continue

        # Spearman rho
        s_rho = safe_spearman(unique_llm_context["cloze_prob"], unique_llm_context["llm_next_word_prob"])

        rows.append({
            "llm_name": llm,
            "context": ctx,
            "n_items": n,
            "spearman_rho": s_rho,
            # (optional) raw sums for weighting
            "sum_cloze": unique_llm_context["cloze_prob"].sum(),
            "sum_llm": unique_llm_context["llm_next_word_prob"].sum(),
        })

    per_ctx = pd.DataFrame(rows)
    per_ctx.to_csv(OUT_PER_CONTEXT, index=False)

    # Summary per model
    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return np.nan
        return np.average(x[mask], weights=w[mask])

    summaries = []
    for llm, context in per_ctx.groupby("llm_name", sort=False):
        # Simple means across contexts
        mean_rho = float(np.nanmean(context["spearman_rho"]))

        # Weighted by number of items in each context
        w_rho = wmean(context["spearman_rho"], context["n_items"])

        summaries.append({
            "llm_name": llm,
            "contexts_evaluated": int(len(context)),
            "mean_spearman": mean_rho,
            "weighted_spearman": w_rho,
        })

    summary = pd.DataFrame(summaries)
    # Sort by the headline metric you care about (Spearman weighted)
    summary = summary.sort_values(by="weighted_spearman", ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Wrote {len(per_ctx)} per-context rows -> {OUT_PER_CONTEXT}")
    print(f"Wrote {len(summary)} model summaries   -> {OUT_SUMMARY}")
    print("\nLeaderboard by weighted Spearman:")
    print(summary[["llm_name", "weighted_spearman"]].to_string(index=False))

if __name__ == "__main__":
    main()
