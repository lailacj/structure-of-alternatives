# -*- coding: utf-8 -*-

"""
Compute rank-similarity between human cloze probabilities and LLM next-word probabilities.

Input CSV must have columns:
  llm_name, context, word, cloze_prob, llm_next_word_prob, llm_log_prob  (log not used here)

Outputs:
  - per_context_rank_agreement.csv  (one row per llm_name x context)
  - summary_by_llm.csv              (aggregate across contexts)

Notes:
  - We use rank-based metrics:
      * Spearman's rho (monotonic rank agreement)
      * Kendall's tau (pairwise order agreement)
    These are the most standard “similarity ranking” measures.
  - We also report “top1_match” per context: whether argmax_llm == argmax_cloze.
  - Groups with fewer than MIN_ITEMS are skipped (correlations are unstable).
"""

import pandas as pd
import numpy as np

# ---------- CONFIG ----------
INPUT_CSV = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/results_llm_next_word_probs.csv"
MIN_ITEMS = 3               # minimum candidates in a context to compute correlations
DROP_NAN_PROBS = True       # drop rows with NaN probs (e.g., BERT multi-piece)
OUT_PER_CONTEXT = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/per_context_rank_agreement.csv"
OUT_SUMMARY = "/users/ljohnst7/data/ljohnst7/structure-of-alternatives/results/llm/summary_by_llm.csv"
# ---------------------------

def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """Pandas' corr(method='spearman') handles ties; returns np.nan if not enough variance."""
    try:
        return float(a.corr(b, method="spearman"))
    except Exception:
        return np.nan

def safe_kendall(a: pd.Series, b: pd.Series) -> float:
    """Pandas' corr(method='kendall') handles ties; returns np.nan if not enough variance."""
    try:
        return float(a.corr(b, method="kendall"))
    except Exception:
        return np.nan

def top1_match(cloze: pd.Series, llm: pd.Series) -> float:
    """1 iff argmax aligns (ties broken by first occurrence)."""
    if cloze.empty or llm.empty:
        return np.nan
    return float(cloze.idxmax() == llm.idxmax())

def main():
    df = pd.read_csv(INPUT_CSV)

    # Basic hygiene
    required = {"llm_name", "context", "word", "cloze_prob", "llm_next_word_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    # Optionally drop NaNs (e.g., BERT multi-piece words); otherwise correlations will become NaN
    if DROP_NAN_PROBS:
        df = df.dropna(subset=["cloze_prob", "llm_next_word_prob"]).copy()

    # Compute per (llm_name, context)
    rows = []
    for (llm, ctx), g in df.groupby(["llm_name", "context"], sort=False):
        # Keep only finite values
        gg = g[np.isfinite(g["cloze_prob"]) & np.isfinite(g["llm_next_word_prob"])].copy()
        n = len(gg)
        if n < MIN_ITEMS:
            continue

        # Rank correlations (Spearman/Kendall)
        s_rho = safe_spearman(gg["cloze_prob"], gg["llm_next_word_prob"])
        k_tau = safe_kendall(gg["cloze_prob"], gg["llm_next_word_prob"])

        # Top-1 agreement
        # Use positions; if duplicates in words, idxmax uses first occurrence (fine for our purpose)
        t1 = top1_match(gg["cloze_prob"], gg["llm_next_word_prob"])

        rows.append({
            "llm_name": llm,
            "context": ctx,
            "n_items": n,
            "spearman_rho": s_rho,
            "kendall_tau": k_tau,
            "top1_match": t1,
            # (optional) raw sums for weighting
            "sum_cloze": gg["cloze_prob"].sum(),
            "sum_llm": gg["llm_next_word_prob"].sum(),
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
    for llm, g in per_ctx.groupby("llm_name", sort=False):
        # Simple means across contexts (ignore NaNs)
        mean_rho = float(np.nanmean(g["spearman_rho"])) if not g["spearman_rho"].empty else np.nan
        mean_tau = float(np.nanmean(g["kendall_tau"])) if not g["kendall_tau"].empty else np.nan
        mean_top1 = float(np.nanmean(g["top1_match"])) if not g["top1_match"].empty else np.nan

        # Weighted by number of items in each context
        w_rho = wmean(g["spearman_rho"], g["n_items"])
        w_tau = wmean(g["kendall_tau"], g["n_items"])
        w_top1 = wmean(g["top1_match"], g["n_items"])

        summaries.append({
            "llm_name": llm,
            "contexts_evaluated": int(len(g)),
            "mean_spearman": mean_rho,
            "mean_kendall": mean_tau,
            "mean_top1": mean_top1,
            "weighted_spearman": w_rho,
            "weighted_kendall": w_tau,
            "weighted_top1": w_top1,
        })

    summary = pd.DataFrame(summaries)
    # Sort by the headline metric you care about (Spearman weighted)
    summary = summary.sort_values(by="weighted_spearman", ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Wrote {len(per_ctx)} per-context rows -> {OUT_PER_CONTEXT}")
    print(f"Wrote {len(summary)} model summaries   -> {OUT_SUMMARY}")
    print("\nLeaderboard by weighted Spearman:")
    print(summary[["llm_name", "weighted_spearman", "weighted_kendall", "weighted_top1"]].to_string(index=False))

if __name__ == "__main__":
    main()
