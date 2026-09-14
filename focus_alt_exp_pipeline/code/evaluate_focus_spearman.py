"""Within-context word-ranking and negation Spearman evaluation for novel focus.

Ranks use zero as best, with average ranks for ties. No pooled correlation or
parameter selection is performed here. Saved neutral whole-continuation summed
log probabilities rank the six tested words; held-out probabilities rank pairs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
HUMAN_FILE = PIPELINE / "human_exp_data/sca_dataframe.csv"
STRUCTURES = ("set", "ordering", "conjunction", "disjunction")


def rank_correlation(human, model):
    """Return coefficient and explicit status, rejecting incomplete observations."""
    x, y = np.asarray(human, dtype=float), np.asarray(model, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman requires aligned, finite vectors")
    if len(x) < 2:
        return np.nan, "fewer_than_two_items"
    if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
        return np.nan, "constant_human_or_model"
    return float(pd.Series(x).rank().corr(pd.Series(y).rank())), "defined"


def _ranks(values):
    return values.rank(method="average", ascending=False) - 1


def evaluate(human: pd.DataFrame, source: pd.DataFrame, oof: pd.DataFrame):
    """Return context results, paired ranks, and equal-context summaries."""
    human = human.rename(columns={"story": "context", "cleaned_trigger": "word"}).copy()
    for col in ("context", "word", "cleaned_query"):
        if human[col].isna().any():
            raise ValueError("Missing human context or word")
        human[col] = human[col].astype(str).str.strip().str.lower()
    human["trigger_relevance"] = pd.to_numeric(human["trigger_relevance"], errors="raise")
    human["neg"] = pd.to_numeric(human["neg"], errors="raise")
    if not human["neg"].isin([0, 1]).all():
        raise ValueError("Human neg must be binary")
    words = human[["context", "word", "trigger_relevance"]].drop_duplicates()
    if words.duplicated(["context", "word"]).any():
        raise ValueError("Inconsistent human word ranks")
    for context, rows in words.groupby("context"):
        if len(rows) != 6 or set(rows.trigger_relevance) != set(range(6)):
            raise ValueError(f"Expected six words ranked 0–5 in {context}")
    pairs = human.groupby(["context", "word", "cleaned_query"], as_index=False).agg(
        human_rate=("neg", "mean"), human_total=("neg", "size"))
    pairs = pairs.rename(columns={"word": "trigger", "cleaned_query": "query"})
    for context, rows in pairs.groupby("context"):
        vocab = set(words.loc[words.context.eq(context), "word"])
        expected = {(a, b) for a in vocab for b in vocab if a != b}
        if set(zip(rows["trigger"], rows["query"])) != expected:
            raise ValueError(f"Expected all 30 tested trigger–query pairs in {context}")

    source = source.loc[source.dataset_family.eq("novel_focus")].copy()
    if not source.generation_frame.eq("no_frame").all():
        raise ValueError("Word rankings require neutral context-only scores")
    if source.model_name.nunique() != 1 or source.model_revision.nunique() != 1:
        raise ValueError("Expected one model and revision")
    source = source.rename(columns={"context_id": "context"})
    for col in ("context", "trigger", "query"):
        source[col] = source[col].astype(str).str.strip().str.lower()
    if source.groupby("context").generation_prompt.nunique().ne(1).any():
        raise ValueError("Expected one neutral prompt per context")
    keys = ["context", "trigger", "query"]
    joined = pairs.merge(source, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_source"))
    if not joined._merge.eq("both").all() or not np.allclose(joined.human_rate, joined.human_rate_source):
        raise ValueError("Saved source coverage or human rates disagree with human CSV")
    model_name = str(source.model_name.iloc[0])
    candidates = pd.concat([
        source[["context", role, f"{role}_logprob_sum"]].rename(columns={role: "word", f"{role}_logprob_sum": "model_value"})
        for role in ("trigger", "query")], ignore_index=True)
    candidates.model_value = pd.to_numeric(candidates.model_value, errors="raise")
    if not np.isfinite(candidates.model_value).all() or candidates.model_value.gt(0).any():
        raise ValueError("Invalid whole-alternative log probabilities")
    if candidates.groupby(["context", "word"]).model_value.nunique().ne(1).any():
        raise ValueError("Inconsistent repeated neutral word scores")
    word_pairs = words.merge(candidates.drop_duplicates(), on=["context", "word"], how="outer", validate="one_to_one", indicator=True)
    if not word_pairs._merge.eq("both").all():
        raise ValueError("Missing or extra tested word scores")
    word_pairs = word_pairs.drop(columns="_merge").rename(columns={"trigger_relevance": "human_rank"})
    word_pairs["model_rank"] = word_pairs.groupby("context").model_value.transform(_ranks)
    word_pairs["model"] = model_name
    word_results = []
    for context, rows in word_pairs.groupby("context"):
        rho, status = rank_correlation(rows.human_rank, rows.model_rank)
        word_results.append(dict(context=context, model=model_name, n=len(rows), spearman_rho=rho, status=status))

    neg_frames = []
    for label, column in [("No linking structure", "query_logprob_sum"), ("X but not Y", "x_but_not_y_logprob_sum")]:
        frame = joined[keys + ["item_id", "human_rate", "human_total"]].copy()
        frame["model_value"] = np.exp(pd.to_numeric(joined[column], errors="raise"))
        frame["structure"], frame["variant"] = label, "direct"
        neg_frames.append(frame)
    focus = oof.loc[oof.analysis_dataset_id.eq("novel_focus")]
    if set(focus.variant) != {"top_k", "top_p"}:
        raise ValueError("Expected both held-out variants")
    for variant, predictions in focus.groupby("variant"):
        matched = joined.merge(predictions, left_on="item_id", right_on="analysis_unit_id", how="outer", validate="one_to_one", suffixes=("", "_oof"), indicator="oof_match")
        if not matched.oof_match.eq("both").all() or not np.allclose(matched.human_rate, matched.human_rate_oof):
            raise ValueError("Held-out coverage or human rates disagree")
        for structure in STRUCTURES:
            frame = matched[keys + ["item_id", "human_rate", "human_total", "cv_fold", "boundary"]].copy()
            frame["model_value"] = matched[f"{structure}_probability"]
            frame["structure"], frame["variant"] = structure, variant
            neg_frames.append(frame)
    neg_pairs = pd.concat(neg_frames, ignore_index=True)
    if not np.isfinite(neg_pairs.model_value).all() or not neg_pairs.model_value.between(0, 1).all():
        raise ValueError("Invalid negation predictions")
    neg_pairs["model"] = model_name
    group_keys = ["context", "model", "structure", "variant"]
    for col, rank in [("human_rate", "human_rank"), ("model_value", "model_rank")]:
        neg_pairs[rank] = neg_pairs.groupby(group_keys)[col].transform(_ranks)
    neg_results = []
    for key, rows in neg_pairs.groupby(group_keys):
        rho, status = rank_correlation(rows.human_rank, rows.model_rank)
        neg_results.append(dict(zip(group_keys, key), n=len(rows), spearman_rho=rho, status=status))
    word_results, neg_results = pd.DataFrame(word_results), pd.DataFrame(neg_results)
    summaries = []
    for measure, table, grouping in [("word_ranking", word_results, ["model"]), ("negation", neg_results, ["model", "structure", "variant"])]:
        summary = table.groupby(grouping, as_index=False).agg(
            mean_within_context_spearman=("spearman_rho", "mean"),
            valid_contexts=("spearman_rho", "count"), total_contexts=("context", "size"))
        summary["measure"] = measure
        summaries.append(summary)
    return {"word_spearman_by_context": word_results, "word_paired_ranks": word_pairs,
            "negation_spearman_by_context": neg_results, "negation_paired_ranks": neg_pairs,
            "mean_within_context_spearman": pd.concat(summaries, ignore_index=True)}


def write_results(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    summary = tables["mean_within_context_spearman"].fillna("")
    report = "# Within-context Spearman\n\nWord rankings use six tested alternatives and summed neutral continuation log probabilities. Negation rankings use all 30 ordered trigger–query pairs. Ties receive average ranks; constant vectors are undefined. Means weight valid contexts equally; valid/total counts are reported. Boundary selection still uses training log score.\n\n"
    report += "| " + " | ".join(summary.columns) + " |\n| " + " | ".join(["---"] * len(summary.columns)) + " |\n"
    report += "\n".join("| " + " | ".join(map(str, row)) + " |" for row in summary.itertuples(index=False, name=None))
    (output / "SPEARMAN.md").write_text(report + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-data", type=Path, default=HUMAN_FILE)
    parser.add_argument("--source-rows", type=Path, default=PIPELINE / "scoring_manifests/set_variant_qwen/source_rows.csv")
    parser.add_argument("--results-dir", type=Path, default=PIPELINE / "results/set_variant_qwen")
    args = parser.parse_args()
    tables = evaluate(pd.read_csv(args.human_data), pd.read_csv(args.source_rows), pd.read_csv(args.results_dir / "cv_results/oof_predictions.csv"))
    write_results(tables, args.results_dir / "spearman")
    print(tables["mean_within_context_spearman"].to_string(index=False))


if __name__ == "__main__":
    main()
