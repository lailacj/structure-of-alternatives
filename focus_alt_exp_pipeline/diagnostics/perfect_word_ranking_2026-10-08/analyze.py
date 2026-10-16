"""Descriptive exclusion correlations conditional on word-rank agreement.

Uses corrected, saved held-out predictions; no model or boundary is refitted.
Run with the bundled Python runtime (numpy and pandas).
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PIPELINE = OUT.parents[1]
RESULTS = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15"
FILES = {
    "word_correlations": RESULTS / "spearman/word_spearman_by_context.csv",
    "word_ranks": RESULTS / "spearman/word_paired_ranks.csv",
    "predictions": RESULTS / "spearman/negation_paired_ranks.csv",
    "source_rows": RESULTS / "inputs/source_rows.csv",
    "human": PIPELINE / "human_exp_data/sca_dataframe.csv",
}
LABELS = [
    ("No linking structure", "direct", "No linking"),
    ("X but not Y", "direct", "X but not Y"),
    ("set", "top_k", "Set Top-K"),
    ("set", "top_p", "Set Top-p"),
    ("ordering", "top_k", "Ordering"),
    ("conjunction", "top_k", "Conjunction Top-K"),
    ("conjunction", "top_p", "Conjunction Top-p"),
    ("disjunction", "top_k", "Disjunction Top-K"),
    ("disjunction", "top_p", "Disjunction Top-p"),
]


def correlations(rows):
    x = rows.model_value.astype(float)
    y = rows.human_rate.astype(float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if x.nunique() < 2 or y.nunique() < 2:
        return {"pearson_r": np.nan, "spearman_rho": np.nan,
                "status": "constant_vector"}
    return {"pearson_r": float(x.corr(y)),
            "spearman_rho": float(x.rank(method="average").corr(y.rank(method="average"))),
            "status": "defined"}


def main(word_rho_gt=None, output_dir=OUT):
    output_dir = Path(output_dir)
    if word_rho_gt is not None:
        if not -1 <= word_rho_gt < 1:
            raise ValueError("The strict word-rho cutoff must be in [-1, 1).")
        if output_dir.resolve() == OUT.resolve():
            raise ValueError("Choose a separate output directory to preserve the perfect-ranking analysis.")
    wc = pd.read_csv(FILES["word_correlations"])
    wr = pd.read_csv(FILES["word_ranks"])
    pred = pd.read_csv(FILES["predictions"])
    source = pd.read_csv(FILES["source_rows"])
    human = pd.read_csv(FILES["human"])
    # Reconstruct the human word ranks used by the established analysis.
    norms = human[["story", "cleaned_trigger", "trigger_relevance"]].drop_duplicates()
    norms = norms.rename(columns={"story": "context", "cleaned_trigger": "word"})
    verified = wr.merge(norms, on=["context", "word"], validate="one_to_one")
    assert len(verified) == 96
    assert np.array_equal(verified.human_rank, verified.trigger_relevance)
    for context, group in verified.groupby("context"):
        assert len(group) == 6
        model_rank = group.model_value.rank(ascending=False, method="average") - 1
        assert np.allclose(model_rank, group.model_rank)
        rho = group.human_rank.corr(model_rank)
        saved = wc.loc[wc.context.eq(context), "spearman_rho"].item()
        assert np.isclose(rho, saved, rtol=0, atol=1e-12)
    if word_rho_gt is None:
        # A tight numerical tolerance handles roundoff, not near-perfect rankings.
        selected = sorted(wc.loc[np.isclose(wc.spearman_rho, 1, rtol=0, atol=1e-12), "context"])
        assert selected == ["bakery", "fridge"]
        exact = verified[verified.context.isin(selected)]
        assert np.array_equal(exact.human_rank, exact.model_rank)
        selection_description = "Word Spearman equals 1 within absolute roundoff tolerance 1e-12; exact equality of six human/model rank vectors verified."
    else:
        selected = sorted(wc.loc[wc.spearman_rho.gt(word_rho_gt), "context"])
        selection_description = f"Word Spearman strictly greater than {word_rho_gt}; comparison uses unrounded coefficients."
    if not selected:
        raise ValueError("No contexts satisfy the selected criterion.")
    selected_words = verified[verified.context.isin(selected)]
    n_pairs = 30 * len(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    wc["selected"] = wc.context.isin(selected)
    wc.to_csv(output_dir / "context_selection.csv", index=False)

    h = human.groupby(["story", "cleaned_trigger", "cleaned_query"]).neg.agg(["mean", "size"])
    subset = pred[pred.context.isin(selected)].copy()
    for row in subset.itertuples():
        actual = h.loc[(row.context, row.trigger, row.query)]
        assert abs(row.human_rate - actual["mean"]) < 1e-12
        assert row.human_total == actual["size"]
    # Ordering appears in both variant exports but is the same predictor.
    ordering = pred[pred.structure.eq("ordering")].pivot(index="item_id", columns="variant", values="model_value")
    assert np.array_equal(ordering.top_k, ordering.top_p)
    base = subset[subset.structure.eq("No linking structure")]
    raw = source[source.dataset_family.eq("novel_focus")]
    check = base.merge(raw, on="item_id", validate="one_to_one")
    assert np.allclose(check.model_value, np.exp(check.query_logprob_sum), rtol=1e-12, atol=0)

    summary, contexts, saved_items = [], [], []
    for structure, variant, label in LABELS:
        full = pred[pred.structure.eq(structure) & pred.variant.eq(variant)]
        keep = full[full.context.isin(selected)].copy()
        assert len(keep) == n_pairs and keep.item_id.nunique() == n_pairs
        saved_items.append(keep.assign(predictor=label))
        local = []
        for context, rows in keep.groupby("context"):
            assert len(rows) == 30
            result = {"model": label, "context": context, "n_pairs": len(rows), **correlations(rows)}
            contexts.append(result)
            local.append(result)
        means = pd.DataFrame(local)
        pooled = correlations(keep)
        all_contexts = pd.DataFrame([correlations(g) for _, g in full.groupby("context")])
        summary.append({
            "model": label, "n_contexts": len(selected), "n_pairs": len(keep),
            "valid_contexts": int(means.pearson_r.notna().sum()),
            "mean_within_pearson": means.pearson_r.mean(),
            "mean_within_spearman": means.spearman_rho.mean(),
            "pooled_pearson": pooled["pearson_r"],
            "pooled_spearman": pooled["spearman_rho"],
            "all_contexts_mean_pearson": all_contexts.pearson_r.mean(),
            "all_contexts_mean_spearman": all_contexts.spearman_rho.mean(),
            "all_contexts_valid_n": int(all_contexts.pearson_r.notna().sum()),
        })
    summary = pd.DataFrame(summary)
    for metric in ["mean_within_pearson", "mean_within_spearman", "pooled_pearson", "pooled_spearman"]:
        summary["delta_vs_no_linking_" + metric] = summary[metric] - summary.iloc[0][metric]
    summary.to_csv(output_dir / "correlation_summary.csv", index=False)
    pd.DataFrame(contexts).to_csv(output_dir / "correlations_by_context.csv", index=False)
    pd.concat(saved_items).to_csv(output_dir / "selected_predictions.csv", index=False)
    provenance = {
        "selected_contexts": selected, "selected_context_count": len(selected),
        "total_context_count": len(wc), "n_distinct_pairs": n_pairs,
        "n_human_responses": int(base.human_total.sum()),
        "word_rank_definition": "Human trigger_relevance norming ranks versus descending summed neutral continuation log probabilities, over six tested words per context; not cloze frequency correlations.",
        "selection": selection_description,
        "evaluation": f"Pearson and average-tie Spearman of model exclusion probabilities against observed item rates. Equal item weights; arithmetic mean of defined per-context correlations. Pooled Spearman reranks all {n_pairs} items together.",
        "predictions": "Original corrected held-out predictions retained; no boundary selection or refit on the selected subset. Identical Ordering variants deduplicated.",
        "limitations": f"{len(selected)} selected contexts. Descriptive selected-subset analysis, not an equivalence/significance test or evidence that human/model probabilities or full distributions agree. Undefined constant-vector correlations are excluded from within-context means, with valid counts reported.",
        "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in FILES.items()},
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "inputs"}, indent=2))
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(pd.DataFrame(contexts).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(selected_words.sort_values(["context", "human_rank"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-rho-gt", type=float, default=None,
                        help="Strict word-ranking Spearman lower bound; omit for perfect agreement.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.word_rho_gt, args.output_dir)
