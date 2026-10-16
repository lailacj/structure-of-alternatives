"""Describe where saved linking predictions improve on the neutral baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from plot_four_model_fit_comparison import DEFAULT_RESULTS, DATASETS, METRICS, MODELS, read_fits

TOLERANCE = 1e-12


def classify(delta):
    if pd.isna(delta):
        return "unavailable"
    return "better" if delta > TOLERANCE else "worse" if delta < -TOLERANCE else "tied"


def differences(fits):
    baseline = fits.loc[fits.model.eq(MODELS[0]), ["metric", "dataset", "value"]].rename(columns={"value": "baseline"})
    joined = fits.loc[~fits.model.eq(MODELS[0])].merge(baseline, on=["metric", "dataset"], validate="many_to_one")
    joined["difference"] = joined.value - joined.baseline
    joined["comparison"] = joined.difference.map(classify)
    return joined


def draw(deltas, output):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 8.6), sharey=True)
    fig.subplots_adjust(left=.19, right=.975, top=.79, bottom=.19, wspace=.16)
    fig.text(.025, .95, "When do linking structures improve fit?", fontsize=23, weight="bold", color="#182733")
    fig.text(.025, .905, "Each cell = linking fit − no-linking fit. Positive values favor linking; negative values favor no linking.", fontsize=12, color="#52616B")
    order = [d for d, _ in DATASETS]
    for ax, (metric, (_, title, _)) in zip(axes, METRICS.items()):
        values = deltas.loc[deltas.metric.eq(metric)].pivot(index="dataset", columns="model", values="difference").loc[order, MODELS[1:]].to_numpy()
        limit = 4. if metric == "log_score" else .4
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#ECEFF1")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax.set_title(title, fontsize=15, pad=47)
        ax.set_xticks(range(3), ["X but not Y", "Ordering", "Set Top-K"], fontsize=10.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=8)
        ax.set_yticks(range(10), [label.replace("\n", " ") + ("*" if i == 9 else "") for i, (_, label) in enumerate(DATASETS)], fontsize=11)
        for y in range(10):
            for x in range(3):
                value = values[y, x]
                label = "N/A" if np.isnan(value) else f"{value:+.2f}" if metric == "log_score" else f"{value:+.3f}"
                color = "white" if np.isfinite(value) and abs(value) > limit*.58 else "#17232B"
                ax.text(x, y, label, ha="center", va="center", fontsize=11, color=color)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.05, pad=.05)
        bar.set_label("Change in mean log score" if metric == "log_score" else "Change in correlation", fontsize=10)
        bar.ax.tick_params(labelsize=9)
    fig.text(.025, .07, "*Novel-focus Spearman is the mean of 16 within-context correlations; Pearson and log score use all 480 pairs.", fontsize=10, color="#52616B")
    fig.text(.025, .038, "Corrected Qwen2-7B results. Descriptive differences, with no uncertainty intervals. N/A indicates an unavailable model.", fontsize=10, color="#52616B")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"linking_minus_no_linking.{extension}", dpi=230, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    result = args.results_dir
    out = result / "advisor_summary/baseline_comparison"
    out.mkdir(exist_ok=True, parents=True)
    fits, sources = read_fits(result)
    delta = differences(fits)
    delta.to_csv(out / "selected_model_differences.csv", index=False)
    counts = delta.groupby(["metric", "model", "comparison"]).size().unstack(fill_value=0)
    counts.to_csv(out / "comparison_counts.csv")
    draw(delta, out)

    # Recompute dataset scores/correlations from matched item predictions.
    items_path = result / "advisor_summary/item_level_scatterplots/item_level_predictions_and_log_scores.csv"
    items = pd.read_csv(items_path)
    sources.append(items_path)
    for row in fits.itertuples():
        if not row.available:
            continue
        a = items.loc[items.dataset.eq(row.dataset) & items.model.eq(row.model)]
        assert not a.analysis_unit_id.duplicated().any()
        if row.metric == "log_score":
            value = a.item_log_score.mean()
        elif row.metric == "pearson_r":
            value = a.human_rate.corr(a.prediction)
        elif row.dataset != DATASETS[-1][0]:
            value = a.human_rate.rank().corr(a.prediction.rank())
        else:
            value = np.mean([g.human_rate.rank().corr(g.prediction.rank()) for _, g in a.groupby(a.analysis_unit_id.str.split("::").str[0])])
        assert np.isclose(value, row.value, rtol=0, atol=1e-12), row

    # Audit the larger model family separately from the prespecified three.
    full = []
    focus_path = result / "spearman/mean_within_context_spearman.csv"
    focus = pd.read_csv(focus_path)
    specs = {"No linking structure": ("No linking structure", "direct"), "X but not Y": ("X but not Y", "direct"), "Ordering": ("ordering", "top_k")}
    for title, structure in [("Set", "set"), ("Conjunction", "conjunction"), ("Disjunction", "disjunction")]:
        for suffix, variant in [("Top-K", "top_k"), ("Top-p", "top_p")]:
            specs[f"{title} {suffix}"] = (structure, variant)
    for metric, (stem, _, _) in METRICS.items():
        path = result / "linking_structure_tables" / f"{stem}_by_dataset_and_linking_structure.csv"
        table = pd.read_csv(path).set_index("Dataset")
        if metric == "spearman_rho":
            for model, (structure, variant) in specs.items():
                match = focus.loc[focus.measure.eq("negation") & focus.structure.eq(structure) & focus.variant.eq(variant)]
                assert len(match) == 1
                table.loc[DATASETS[-1][0], model] = match.iloc[0].mean_within_context_spearman
        for dataset, row in table.iterrows():
            for model, value in row.items():
                full.append(dict(metric=metric, dataset=dataset, model=model, value=value))
    full_delta = differences(pd.DataFrame(full))
    full_delta.to_csv(out / "all_evaluated_structure_differences.csv", index=False)

    # Investigate probability level and extreme predictions without changing fits.
    selected = items.loc[items.model.isin(MODELS)].copy()
    diagnostics = selected.groupby(["dataset", "model"]).agg(
        n=("prediction", "size"), mean_human=("human_rate", "mean"),
        mean_prediction=("prediction", "mean"), zero_predictions=("prediction", lambda x: int(x.eq(0).sum())),
        one_predictions=("prediction", lambda x: int(x.eq(1).sum())), mean_log_score=("item_log_score", "mean"))
    diagnostics.to_csv(out / "probability_diagnostics.csv")
    base = selected.loc[selected.model.eq(MODELS[0]), ["dataset", "analysis_unit_id", "prediction", "item_log_score"]].rename(columns={"prediction": "baseline_prediction", "item_log_score": "baseline_item_log_score"})
    paired = selected.loc[~selected.model.eq(MODELS[0])].merge(base, on=["dataset", "analysis_unit_id"], validate="many_to_one")
    paired["log_score_difference"] = paired.item_log_score - paired.baseline_item_log_score
    paired.to_csv(out / "paired_item_log_score_differences.csv", index=False)
    assert np.allclose(paired.groupby(["dataset", "model"]).log_score_difference.mean().sort_index(),
                       delta.loc[delta.metric.eq("log_score") & delta.available].set_index(["dataset", "model"]).difference.sort_index())

    # Keep context-level results distinct from pooled focus results.
    rows = []
    novel = selected.loc[selected.analysis_dataset_id.eq("novel_focus")].copy()
    novel["context"] = novel.analysis_unit_id.str.split("::").str[0]
    for (context, model), a in novel.groupby(["context", "model"]):
        for metric, value in [("log_score", a.item_log_score.mean()), ("pearson_r", a.human_rate.corr(a.prediction)),
                              ("spearman_rho", a.human_rate.rank().corr(a.prediction.rank()))]:
            rows.append(dict(dataset=context, model=model, metric=metric, value=value))
    context_fits = pd.DataFrame(rows)
    context_delta = differences(context_fits)
    context_delta.to_csv(out / "focus_context_differences.csv", index=False)
    context_delta.groupby(["metric", "model", "comparison"]).size().unstack(fill_value=0).to_csv(out / "focus_context_counts.csv")
    sensitivity = []
    for model, a in novel.groupby("model"):
        for metric, pooled in [("pearson_r", a.human_rate.corr(a.prediction)), ("spearman_rho", a.human_rate.rank().corr(a.prediction.rank()))]:
            mean = context_fits.loc[context_fits.model.eq(model) & context_fits.metric.eq(metric), "value"].mean()
            sensitivity.append(dict(model=model, metric=metric, pooled_480_pairs=pooled, mean_16_contexts=mean))
    pd.DataFrame(sensitivity).to_csv(out / "focus_aggregation_sensitivity.csv", index=False)
    strong = selected.loc[selected.analysis_dataset_id.isin(["rnx_estrong", "rnx_eonlystrong"])].copy()
    strong["item"] = strong.analysis_unit_id.str.split("::").str[-1]
    matched = strong.pivot(index=["model", "item"], columns="analysis_dataset_id", values="prediction")
    assert (matched.rnx_estrong == matched.rnx_eonlystrong).all()
    metadata = dict(scope="Three selected structures; full evaluated family reported separately", tolerance=TOLERANCE,
                    dataset_values_reproduced_from_items=105, strong_condition_predictions_identical=True,
                    inference="Descriptive point estimates; no significance or noninferiority tests",
                    sources=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Verified 105 fit values against item predictions. Saved comparisons to {out}")


if __name__ == "__main__":
    main()
