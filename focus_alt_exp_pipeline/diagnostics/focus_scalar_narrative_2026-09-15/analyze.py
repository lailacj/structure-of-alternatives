"""Read-only investigation of the saved September 8 results, not a model rerun.

All generated outputs stay beside this script. Array-target scores are recovered
from the provenance-matched offline viewer; they are never renormalized over the
displayed subset. The historical scorer correction is NOT applied here.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

from html import escape
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PIPELINE = OUT.parents[1]
sys.path.insert(0, str(PIPELINE / "code"))
from build_linking_structure_tables import direct_analysis_units, DATASETS, SAMPLED_COLUMNS
from evaluate_set_variant_grid import item_log_score

RESULTS = PIPELINE / "results/set_variant_qwen"
SOURCE = PIPELINE / "scoring_manifests/set_variant_qwen/source_rows.csv"
OOF = RESULTS / "cv_results/oof_predictions.csv"
VIEWER = PIPELINE / "results_viewer/index.html"


def corr(a, b, method="pearson"):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    if len(a) < 2 or a.max() - a.min() < 1e-14 or b.max() - b.min() < 1e-14:
        return float("nan")
    if method == "spearman":
        a, b = a.rank(), b.rank()
    return a.corr(b)


def bootstrap_difference(rows, a, b, metric, rng, reps=4000):
    """Paired cluster bootstrap of fixed OOF predictions; no selection refit."""
    groups = [g.index.to_numpy() for _, g in rows.groupby("cv_group_id")]
    x, z, y = rows[a].to_numpy(), rows[b].to_numpy(), rows.human_rate.to_numpy()
    point = corr(x, y, metric) - corr(z, y, metric)
    draws = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        draws.append(corr(x[idx], y[idx], metric) - corr(z[idx], y[idx], metric))
    draws = np.array(draws)
    valid = draws[np.isfinite(draws)]
    return dict(difference=point, low=np.quantile(valid, .025), high=np.quantile(valid, .975),
                clusters=len(groups), bootstrap_reps=reps, valid_reps=len(valid))


def main():
    source, oof = pd.read_csv(SOURCE), pd.read_csv(OOF)
    payload = json.loads(re.search(r'<script id="results-data" type="application/json">(.*?)</script>',
                                  VIEWER.read_text(), re.S).group(1))
    inputs = [SOURCE, OOF, RESULTS / "prediction_grid.csv"]
    for path in inputs:
        saved = {r["sha256"] for r in payload["provenance"] if Path(r["path"]).name == path.name}
        assert saved == {hashlib.sha256(path.read_bytes()).hexdigest()}, path
    neutral = {p["id"]: p for p in payload["prompts"] if p["frame"] == "Neutral"}
    assert len(neutral) == source.prompt_id.nunique() == 360
    lookup = {pid: {w["word"].strip().lower(): w for w in p["distribution"]}
              for pid, p in neutral.items()}
    target_rows = []
    for _, row in source.iterrows():
        assert neutral[row.prompt_id]["text"] == row.generation_prompt
        for role in ("query", "trigger"):
            w = lookup[row.prompt_id][row[role].strip().lower()]
            assert np.isfinite(w["logp"]) and 0 < w["normalized"] <= 1
            target_rows.append(dict(prompt_id=row.prompt_id, word=row[role],
                                    source_logp=row[f"{role}_logprob_sum"],
                                    array_logp=w["logp"], array_normalized=w["normalized"],
                                    array_rank=w["rank"]))
    targets = pd.DataFrame(target_rows).drop_duplicates()
    assert not targets.duplicated(["prompt_id", "word"]).any()
    targets["logp_difference"] = targets.array_logp - targets.source_logp
    # Normalization must differ from raw log probability by ONE constant per
    # prompt. This check would catch accidental top-50/target-only rescaling.
    normalizers = targets.array_logp - np.log(targets.array_normalized)
    spread = normalizers.groupby(targets.prompt_id).agg(lambda x: x.max() - x.min())
    assert spread.max() < 1e-10
    targets.to_csv(OUT / "scoring_artifact_audit.csv", index=False)
    for role in ("query", "trigger"):
        cols = ["prompt_id", "word", "array_logp", "array_normalized", "array_rank"]
        right = targets[cols].rename(columns={"word": role, **{c: f"{role}_{c}" for c in cols[2:]}})
        source = source.merge(right, on=["prompt_id", role], validate="many_to_one")
    source["Array raw probability"] = np.exp(source.query_array_logp)
    source["Array normalized probability"] = source.query_array_normalized
    source["Array log probability"] = source.query_array_logp
    source["Direct log probability"] = source.query_logprob_sum
    source["Array exact ordering"] = (source.query_array_normalized /
                                       (source.query_array_normalized + source.trigger_array_normalized))
    source["analysis_dataset_id"] = np.where(source.dataset_family.eq("novel_focus"), "novel_focus",
        np.where(source.dataset_family.eq("hu_2023_benchmark"), "hu_" + source.dataset,
                 "rnx_" + source.condition.str.lower()))
    hu = source.dataset_family.eq("hu_2023_benchmark")
    source["analysis_unit_id"] = source.item_id
    source.loc[hu, "analysis_unit_id"] = source.loc[hu, "dataset"] + "::" + source.loc[hu, "scale_id"]
    source = source.loc[~hu | source.hu_original_analysis_included.astype(str).str.lower().eq("true")].copy()
    extra_cols = ["Array raw probability", "Array normalized probability", "Array log probability",
                  "Direct log probability", "Array exact ordering"]
    keys = ["analysis_dataset_id", "analysis_unit_id"]
    extra = source.groupby(keys, as_index=False)[extra_cols].mean()
    # For log-scale diagnostics, average templates' probabilities THEN log, just
    # as the published direct-probability report aggregates template scores.
    extra["Array log probability"] = np.log(extra["Array raw probability"])
    direct = direct_analysis_units(pd.read_csv(SOURCE))
    extra = extra.drop(columns="Direct log probability")
    units = direct.merge(extra, on=keys, validate="one_to_one")
    units["Direct log probability"] = np.log(units["No linking structure"])
    k = oof.loc[oof.variant.eq("top_k")]
    units = units.merge(k[keys + ["cv_group_id", "cv_fold"]], on=keys, validate="one_to_one")
    for label, (variant, structure) in SAMPLED_COLUMNS.items():
        rows = oof.loc[oof.variant.eq(variant), keys + [f"{structure}_probability"]]
        units = units.merge(rows.rename(columns={f"{structure}_probability": label}), on=keys, validate="one_to_one")
    assert len(units) == 993
    for variant in ("Top-K", "Top-p"):
        s, a, o = units[f"Set {variant}"], units[f"Conjunction {variant}"], units["Ordering"]
        assert (a <= np.minimum(s, o) + 1e-12).all()
        assert np.allclose(units[f"Disjunction {variant}"], s + o - a, atol=1e-12)
    # Attach metadata only where a one-to-one match exists (Hu has templates).
    meta = source.groupby(keys, as_index=False).agg(context=("context_id", "first"),
        trigger=("trigger", "first"), query=("query", "first"), prompt_id=("prompt_id", "first"),
        group_id=("group_id", "first"))
    units = units.merge(meta, on=keys, validate="one_to_one")
    model_cols = ["No linking structure", "X but not Y", *SAMPLED_COLUMNS, *extra_cols]
    assert len(model_cols) == len(set(model_cols))
    metrics = []
    for ds, rows in units.groupby("analysis_dataset_id"):
        for model in model_cols:
            if rows[model].isna().all():
                continue
            assert rows[model].notna().all()
            metrics.append(dict(dataset=ds, model=model, n=len(rows),
                pearson=corr(rows[model], rows.human_rate), spearman=corr(rows[model], rows.human_rate, "spearman"),
                log_score=float(item_log_score(rows.human_rate, rows[model]).mean()) if "log probability" not in model else np.nan))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / "dataset_metrics.csv", index=False)
    units.to_csv(OUT / "matched_item_predictions.csv", index=False)
    # Verify all saved Pearson cells before using the derived item table.
    saved = pd.read_csv(RESULTS / "linking_structure_tables/correlations_by_dataset_and_linking_structure.csv").set_index("Dataset")
    ds_labels = dict(DATASETS)
    for _, r in metrics.iterrows():
        if r.model in saved.columns:
            assert np.isclose(r.pearson, saved.loc[ds_labels[r.dataset], r.model], atol=1e-12)

    focus = units.loc[units.analysis_dataset_id.eq("novel_focus")].copy()
    within = []
    for context, rows in focus.groupby("context"):
        for model in model_cols:
            within.append(dict(context=context, model=model, spearman=corr(rows[model], rows.human_rate, "spearman"),
                               pearson=corr(rows[model], rows.human_rate)))
    within = pd.DataFrame(within)
    within.to_csv(OUT / "focus_within_story.csv", index=False)
    within_summary = within.groupby("model").agg(mean_rho=("spearman", "mean"), valid=("spearman", "count"),
                                                mean_r=("pearson", "mean"))
    within_summary.to_csv(OUT / "focus_within_story_summary.csv")

    # Separate unique candidate plausibility ranks from exclusion judgments.
    words = pd.read_csv(RESULTS / "spearman/word_paired_ranks.csv")
    source_focus = source.loc[source.dataset_family.eq("novel_focus")]
    qrates = source_focus.groupby(["context_id", "query"], as_index=False).agg(
        human_rate=("human_rate", "mean"), human_yes=("human_yes", "sum"), human_total=("human_total", "sum"),
        array_logp=("query_array_logp", "first"), array_normalized=("query_array_normalized", "first"),
        source_logp=("query_logprob_sum", "first"))
    qrates = qrates.rename(columns={"context_id": "context", "query": "word"})
    qrates["human_rate_trial_weighted"] = qrates.human_yes / qrates.human_total
    words = words.merge(qrates, on=["context", "word"], validate="one_to_one")
    assert len(words) == 96
    word_stats = []
    for context, g in words.groupby("context"):
        word_stats.append(dict(context=context,
            source_vs_human_plausibility=corr(-g.human_rank, g.source_logp, "spearman"),
            array_vs_human_plausibility=corr(-g.human_rank, g.array_logp, "spearman"),
            source_vs_query_exclusion=corr(g.human_rate, g.source_logp, "spearman"),
            array_vs_query_exclusion=corr(g.human_rate, g.array_logp, "spearman"),
            array_vs_query_exclusion_trial_weighted=corr(g.human_rate_trial_weighted, g.array_logp, "spearman"),
            source_vs_array=corr(g.source_logp, g.array_logp, "spearman")))
    words.to_csv(OUT / "focus_candidate_examples.csv", index=False)
    word_stats = pd.DataFrame(word_stats)
    word_stats.to_csv(OUT / "focus_word_ranking.csv", index=False)

    # Does the trigger-sensitive part of the models track differences across
    # triggers after removing each story/query's average? Descriptive only.
    residuals = []
    y_res = focus.human_rate - focus.groupby(["context", "query"]).human_rate.transform("mean")
    for model in model_cols:
        pred_res = focus[model] - focus.groupby(["context", "query"])[model].transform("mean")
        residuals.append(dict(model=model, residual_pearson=corr(pred_res, y_res),
                              prediction_residual_sd=pred_res.std(), human_residual_sd=y_res.std()))
    pd.DataFrame(residuals).to_csv(OUT / "focus_trigger_residuals.csv", index=False)

    # Fixed-prediction paired cluster intervals; exploratory, unadjusted.
    rng = np.random.default_rng(20260915)
    contrasts = []
    for ds, a, b in [("novel_focus", "Set Top-K", "No linking structure"),
                     ("novel_focus", "Set Top-K", "Array normalized probability"),
                     ("novel_focus", "Set Top-K", "Array log probability"),
                     ("hu_rx22", "X but not Y", "Set Top-K"),
                     ("hu_pvt21", "X but not Y", "Set Top-K"),
                     ("hu_g18", "X but not Y", "Set Top-p")]:
        g = units.loc[units.analysis_dataset_id.eq(ds)].reset_index(drop=True)
        contrasts.append(dict(dataset=ds, a=a, b=b, metric="pearson",
                              **bootstrap_difference(g, a, b, "pearson", rng)))
    pd.DataFrame(contrasts).to_csv(OUT / "exploratory_paired_bootstrap.csv", index=False)

    # R&X conditions share scales; record the information lost by stripping only.
    condition_pairs = []
    for left, right in [("rnx_esi", "rnx_eonly"), ("rnx_estrong", "rnx_eonlystrong")]:
        pair = units.loc[units.analysis_dataset_id.eq(left)].merge(
            units.loc[units.analysis_dataset_id.eq(right)], on="group_id", suffixes=("_a", "_b"), validate="one_to_one")
        assert len(pair) == 60
        condition_pairs.append(dict(condition_a=left, condition_b=right, n=60,
            identical_prompts=int(pair.prompt_id_a.eq(pair.prompt_id_b).sum()),
            human_mean_a=pair.human_rate_a.mean(), human_mean_b=pair.human_rate_b.mean(),
            mean_human_change=(pair.human_rate_b-pair.human_rate_a).mean(),
            max_set_prediction_difference=np.max(np.abs(pair["Set Top-K_b"] - pair["Set Top-K_a"]))))
    pd.DataFrame(condition_pairs).to_csv(OUT / "only_condition_check.csv", index=False)

    make_figures(metrics, within_summary, words, focus)
    provenance = {"seed": 20260915, "status": "exploratory; historical scores; no refit or rescoring",
        "array_targets": len(targets), "source_array_mismatches_gt_1e_minus_4": int((targets.logp_difference.abs() > 1e-4).sum()),
        "analysis_units": len(units), "full_support_normalization": "preserved from array export, not computed over displayed subset",
        "exact_ordering_max_absolute_error_vs_500_draw_saved": float(np.max(np.abs(units["Array exact ordering"] - units["Ordering"]))),
        "inputs": {str(p.relative_to(PIPELINE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + [VIEWER]}}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("ALL SAVED PEARSON CELLS REPRODUCED; input hashes match viewer.")
    print("\nDATASET METRICS\n", metrics.pivot(index="dataset", columns="model", values="pearson").round(3).to_string())
    print("\nFOCUS WITHIN-STORY\n", within_summary.round(3).to_string())
    print("\nWORD RANKING MEANS\n", word_stats.select_dtypes("number").mean().to_string())
    print("\nBOOTSTRAP\n", pd.DataFrame(contrasts).round(3).to_string(index=False))
    print("\nTRIGGER RESIDUALS\n", pd.DataFrame(residuals).round(3).to_string(index=False))
    print("\nONLY CONDITIONS\n", pd.DataFrame(condition_pairs).round(3).to_string(index=False))


def make_figures(metrics, within, words, focus):
    """Static SVG, using only verified numbers (no optional plotting runtime)."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="740" viewBox="0 0 1440 740">',
           '<rect width="1440" height="740" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#202b34;font-size:17px}.small{font-size:14px;fill:#52616d}</style>']
    def text(x, y, value, extra=""):
        svg.append(f'<text x="{x}" y="{y}" {extra}>{escape(value)}</text>')
    text(35, 42, "Focus versus scalar alternatives: a scoring and linking-rule diagnostic", 'style="font-size:25px"')
    text(35, 72, "Exploratory analysis of historical scores — not final paper results", 'class="small"')
    ds = ["novel_focus", "hu_vt16", "hu_g18", "hu_pvt21", "hu_rx22"]
    labels = ["Focus alternatives", "van Tiel (2016)", "Gotzner (2018)", "Pankratz & van Tiel (2021)", "Ronai & Xiang (2022)"]
    models = ["No linking structure", "X but not Y", "Set Top-K", "Set Top-p"]
    colors = ["#87929c", "#d17033", "#246a9a", "#48a58e"]
    tab = metrics.pivot(index="dataset", columns="model", values="pearson")
    text(35, 121, "A. Saved focus/scalar correlations", 'style="font-weight:bold"')
    x0, scale = 335, 475
    for tick in [-.2, 0, .2, .4, .6]:
        x = x0 + tick * scale
        svg.append(f'<line x1="{x}" x2="{x}" y1="155" y2="535" stroke="#e2e6e9"/>')
        text(x, 563, f"{tick:.1f}", 'text-anchor="middle" class="small"')
    for j, (dataset, label) in enumerate(zip(ds, labels)):
        y = 180 + 78*j
        text(230, y + 5, label, 'text-anchor="end" style="font-size:14px"')
        for offset, model, color in zip([-18, -6, 6, 18], models, colors):
            x = x0 + tab.loc[dataset, model] * scale
            svg.append(f'<circle cx="{x}" cy="{y+offset}" r="6" fill="{color}"/>')
    text(340, 596, "Pearson r with human exclusion", 'style="font-size:15px"')
    for j, (model, color) in enumerate(zip(models, colors)):
        x, y = 60 + (j%2)*280, 632 + (j//2)*27
        svg.append(f'<circle cx="{x}" cy="{y-5}" r="5" fill="{color}"/>')
        text(x+14, y, model, 'class="small"')
    text(740, 121, "B. Focus: matched sampling-input checks", 'style="font-weight:bold"')
    m = ["No linking structure", "Array raw probability", "Array normalized probability", "Array log probability", "Set Top-K"]
    ticks = ["Reported direct baseline", "Array raw P(q)", "Array normalized π(q)",
             "Array log P(q): diagnostic", "Set Top-K (K = 100)"]
    vals = tab.loc["novel_focus", m].to_numpy()
    for j, value in enumerate(vals):
        y = 180+78*j
        text(990, y+5, ticks[j], 'text-anchor="end" style="font-size:14px"')
        color = "#246a9a" if j == 4 else "#becbd3"
        svg.append(f'<rect x="1010" y="{y-17}" width="{value*475}" height="34" fill="{color}"/>')
        text(1018+value*475, y+5, f"{value:.3f}", 'style="font-size:15px"')
    for tick in [0, .2, .4, .6]:
        text(1010+tick*475, 563, f"{tick:.1f}", 'text-anchor="middle" class="small"')
    text(1010, 596, "Pearson r (480 focus pairs)", 'style="font-size:15px"')
    text(740, 634, "Raw, normalized, and log scores in B use the same array", 'class="small"')
    text(740, 657, "as Set. Log P(q) is not an exclusion probability.", 'class="small"')
    text(35, 710, "A uses matched items but different direct/sampling score artifacts. Neither panel resolves the pending word-boundary correction.", 'class="small"')
    svg.append('</svg>')
    (OUT / "focus_scalar_diagnostic.svg").write_text("\n".join(svg))


if __name__ == "__main__":
    main()
