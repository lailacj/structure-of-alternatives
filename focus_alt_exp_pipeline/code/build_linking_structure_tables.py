"""Build aligned cross-dataset correlation and log-score reporting tables.

The direct Qwen continuations (neutral/no-link and X-but-not-Y) are aggregated
to the same analysis units as the sampled-prefix out-of-fold predictions.  The
wide outputs retain a fixed dataset/structure order and distinguish unavailable
models from undefined statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_set_variant_grid import item_log_score
from evaluate_focus_spearman import HUMAN_FILE, evaluate as evaluate_spearman, write_results, rank_correlation


DATASETS = [
    ("hu_vt16", "van Tiel et al. (2016)"),
    ("hu_g18", "Gotzner et al. (2018)"),
    ("hu_pvt21", "Pankratz & van Tiel (2021)"),
    ("hu_rx22", "Ronai & Xiang (2022)"),
    ("rnx_esi", "R&X ESI"),
    ("rnx_eweak", "R&X Eweak"),
    ("rnx_estrong", "R&X Estrong"),
    ("rnx_eonly", "R&X Eonly"),
    ("rnx_eonlystrong", "R&X Eonlystrong"),
    ("novel_focus", "Novel Focus Alternative Study"),
]

COLUMNS = [
    "No linking structure",
    "X but not Y",
    "Set Top-K",
    "Set Top-p",
    "Ordering",
    "Conjunction Top-K",
    "Conjunction Top-p",
    "Disjunction Top-K",
    "Disjunction Top-p",
]

SAMPLED_COLUMNS = {
    "Set Top-K": ("top_k", "set"),
    "Set Top-p": ("top_p", "set"),
    "Ordering": ("top_k", "ordering"),
    "Conjunction Top-K": ("top_k", "conjunction"),
    "Conjunction Top-p": ("top_p", "conjunction"),
    "Disjunction Top-K": ("top_k", "disjunction"),
    "Disjunction Top-p": ("top_p", "disjunction"),
}


def _dataset_id(row: pd.Series) -> str:
    if row["dataset_family"] == "hu_2023_benchmark":
        return f"hu_{row['dataset']}"
    if row["dataset_family"] == "ronai_xiang_2024":
        return f"rnx_{str(row['condition']).lower()}"
    if row["dataset_family"] == "novel_focus":
        return "novel_focus"
    raise ValueError(f"Unsupported dataset family: {row['dataset_family']}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype("string").fillna("false").str.strip().str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError("hu_original_analysis_included contains non-boolean values")
    return normalized.eq("true")


def direct_analysis_units(source_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate direct continuation probabilities to evaluation-unit grain."""

    raw = source_rows.copy()
    raw["analysis_dataset_id"] = raw.apply(_dataset_id, axis=1)
    hu = raw["dataset_family"].eq("hu_2023_benchmark")
    keep = ~hu | _as_bool(raw["hu_original_analysis_included"])
    raw = raw.loc[keep].copy()
    raw["analysis_unit_id"] = raw["item_id"].astype(str)
    raw.loc[hu.loc[raw.index], "analysis_unit_id"] = (
        raw.loc[hu.loc[raw.index], "dataset"].astype(str)
        + "::"
        + raw.loc[hu.loc[raw.index], "scale_id"].astype(str)
    )
    raw["No linking structure"] = np.exp(
        pd.to_numeric(raw["query_logprob_sum"], errors="raise")
    )
    raw["X but not Y"] = np.exp(
        pd.to_numeric(raw["x_but_not_y_logprob_sum"], errors="coerce")
    )
    units = raw.groupby(
        ["analysis_dataset_id", "analysis_unit_id"], as_index=False
    ).agg(
        human_rate=("human_rate", "mean"),
        **{
            "No linking structure": ("No linking structure", "mean"),
            "X but not Y": ("X but not Y", "mean"),
        },
    )
    return units


def _direct_summaries(
    units: pd.DataFrame, expected_units: pd.DataFrame
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    joined = expected_units.merge(
        units,
        on=["analysis_dataset_id", "analysis_unit_id"],
        how="left",
        suffixes=("_oof", "_direct"),
        validate="one_to_one",
    )
    if not np.allclose(joined["human_rate_oof"], joined["human_rate_direct"], atol=1e-12):
        raise ValueError("Direct and sampled tables disagree on human rates")

    correlations: dict[tuple[str, str], float] = {}
    log_scores: dict[tuple[str, str], float] = {}
    coverage: dict[tuple[str, str], str] = {}
    for dataset_id, rows in joined.groupby("analysis_dataset_id", sort=False):
        for column in ("No linking structure", "X but not Y"):
            available = rows[column].notna()
            n = int(available.sum())
            if not n:
                coverage[(dataset_id, column)] = "not applicable"
                continue
            if n != len(rows):
                raise ValueError(
                    f"Partial direct-score coverage for {dataset_id}/{column}: {n}/{len(rows)}"
                )
            x = rows[column]
            y = rows["human_rate_oof"]
            correlations[(dataset_id, column)] = (
                float(x.corr(y)) if x.nunique() > 1 and y.nunique() > 1 else float("nan")
            )
            log_scores[(dataset_id, column)] = float(item_log_score(y, x).mean())
            coverage[(dataset_id, column)] = f"available (N={n})"
    return correlations, log_scores, coverage


def build_tables(
    source_rows: pd.DataFrame,
    oof: pd.DataFrame,
    sampled_correlations: pd.DataFrame,
    sampled_log_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = oof.loc[oof["variant"].eq("top_k"), [
        "analysis_dataset_id", "analysis_unit_id", "human_rate"
    ]].drop_duplicates()
    direct = direct_analysis_units(source_rows)
    direct_r, direct_ll, direct_coverage = _direct_summaries(direct, expected)

    labels = dict(DATASETS)
    corr = pd.DataFrame(index=[key for key, _ in DATASETS], columns=COLUMNS, dtype=float)
    scores = pd.DataFrame(index=corr.index, columns=COLUMNS, dtype=float)
    coverage = pd.DataFrame(index=corr.index, columns=COLUMNS, dtype=object)
    for key in corr.index:
        for column in ("No linking structure", "X but not Y"):
            corr.loc[key, column] = direct_r.get((key, column), np.nan)
            scores.loc[key, column] = direct_ll.get((key, column), np.nan)
            coverage.loc[key, column] = direct_coverage.get((key, column), "missing")

    for column, (variant, structure) in SAMPLED_COLUMNS.items():
        corr_column = f"{structure}_pearson_r"
        selected_corr = sampled_correlations.loc[
            sampled_correlations["variant"].eq(variant)
        ].set_index("analysis_dataset_id")
        selected_scores = sampled_log_scores.loc[
            sampled_log_scores["variant"].eq(variant)
            & sampled_log_scores["structure"].eq(structure)
        ].set_index("analysis_dataset_id")
        for key in corr.index:
            if key not in selected_corr.index or key not in selected_scores.index:
                coverage.loc[key, column] = "missing"
                continue
            corr.loc[key, column] = float(selected_corr.loc[key, corr_column])
            scores.loc[key, column] = float(selected_scores.loc[key, "mean_oof_log_score"])
            n = int(selected_scores.loc[key, "N"])
            coverage.loc[key, column] = f"available (N={n})"

    for table in (corr, scores, coverage):
        table.index = table.index.map(labels)
        table.index.name = "Dataset"
    return corr.reset_index(), scores.reset_index(), coverage.reset_index()


def build_dataset_spearman(source_rows: pd.DataFrame, oof: pd.DataFrame):
    """Rank scales within each Hu dataset and items within each R&X condition.

    Use the exact analysis units and probability aggregation used for Pearson.
    Novel focus remains exclusively in the separate within-context analysis.
    """
    ids = [key for key, _ in DATASETS if key != "novel_focus"]
    predictions = oof.loc[oof.analysis_dataset_id.isin(ids)].copy()
    direct = direct_analysis_units(source_rows)
    summaries, paired = [], []
    keys = ["analysis_dataset_id", "analysis_unit_id"]
    for dataset_id in ids:
        rows = predictions.loc[predictions.analysis_dataset_id.eq(dataset_id)]
        if set(rows.variant) != {"top_k", "top_p"}:
            raise ValueError(f"Missing held-out variants for {dataset_id}")
        k = rows.loc[rows.variant.eq("top_k")].copy()
        top_p = rows.loc[rows.variant.eq("top_p")]
        joined = k.merge(top_p, on=keys, how="outer", validate="one_to_one", suffixes=("", "_top_p"), indicator=True)
        if not joined._merge.eq("both").all() or not np.allclose(joined.human_rate, joined.human_rate_top_p):
            raise ValueError(f"Held-out units or rates disagree for {dataset_id}")
        if not np.allclose(joined.ordering_probability, joined.ordering_probability_top_p):
            raise ValueError("Ordering must be boundary-independent")
        joined = joined.drop(columns="_merge").merge(
            direct.loc[direct.analysis_dataset_id.eq(dataset_id)], on=keys,
            how="outer", validate="one_to_one", suffixes=("", "_direct"), indicator=True)
        if not joined._merge.eq("both").all() or not np.allclose(joined.human_rate, joined.human_rate_direct):
            raise ValueError(f"Direct and held-out units or rates disagree for {dataset_id}")
        group_type = "within_dataset" if dataset_id.startswith("hu_") else "within_condition"
        for model in COLUMNS:
            if model in SAMPLED_COLUMNS:
                variant, structure = SAMPLED_COLUMNS[model]
                column = structure + "_probability" + ("_top_p" if variant == "top_p" else "")
            else:
                variant, structure, column = "direct", model, model
            values = joined[column]
            unavailable = dataset_id.startswith("rnx_") and model == "X but not Y"
            if unavailable:
                if values.notna().any():
                    raise ValueError("Unexpected X-but-not-Y scores for R&X")
                rho, status = np.nan, "not_applicable"
            else:
                if not values.between(0, 1).all() or not joined.human_rate.between(0, 1).all():
                    raise ValueError(f"Missing or invalid probabilities for {dataset_id}/{model}")
                rho, status = rank_correlation(joined.human_rate, values)
                frame = joined[keys + ["human_rate", "cv_fold"]].copy()
                frame["model_probability"] = values
                frame["human_rank"] = joined.human_rate.rank(method="average", ascending=False) - 1
                frame["model_rank"] = values.rank(method="average", ascending=False) - 1
                frame["model"], frame["group_type"] = model, group_type
                paired.append(frame)
            summaries.append(dict(analysis_dataset_id=dataset_id, dataset=dict(DATASETS)[dataset_id],
                                  group_type=group_type, model=model, n=0 if unavailable else len(joined),
                                  spearman_rho=rho, status=status))
    summary = pd.DataFrame(summaries)
    wide = summary.pivot(index="dataset", columns="model", values="spearman_rho").reindex(
        index=[label for key, label in DATASETS if key in ids], columns=COLUMNS)
    wide.index.name = "Dataset"
    return wide.reset_index(), summary, pd.concat(paired, ignore_index=True)


def _markdown_table(table: pd.DataFrame, digits: int = 3) -> str:
    formatted = table.copy()
    for column in COLUMNS:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}"
            )
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(clean(column) for column in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = [
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv = args.results_dir / "cv_results"
    spearman = evaluate_spearman(pd.read_csv(HUMAN_FILE), pd.read_csv(args.source_rows),
                                 pd.read_csv(cv / "oof_predictions.csv"))
    write_results(spearman, args.results_dir / "spearman")
    corr, scores, coverage = build_tables(
        pd.read_csv(args.source_rows),
        pd.read_csv(cv / "oof_predictions.csv"),
        pd.read_csv(cv / "correlations.csv"),
        pd.read_csv(cv / "oof_log_scores_by_dataset_and_structure.csv"),
    )
    dataset_spearman, spearman_coverage, paired_ranks = build_dataset_spearman(
        pd.read_csv(args.source_rows), pd.read_csv(cv / "oof_predictions.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_spearman.to_csv(args.output_dir / "spearman_by_dataset_and_linking_structure.csv", index=False)
    spearman_coverage.to_csv(args.output_dir / "spearman_coverage.csv", index=False)
    paired_ranks.to_csv(args.output_dir / "spearman_paired_ranks.csv", index=False)
    corr.to_csv(args.output_dir / "correlations_by_dataset_and_linking_structure.csv", index=False)
    scores.to_csv(args.output_dir / "log_scores_by_dataset_and_linking_structure.csv", index=False)
    coverage.to_csv(args.output_dir / "coverage_by_dataset_and_linking_structure.csv", index=False)
    report = (
        "# Cross-dataset linking-structure results\n\n"
        "All columns use the same analysis units within each dataset. Higher is better "
        "for both Pearson correlation and mean proper log score.\n\n"
        "Focus-context word-ranking and negation Spearman results, paired ranks, and equal-context means are in `../spearman/SPEARMAN.md`.\n\n"
        "## Pearson correlations\n\n"
        + _markdown_table(corr)
        + "\n\n## Spearman correlations: Hu datasets and R&X conditions\n\n"
        + "Hu ranks scales within each dataset, after averaging van Tiel template probabilities. "
        + "R&X ranks the 60 items separately within each condition. These use the same units "
        + "as Pearson. Average ranks handle ties; constant predictions are undefined. "
        + "Focus uses the separate within-context analysis above.\n\n"
        + _markdown_table(dataset_spearman)
        + "\n\n## Mean proper log scores\n\n"
        + _markdown_table(scores)
        + "\n\n## Coverage\n\n"
        + _markdown_table(coverage)
        + "\n\nX-but-not-Y is structurally unavailable for the five R&X conditions. "
        "NA in any other correlation cell means that the statistic is undefined.\n"
    )
    (args.output_dir / "LINKING_STRUCTURE_TABLES.md").write_text(report, encoding="utf-8")
    print(f"[complete] wrote linking-structure tables to {args.output_dir}")


if __name__ == "__main__":
    main()
