"""Build development cross-dataset correlations and held-out log scores.

The analysis grain follows each source study's human observation grain.  In
particular, Hu et al.'s original string-based inclusion rule is applied and the
three van Tiel sentence templates are averaged to one scale before evaluation.

Set parameters are fitted in stratified grouped cross-validation using an
item-mean Bernoulli log score that is available for every dataset, including
Hu sources that publish rates without exact response counts.  One threshold
and one Gumbel scale are learned from the Set model in each training fold and
reused unchanged for Ordering, Conjunction, and Disjunction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from .absolute_threshold_models import structure_probabilities
    from .canonical_observations import validate_canonical_observations
except ImportError:
    from absolute_threshold_models import structure_probabilities
    from canonical_observations import validate_canonical_observations


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT / "focus_alt_exp_pipeline" / "canonical_data" / "all_observations.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "focus_alt_exp_pipeline" / "results" / "big_table_development"
)

STRUCTURES: Final[tuple[str, ...]] = (
    "set",
    "ordering",
    "disjunction",
    "conjunction",
)
PROBABILITY_EPSILON: Final[float] = 1e-10
DEFAULT_FOLD_COUNT: Final[int] = 10
SCALE_BOUNDS: Final[tuple[float, float]] = (0.1, 20.0)

HU_ORIGINAL_EXPECTEDNESS_R: Final[dict[str, float]] = {
    # Exact sign-reversals of Hu et al.'s Figure 3a surprisal correlations,
    # reconstructed from their public notebook at commit 50a7064.  The table
    # labels these as expectedness correlations, hence the positive sign.
    "hu_rx22": 0.3607049778615833,
    "hu_pvt21": 0.17985967831267793,
    "hu_g18": 0.053135193359448996,
    "hu_vt16": 0.04459122735688337,
}

ANALYSIS_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("hu_rx22", "Hu: Ronai & Xiang (2022)"),
    ("hu_pvt21", "Hu: Pankratz & van Tiel (2021)"),
    ("hu_g18", "Hu: Gotzner et al. (2018)"),
    ("hu_vt16", "Hu: van Tiel et al. (2016)"),
    ("rnx_esi", "Ronai & Xiang (2024): ESI"),
    ("rnx_eweak", "Ronai & Xiang (2024): Eweak"),
    ("rnx_estrong", "Ronai & Xiang (2024): Estrong"),
    ("rnx_eonly", "Ronai & Xiang (2024): Eonly"),
    ("rnx_eonlystrong", "Ronai & Xiang (2024): Eonlystrong"),
    ("novel_focus", "Novel focus alternatives"),
)
ROW_LABELS: Final[dict[str, str]] = dict(ANALYSIS_ROWS)
ROW_ORDER: Final[dict[str, int]] = {
    row_id: index for index, (row_id, _) in enumerate(ANALYSIS_ROWS)
}
EXPECTED_ANALYSIS_UNITS: Final[dict[str, int]] = {
    "hu_rx22": 57,
    "hu_pvt21": 50,
    "hu_g18": 67,
    "hu_vt16": 39,
    "rnx_esi": 60,
    "rnx_eweak": 60,
    "rnx_estrong": 60,
    "rnx_eonly": 60,
    "rnx_eonlystrong": 60,
    "novel_focus": 480,
}


@dataclass(frozen=True)
class ThresholdScaleFit:
    threshold: float
    scale: float
    objective: float
    success: bool
    threshold_at_boundary: bool
    scale_at_boundary: bool


def _single_value(values: pd.Series, *, label: str) -> object:
    unique = values.drop_duplicates().tolist()
    if len(unique) != 1:
        raise ValueError(f"{label} must have one value, found {unique}")
    return unique[0]


def _analysis_dataset_id(dataset_family: str, dataset: str, condition: str) -> str:
    if dataset_family == "hu_2023_benchmark":
        return f"hu_{dataset}"
    if dataset_family == "ronai_xiang_2024":
        return f"rnx_{condition.lower()}"
    if dataset_family == "novel_focus":
        return "novel_focus"
    raise ValueError(f"Unsupported dataset family: {dataset_family}")


def _base_unit_record(row: pd.Series) -> dict[str, object]:
    analysis_dataset_id = _analysis_dataset_id(
        str(row["dataset_family"]),
        str(row["dataset"]),
        str(row["condition"]),
    )
    return {
        "dataset_family": str(row["dataset_family"]),
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "analysis_dataset_id": analysis_dataset_id,
        "analysis_label": ROW_LABELS[analysis_dataset_id],
        "analysis_unit_id": str(row["item_id"]),
        "cv_group_id": (
            f"{row['dataset_family']}::{row['dataset']}::{row['group_id']}"
        ),
        "human_rate": float(row["human_rate"]),
        "human_yes": row["human_yes"],
        "human_total": row["human_total"],
        "human_count_status": str(row["human_count_status"]),
        "context_count": 1,
        "trigger_logprob_sum": float(row["trigger_logprob_sum"]),
        "query_logprob_sum": float(row["query_logprob_sum"]),
        "x_but_not_y_applicable": bool(row["x_but_not_y_applicable"]),
        "x_but_not_y_logprob_sum": row["x_but_not_y_logprob_sum"],
        "model_revision": str(row["model_revision"]),
    }


def _aggregate_hu_units(hu: pd.DataFrame) -> list[dict[str, object]]:
    required = {"hu_original_analysis_included", "scale_id"}
    missing = required.difference(hu.columns)
    if missing:
        raise ValueError(f"Hu canonical rows are missing columns: {sorted(missing)}")
    included = hu.loc[hu["hu_original_analysis_included"].astype(bool)].copy()
    records: list[dict[str, object]] = []
    for (dataset, scale_id), group in included.groupby(
        ["dataset", "scale_id"],
        sort=True,
        dropna=False,
    ):
        dataset_family = str(_single_value(group["dataset_family"], label="Hu family"))
        condition = str(_single_value(group["condition"], label="Hu condition"))
        human_rate = float(_single_value(group["human_rate"], label="Hu human rate"))
        count_status = str(
            _single_value(group["human_count_status"], label="Hu count status")
        )
        revision = str(_single_value(group["model_revision"], label="Hu model revision"))
        analysis_dataset_id = _analysis_dataset_id(dataset_family, str(dataset), condition)

        if count_status == "exact":
            if len(group) != 1:
                raise ValueError("Exact-count Hu units cannot repeat one human count across contexts")
            human_yes = int(group["human_yes"].iloc[0])
            human_total = int(group["human_total"].iloc[0])
        else:
            human_yes = np.nan
            human_total = np.nan

        records.append(
            {
                "dataset_family": dataset_family,
                "dataset": str(dataset),
                "condition": condition,
                "analysis_dataset_id": analysis_dataset_id,
                "analysis_label": ROW_LABELS[analysis_dataset_id],
                "analysis_unit_id": f"{dataset}::{scale_id}",
                "cv_group_id": f"{dataset_family}::{dataset}::{scale_id}",
                "human_rate": human_rate,
                "human_yes": human_yes,
                "human_total": human_total,
                "human_count_status": count_status,
                "context_count": int(len(group)),
                # Hu et al. average surprisal across the three vt16 templates.
                # Averaging log probability is the expectedness-sign equivalent.
                "trigger_logprob_sum": float(group["trigger_logprob_sum"].mean()),
                "query_logprob_sum": float(group["query_logprob_sum"].mean()),
                "x_but_not_y_applicable": True,
                "x_but_not_y_logprob_sum": float(
                    group["x_but_not_y_logprob_sum"].mean()
                ),
                "model_revision": revision,
            }
        )
    return records


def assign_stratified_group_folds(
    units: pd.DataFrame,
    *,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> pd.Series:
    """Assign deterministic folds while keeping every source group intact."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    assignments: dict[str, int] = {}
    group_strata = units[["dataset_family", "dataset", "cv_group_id"]].drop_duplicates()
    for _, stratum in group_strata.groupby(["dataset_family", "dataset"], sort=True):
        groups = sorted(stratum["cv_group_id"].astype(str).tolist())
        if len(groups) < fold_count:
            raise ValueError(
                "Every dataset must contain at least fold_count groups; "
                f"found {len(groups)}"
            )
        for index, group_id in enumerate(groups):
            fold = index % fold_count
            if group_id in assignments and assignments[group_id] != fold:
                raise ValueError("One CV group received conflicting fold assignments")
            assignments[group_id] = fold
    folds = units["cv_group_id"].map(assignments)
    if folds.isna().any():
        raise ValueError("Some analysis units did not receive a CV fold")
    return folds.astype(int)


def build_analysis_units(
    canonical: pd.DataFrame,
    *,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> pd.DataFrame:
    """Return one analysis row per independent human item/scale observation."""

    validate_canonical_observations(canonical, require_complete_provenance=True)
    revision_values = canonical["model_revision"].astype(str).unique().tolist()
    if len(revision_values) != 1:
        raise ValueError(f"Expected one model revision, found {revision_values}")

    hu = canonical.loc[canonical["dataset_family"].eq("hu_2023_benchmark")]
    other = canonical.loc[~canonical["dataset_family"].eq("hu_2023_benchmark")]
    records = _aggregate_hu_units(hu)
    records.extend(_base_unit_record(row) for _, row in other.iterrows())
    units = pd.DataFrame.from_records(records)
    units["row_order"] = units["analysis_dataset_id"].map(ROW_ORDER)
    if units["row_order"].isna().any():
        raise ValueError("Analysis units contain an unknown table row")
    units = units.sort_values(
        ["row_order", "analysis_unit_id"],
        ignore_index=True,
    ).drop(columns="row_order")
    units["cv_fold"] = assign_stratified_group_folds(units, fold_count=fold_count)

    observed_counts = units.groupby("analysis_dataset_id").size().to_dict()
    if observed_counts != EXPECTED_ANALYSIS_UNITS:
        raise ValueError(
            "Analysis-unit coverage changed: "
            f"expected={EXPECTED_ANALYSIS_UNITS}, observed={observed_counts}"
        )
    if units.duplicated(["analysis_dataset_id", "analysis_unit_id"]).any():
        raise ValueError("Analysis units contain duplicate dataset/unit keys")
    if not np.isfinite(
        units[["human_rate", "trigger_logprob_sum", "query_logprob_sum"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Analysis units contain non-finite required values")
    return units


def item_log_score(
    human_rate: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> np.ndarray:
    """Bernoulli log score per equally weighted item, allowing fractional rates."""

    y = np.asarray(human_rate, dtype=float)
    p = np.asarray(probability, dtype=float)
    y, p = np.broadcast_arrays(y, p)
    if not np.all(np.isfinite(y)) or np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("Human rates must be finite and between zero and one")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Model probabilities must be finite and between zero and one")
    clipped = np.clip(p, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)


def _balanced_item_log_score(
    units: pd.DataFrame,
    probability: np.ndarray,
) -> float:
    scored = units[["analysis_dataset_id", "human_rate"]].copy()
    scored["item_log_score"] = item_log_score(scored["human_rate"], probability)
    return float(scored.groupby("analysis_dataset_id")["item_log_score"].mean().mean())


def fit_threshold_scale(units: pd.DataFrame) -> ThresholdScaleFit:
    """Fit a shared Set threshold and Gumbel scale by balanced item log score."""

    query = units["query_logprob_sum"].to_numpy(dtype=float)
    trigger = units["trigger_logprob_sum"].to_numpy(dtype=float)
    threshold_bounds = (float(query.min() - 10.0), float(query.max() + 10.0))
    log_scale_bounds = tuple(float(np.log(value)) for value in SCALE_BOUNDS)

    def objective(parameters: np.ndarray) -> float:
        threshold = float(parameters[0])
        scale = float(np.exp(parameters[1]))
        probabilities = np.asarray(
            structure_probabilities(
                query,
                trigger,
                threshold=threshold,
                scale=scale,
            ).set,
            dtype=float,
        )
        return -_balanced_item_log_score(units, probabilities)

    starts = [
        (float(np.quantile(query, quantile)), float(np.log(scale)))
        for quantile in (0.2, 0.5, 0.8)
        for scale in (1.0, 3.0, 7.0)
    ]
    fits = [
        minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[threshold_bounds, log_scale_bounds],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        for start in starts
    ]
    successful_fits = [fit for fit in fits if fit.success and np.isfinite(fit.fun)]
    if not successful_fits:
        raise RuntimeError("Threshold/scale optimization produced no successful finite fit")
    best = min(successful_fits, key=lambda fit: float(fit.fun))
    threshold = float(best.x[0])
    scale = float(np.exp(best.x[1]))
    threshold_tolerance = max(1e-6, (threshold_bounds[1] - threshold_bounds[0]) * 1e-5)
    scale_tolerance = 1e-5
    return ThresholdScaleFit(
        threshold=threshold,
        scale=scale,
        objective=float(best.fun),
        success=bool(best.success),
        threshold_at_boundary=(
            abs(threshold - threshold_bounds[0]) <= threshold_tolerance
            or abs(threshold - threshold_bounds[1]) <= threshold_tolerance
        ),
        scale_at_boundary=(
            abs(np.log(scale) - log_scale_bounds[0]) <= scale_tolerance
            or abs(np.log(scale) - log_scale_bounds[1]) <= scale_tolerance
        ),
    )


def cross_validated_predictions(
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Set parameters on training folds and predict all structures held out."""

    predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    folds = sorted(units["cv_fold"].unique().tolist())
    for fold in folds:
        train = units.loc[units["cv_fold"].ne(fold)].copy()
        test = units.loc[units["cv_fold"].eq(fold)].copy()
        fit = fit_threshold_scale(train)
        probabilities = structure_probabilities(
            test["query_logprob_sum"].to_numpy(dtype=float),
            test["trigger_logprob_sum"].to_numpy(dtype=float),
            threshold=fit.threshold,
            scale=fit.scale,
        )
        predicted = test.copy()
        for structure in STRUCTURES:
            predicted[f"{structure}_probability"] = np.asarray(
                getattr(probabilities, structure),
                dtype=float,
            )

        train_baselines = train.groupby("analysis_dataset_id")["human_rate"].mean()
        predicted["intercept_probability"] = predicted["analysis_dataset_id"].map(
            train_baselines
        )
        if predicted["intercept_probability"].isna().any():
            raise ValueError("A held-out stratum has no training rows for its intercept")
        predictions.append(predicted)
        fold_records.append(
            {
                "cv_fold": int(fold),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_groups": int(train["cv_group_id"].nunique()),
                "test_groups": int(test["cv_group_id"].nunique()),
                "threshold": fit.threshold,
                "gumbel_scale": fit.scale,
                "balanced_training_objective": fit.objective,
                "optimizer_success": fit.success,
                "threshold_at_boundary": fit.threshold_at_boundary,
                "scale_at_boundary": fit.scale_at_boundary,
            }
        )

    out = pd.concat(predictions, ignore_index=True).sort_values(
        ["analysis_dataset_id", "analysis_unit_id"],
        ignore_index=True,
    )
    if len(out) != len(units) or out.duplicated(
        ["analysis_dataset_id", "analysis_unit_id"]
    ).any():
        raise RuntimeError("Cross-validation did not predict every analysis unit exactly once")
    return out, pd.DataFrame.from_records(fold_records)


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(pairs["x"], pairs["y"])[0, 1])


def summarize_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row_id, label in ANALYSIS_ROWS:
        rows = predictions.loc[predictions["analysis_dataset_id"].eq(row_id)]
        if len(rows) != EXPECTED_ANALYSIS_UNITS[row_id]:
            raise ValueError(f"Unexpected prediction count for {row_id}: {len(rows)}")
        x_applicable = bool(rows["x_but_not_y_applicable"].all())
        records.append(
            {
                "analysis_dataset_id": row_id,
                "dataset": label,
                "N": int(len(rows)),
                "human_count_status": str(
                    _single_value(rows["human_count_status"], label=f"{row_id} count status")
                ),
                "mean_contexts_per_unit": float(rows["context_count"].mean()),
                "hu_original_expectedness_r": HU_ORIGINAL_EXPECTEDNESS_R.get(
                    row_id,
                    np.nan,
                ),
                "qwen_x_but_not_y_r": (
                    pearson_correlation(
                        rows["x_but_not_y_logprob_sum"],
                        rows["human_rate"],
                    )
                    if x_applicable
                    else np.nan
                ),
                "qwen_no_frame_r": pearson_correlation(
                    rows["query_logprob_sum"],
                    rows["human_rate"],
                ),
                "qwen_set_r": pearson_correlation(
                    rows["set_probability"],
                    rows["human_rate"],
                ),
                "qwen_ordering_r": pearson_correlation(
                    rows["ordering_probability"],
                    rows["human_rate"],
                ),
                "qwen_disjunction_r": pearson_correlation(
                    rows["disjunction_probability"],
                    rows["human_rate"],
                ),
                "qwen_conjunction_r": pearson_correlation(
                    rows["conjunction_probability"],
                    rows["human_rate"],
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _response_log_likelihood(
    rows: pd.DataFrame,
    probability_column: str,
) -> tuple[float, int, float]:
    if not rows["human_count_status"].eq("exact").all():
        return float("nan"), 0, float("nan")
    probability = np.clip(
        rows[probability_column].to_numpy(dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    yes = rows["human_yes"].to_numpy(dtype=float)
    total = rows["human_total"].to_numpy(dtype=float)
    log_likelihood = float(
        np.sum(yes * np.log(probability) + (total - yes) * np.log1p(-probability))
    )
    response_count = int(total.sum())
    return log_likelihood, response_count, log_likelihood / response_count


def summarize_log_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    model_columns = {
        "intercept": "intercept_probability",
        **{structure: f"{structure}_probability" for structure in STRUCTURES},
    }
    for row_id, label in ANALYSIS_ROWS:
        rows = predictions.loc[predictions["analysis_dataset_id"].eq(row_id)].copy()
        baseline_item_score = float(
            item_log_score(rows["human_rate"], rows["intercept_probability"]).mean()
        )
        for model, probability_column in model_columns.items():
            mean_item_log_score = float(
                item_log_score(rows["human_rate"], rows[probability_column]).mean()
            )
            total_ll, response_count, response_mean = _response_log_likelihood(
                rows,
                probability_column,
            )
            records.append(
                {
                    "analysis_dataset_id": row_id,
                    "dataset": label,
                    "N": int(len(rows)),
                    "model": model,
                    "mean_item_log_score": mean_item_log_score,
                    "delta_item_log_score_vs_intercept": (
                        mean_item_log_score - baseline_item_score
                    ),
                    "response_total_log_likelihood": total_ll,
                    "response_count": response_count if response_count else np.nan,
                    "response_mean_log_score": response_mean,
                }
            )
    return pd.DataFrame.from_records(records)


def add_wide_log_scores(
    correlations: pd.DataFrame,
    log_scores: pd.DataFrame,
) -> pd.DataFrame:
    wide = correlations.copy()
    for structure in STRUCTURES:
        scores = log_scores.loc[log_scores["model"].eq(structure)].set_index(
            "analysis_dataset_id"
        )
        wide[f"{structure}_mean_item_log_score"] = wide["analysis_dataset_id"].map(
            scores["mean_item_log_score"]
        )
        wide[f"{structure}_response_mean_log_score"] = wide[
            "analysis_dataset_id"
        ].map(scores["response_mean_log_score"])
    return wide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical = pd.read_csv(args.input)
    units = build_analysis_units(canonical, fold_count=args.fold_count)
    predictions, folds = cross_validated_predictions(units)
    correlations = summarize_correlations(predictions)
    log_scores = summarize_log_scores(predictions)
    metrics = add_wide_log_scores(correlations, log_scores)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "analysis_units.csv": units,
        "oof_predictions.csv": predictions,
        "fold_parameters.csv": folds,
        "big_table_correlations.csv": correlations,
        "big_table_log_scores.csv": log_scores,
        "big_table_metrics.csv": metrics,
    }
    for filename, table in outputs.items():
        output = args.output_dir / filename
        table.to_csv(output, index=False)
        print(f"[complete] wrote {output} rows={len(table)}")

    print(
        "[fit] threshold mean/min/max="
        f"{folds['threshold'].mean():.6f}/"
        f"{folds['threshold'].min():.6f}/"
        f"{folds['threshold'].max():.6f}"
    )
    print(
        "[fit] scale mean/min/max="
        f"{folds['gumbel_scale'].mean():.6f}/"
        f"{folds['gumbel_scale'].min():.6f}/"
        f"{folds['gumbel_scale'].max():.6f}"
    )
    print(
        "[warning] These are development results. Freeze the CV and calibration rule "
        "with the authors before copying values into the paper."
    )


if __name__ == "__main__":
    main()
