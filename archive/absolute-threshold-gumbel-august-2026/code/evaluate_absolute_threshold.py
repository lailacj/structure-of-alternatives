"""Grouped evaluation for absolute-threshold alternative structures.

The evaluator consumes canonical observations, fits one absolute Set threshold
on training groups, and applies that same threshold to Set, Conjunction, and
Disjunction in held-out groups. Ordering is parameter-free once the shared
Gumbel scale is fixed.

The current CLI uses leave-one-group-out evaluation. For the novel focus data,
``group_id`` is the story/context, so every prediction is made with its context
held out from threshold fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

try:
    from .absolute_threshold_models import StructureProbabilities, structure_probabilities
    from .canonical_observations import validate_canonical_observations
except ImportError:
    from absolute_threshold_models import StructureProbabilities, structure_probabilities
    from canonical_observations import validate_canonical_observations


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT_DIR
    / "focus_alt_exp_pipeline"
    / "canonical_data"
    / "novel_focus_observations.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "focus_alt_exp_pipeline" / "results" / "absolute_threshold_development"
)

STRUCTURES: Final[tuple[str, ...]] = (
    "set",
    "ordering",
    "conjunction",
    "disjunction",
)
THRESHOLD_FIT_TARGETS: Final[tuple[str, ...]] = (
    "set",
    "conjunction",
    "disjunction",
)
PROBABILITY_EPSILON: Final[float] = 1e-10


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    objective: float
    lower_bound: float
    upper_bound: float
    success: bool
    at_boundary: bool


def _bounded_grid_golden_minimize(
    objective,
    *,
    lower_bound: float,
    upper_bound: float,
    grid_size: int = 257,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> tuple[float, float]:
    """Minimize a scalar objective without requiring SciPy.

    A coarse grid first locates the best basin, then a golden-section search
    refines the adjacent interval. Boundary values remain explicit candidates,
    which also makes threshold-boundary diagnostics reliable.
    """

    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    grid = np.linspace(lower_bound, upper_bound, grid_size)
    values = np.asarray([float(objective(value)) for value in grid], dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Threshold objective produced non-finite grid values")

    best_index = int(np.argmin(values))
    candidates: list[tuple[float, float]] = [
        (float(grid[best_index]), float(values[best_index])),
        (float(grid[0]), float(values[0])),
        (float(grid[-1]), float(values[-1])),
    ]
    if best_index in {0, grid_size - 1}:
        return min(candidates, key=lambda item: item[1])

    left = float(grid[best_index - 1])
    right = float(grid[best_index + 1])
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - inverse_phi * (right - left)
    d = left + inverse_phi * (right - left)
    f_c = float(objective(c))
    f_d = float(objective(d))

    for _ in range(max_iterations):
        if abs(right - left) <= tolerance * (1.0 + abs((left + right) / 2.0)):
            break
        if f_c <= f_d:
            right = d
            d = c
            f_d = f_c
            c = right - inverse_phi * (right - left)
            f_c = float(objective(c))
        else:
            left = c
            c = d
            f_c = f_d
            d = left + inverse_phi * (right - left)
            f_d = float(objective(d))

    refined_threshold = float((left + right) / 2.0)
    refined_objective = float(objective(refined_threshold))
    candidates.append((refined_threshold, refined_objective))
    return min(candidates, key=lambda item: item[1])


def binomial_log_likelihood(
    probability: np.ndarray | pd.Series | float,
    human_yes: np.ndarray | pd.Series,
    human_total: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return each row's aggregated Bernoulli/binomial log likelihood."""

    probability_array = np.asarray(probability, dtype=float)
    yes = np.asarray(human_yes, dtype=float)
    total = np.asarray(human_total, dtype=float)
    probability_array, yes, total = np.broadcast_arrays(probability_array, yes, total)
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("Model probabilities must be finite")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Model probabilities must be between zero and one")
    if np.any((yes < 0.0) | (total <= 0.0) | (yes > total)):
        raise ValueError("Human counts are invalid")

    clipped = np.clip(probability_array, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return yes * np.log(clipped) + (total - yes) * np.log1p(-clipped)


def _probability_for_structure(
    probabilities: StructureProbabilities,
    structure: str,
) -> np.ndarray:
    if structure not in STRUCTURES:
        raise ValueError(f"Unknown structure '{structure}'")
    return np.asarray(getattr(probabilities, structure), dtype=float)


def _predict_structures(
    observations: pd.DataFrame,
    *,
    threshold: float,
    scale: float,
) -> StructureProbabilities:
    return structure_probabilities(
        observations["query_logprob_sum"].to_numpy(dtype=float),
        observations["trigger_logprob_sum"].to_numpy(dtype=float),
        threshold=threshold,
        scale=scale,
    )


def _dataset_balanced_mean_log_score(
    observations: pd.DataFrame,
    probability: np.ndarray,
) -> float:
    scored = observations[["dataset_family", "human_yes", "human_total"]].copy()
    scored["log_likelihood"] = binomial_log_likelihood(
        probability,
        scored["human_yes"],
        scored["human_total"],
    )
    by_family = scored.groupby("dataset_family", as_index=False).agg(
        total_log_likelihood=("log_likelihood", "sum"),
        total_responses=("human_total", "sum"),
    )
    return float((by_family["total_log_likelihood"] / by_family["total_responses"]).mean())


def _default_threshold_bounds(observations: pd.DataFrame, *, scale: float) -> tuple[float, float]:
    all_scores = np.concatenate(
        [
            observations["query_logprob_sum"].to_numpy(dtype=float),
            observations["trigger_logprob_sum"].to_numpy(dtype=float),
        ]
    )
    margin = 10.0 * float(scale)
    return float(np.min(all_scores) - margin), float(np.max(all_scores) + margin)


def fit_shared_threshold(
    observations: pd.DataFrame,
    *,
    fit_target: str = "set",
    scale: float = 1.0,
    bounds: tuple[float, float] | None = None,
) -> ThresholdFit:
    """Fit one threshold by dataset-balanced training mean log score.

    The default fits the Set model and reuses the resulting threshold for every
    threshold-dependent structure. This prevents Conjunction and Disjunction
    from silently defining different alternative sets.
    """

    validate_canonical_observations(observations, require_human_counts=True)
    if fit_target not in THRESHOLD_FIT_TARGETS:
        raise ValueError(
            f"fit_target must be one of {list(THRESHOLD_FIT_TARGETS)}, got {fit_target!r}"
        )
    resolved_scale = float(scale)
    if not np.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("scale must be a finite number greater than zero")

    lower_bound, upper_bound = (
        _default_threshold_bounds(observations, scale=resolved_scale)
        if bounds is None
        else (float(bounds[0]), float(bounds[1]))
    )
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Threshold bounds must be finite")
    if lower_bound >= upper_bound:
        raise ValueError("Threshold lower bound must be less than upper bound")

    def objective(threshold: float) -> float:
        probabilities = _predict_structures(
            observations,
            threshold=float(threshold),
            scale=resolved_scale,
        )
        target_probability = _probability_for_structure(probabilities, fit_target)
        return -_dataset_balanced_mean_log_score(observations, target_probability)

    fitted_threshold, fitted_objective = _bounded_grid_golden_minimize(
        objective,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if not np.isfinite(fitted_threshold) or not np.isfinite(fitted_objective):
        raise RuntimeError("Threshold optimization failed")

    tolerance = max(1e-6, (upper_bound - lower_bound) * 1e-5)
    at_boundary = bool(
        abs(fitted_threshold - lower_bound) <= tolerance
        or abs(fitted_threshold - upper_bound) <= tolerance
    )
    return ThresholdFit(
        threshold=fitted_threshold,
        objective=fitted_objective,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        success=True,
        at_boundary=at_boundary,
    )


def _group_key(observations: pd.DataFrame) -> pd.Series:
    return (
        observations["dataset_family"].astype(str)
        + "::"
        + observations["dataset"].astype(str)
        + "::"
        + observations["group_id"].astype(str)
    )


def leave_one_group_out_predictions(
    observations: pd.DataFrame,
    *,
    fit_target: str = "set",
    scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one out-of-fold prediction per canonical observation."""

    validate_canonical_observations(observations, require_human_counts=True)
    prepared = observations.copy().reset_index(drop=True)
    prepared["_original_row"] = np.arange(len(prepared), dtype=int)
    prepared["_group_key"] = _group_key(prepared)
    groups = sorted(prepared["_group_key"].unique().tolist())
    if len(groups) < 2:
        raise ValueError("Leave-one-group-out evaluation requires at least two groups")

    prediction_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    dataset_families = ",".join(sorted(prepared["dataset_family"].unique()))

    for fold_index, heldout_group in enumerate(groups):
        heldout_mask = prepared["_group_key"].eq(heldout_group)
        train = prepared.loc[~heldout_mask].copy()
        test = prepared.loc[heldout_mask].copy()
        threshold_fit = fit_shared_threshold(
            train,
            fit_target=fit_target,
            scale=scale,
        )
        probabilities = _predict_structures(
            test,
            threshold=threshold_fit.threshold,
            scale=scale,
        )
        baseline_probability = float(train["human_yes"].sum() / train["human_total"].sum())

        fold_id = f"logo_{fold_index:03d}"
        predicted = test.drop(columns=["_group_key"]).copy()
        predicted["cv_scheme"] = "leave_one_group_out"
        predicted["fold_id"] = fold_id
        predicted["heldout_group_key"] = heldout_group
        predicted["threshold_fit_target"] = fit_target
        predicted["threshold_fit_dataset_families"] = dataset_families
        predicted["threshold"] = threshold_fit.threshold
        predicted["gumbel_scale"] = float(scale)
        predicted["baseline_probability"] = baseline_probability
        predicted["baseline_log_likelihood"] = binomial_log_likelihood(
            baseline_probability,
            predicted["human_yes"],
            predicted["human_total"],
        )
        for structure in STRUCTURES:
            probability = _probability_for_structure(probabilities, structure)
            predicted[f"{structure}_probability"] = probability
            predicted[f"{structure}_log_likelihood"] = binomial_log_likelihood(
                probability,
                predicted["human_yes"],
                predicted["human_total"],
            )
        prediction_frames.append(predicted)

        fold_records.append(
            {
                "cv_scheme": "leave_one_group_out",
                "fold_id": fold_id,
                "heldout_group_key": heldout_group,
                "threshold_fit_target": fit_target,
                "threshold_fit_dataset_families": dataset_families,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_responses": int(train["human_total"].sum()),
                "test_responses": int(test["human_total"].sum()),
                "threshold": threshold_fit.threshold,
                "gumbel_scale": float(scale),
                "fit_objective": threshold_fit.objective,
                "threshold_lower_bound": threshold_fit.lower_bound,
                "threshold_upper_bound": threshold_fit.upper_bound,
                "threshold_at_boundary": threshold_fit.at_boundary,
                "baseline_probability": baseline_probability,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values("_original_row", ignore_index=True).drop(
        columns="_original_row"
    )
    if len(predictions) != len(prepared):
        raise RuntimeError("Out-of-fold prediction row count does not match canonical input")
    if predictions.duplicated(["dataset_family", "dataset", "condition", "item_id"]).any():
        raise RuntimeError("Out-of-fold predictions contain duplicate item keys")

    folds = pd.DataFrame.from_records(fold_records)
    return predictions, folds


def _safe_correlations(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    if len(x_values) < 2 or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x_values, y_values)[0, 1])
    x_ranks = pd.Series(x_values).rank(method="average").to_numpy(dtype=float)
    y_ranks = pd.Series(y_values).rank(method="average").to_numpy(dtype=float)
    spearman = float(np.corrcoef(x_ranks, y_ranks)[0, 1])
    return pearson, spearman


def summarize_out_of_fold_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize item-level correlations and response-level log scores."""

    required = {
        "dataset_family",
        "dataset",
        "condition",
        "human_rate",
        "human_total",
        "baseline_log_likelihood",
        *[f"{structure}_probability" for structure in STRUCTURES],
        *[f"{structure}_log_likelihood" for structure in STRUCTURES],
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing summary columns: {sorted(missing)}")

    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", predictions)]
    for keys, subset in predictions.groupby(
        ["dataset_family", "dataset", "condition"],
        sort=True,
        dropna=False,
    ):
        label = "::".join(str(key) for key in keys)
        scopes.append(("condition", label, subset))

    records: list[dict[str, object]] = []
    for scope_type, scope_label, subset in scopes:
        total_responses = int(subset["human_total"].sum())
        baseline_total = float(subset["baseline_log_likelihood"].sum())
        for structure in STRUCTURES:
            probability_column = f"{structure}_probability"
            log_likelihood_column = f"{structure}_log_likelihood"
            pearson_r, spearman_rho = _safe_correlations(
                subset[probability_column],
                subset["human_rate"],
            )
            total_log_likelihood = float(subset[log_likelihood_column].sum())
            records.append(
                {
                    "scope_type": scope_type,
                    "scope": scope_label,
                    "structure": structure,
                    "item_count": len(subset),
                    "response_count": total_responses,
                    "pearson_r": pearson_r,
                    "spearman_rho": spearman_rho,
                    "total_log_likelihood": total_log_likelihood,
                    "mean_log_score": total_log_likelihood / total_responses,
                    "baseline_total_log_likelihood": baseline_total,
                    "baseline_mean_log_score": baseline_total / total_responses,
                    "delta_mean_log_score_vs_intercept": (
                        total_log_likelihood - baseline_total
                    )
                    / total_responses,
                }
            )
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate absolute-threshold structures with grouped held-out predictions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fit-target",
        choices=THRESHOLD_FIT_TARGETS,
        default="set",
        help="Training structure used to fit the one shared threshold.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = pd.read_csv(args.input)
    predictions, folds = leave_one_group_out_predictions(
        observations,
        fit_target=args.fit_target,
        scale=args.scale,
    )
    summary = summarize_out_of_fold_predictions(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "novel_focus_oof_predictions.csv"
    folds_path = args.output_dir / "novel_focus_fold_thresholds.csv"
    summary_path = args.output_dir / "novel_focus_summary.csv"
    predictions.to_csv(predictions_path, index=False)
    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("[complete] wrote development-only absolute-threshold outputs")
    print(f"  predictions={predictions_path}")
    print(f"  folds={folds_path}")
    print(f"  summary={summary_path}")
    print(f"  folds={len(folds)}")
    print(f"  threshold_mean={folds['threshold'].mean():.6f}")
    print(f"  threshold_min={folds['threshold'].min():.6f}")
    print(f"  threshold_max={folds['threshold'].max():.6f}")
    print(f"  boundary_fits={int(folds['threshold_at_boundary'].sum())}")
    if not observations["model_provenance_complete"].astype(bool).all():
        print("[warning] Input contains scores with incomplete Qwen revision provenance.")
    print("[warning] Threshold is fit on novel focus data only and is not the final cross-dataset threshold.")


if __name__ == "__main__":
    main()
