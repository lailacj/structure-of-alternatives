"""Select sampled-prefix Top-K and Top-p boundaries in grouped CV folds.

Input is a candidate prediction grid generated after Qwen distribution scoring.
Each row is one analysis unit, one candidate boundary, and one variant.  The
selector uses only Set's balanced training log likelihood, then reuses the
selected boundary for Set, Conjunction, and Disjunction in the held-out fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPSILON = 1e-10
VARIANTS = ("top_k", "top_p")
STRUCTURES = ("set", "ordering", "conjunction", "disjunction")
REQUIRED_COLUMNS = {
    "variant", "boundary", "analysis_dataset_id", "analysis_label",
    "analysis_unit_id", "cv_fold", "human_rate",
    *(f"{structure}_probability" for structure in STRUCTURES),
}


def item_log_score(human_rate: pd.Series, probability: pd.Series) -> pd.Series:
    y = pd.to_numeric(human_rate, errors="raise")
    p = pd.to_numeric(probability, errors="raise")
    if y.isna().any() or p.isna().any() or ((y < 0) | (y > 1)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Human rates and probabilities must be finite values in [0, 1]")
    p = p.clip(EPSILON, 1.0 - EPSILON)
    return y * np.log(p) + (1.0 - y) * np.log1p(-p)


def _validate_grid(grid: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(grid.columns)
    if missing:
        raise ValueError(f"Prediction grid is missing columns: {sorted(missing)}")
    if set(grid["variant"].unique()).difference(VARIANTS):
        raise ValueError("Prediction grid contains an unknown variant")
    keys = ["variant", "boundary", "analysis_dataset_id", "analysis_unit_id"]
    if grid.duplicated(keys).any():
        raise ValueError("Prediction grid has duplicate variant/boundary/unit rows")
    fold_by_unit = grid.groupby(["analysis_dataset_id", "analysis_unit_id"])["cv_fold"].nunique()
    if not fold_by_unit.eq(1).all():
        raise ValueError("One analysis unit was assigned to more than one fold")
    variants_by_unit = grid.groupby(["analysis_dataset_id", "analysis_unit_id"])["variant"].nunique()
    if not variants_by_unit.eq(len(VARIANTS)).all():
        raise ValueError("Every analysis unit must have both Top-K and Top-p rows")


def select_boundaries(grid: pd.DataFrame) -> pd.DataFrame:
    """Return one training-only selected boundary for each variant and fold."""
    _validate_grid(grid)
    records = []
    for variant in VARIANTS:
        variant_rows = grid.loc[grid["variant"].eq(variant)].copy()
        for heldout_fold in sorted(variant_rows["cv_fold"].unique()):
            training = variant_rows.loc[variant_rows["cv_fold"].ne(heldout_fold)].copy()
            training["set_log_score"] = item_log_score(training["human_rate"], training["set_probability"])
            candidates = (
                training.groupby(["boundary", "analysis_dataset_id"], as_index=False)["set_log_score"]
                .mean()
                .groupby("boundary", as_index=False)["set_log_score"]
                .mean()
                .rename(columns={"set_log_score": "balanced_training_set_log_score"})
            )
            # Stable tie-breaker: smaller K / p is preferred if scores are identical.
            best = candidates.sort_values(
                ["balanced_training_set_log_score", "boundary"], ascending=[False, True]
            ).iloc[0]
            records.append({
                "variant": variant,
                "heldout_fold": int(heldout_fold),
                "selected_boundary": float(best["boundary"]),
                "balanced_training_set_log_score": float(best["balanced_training_set_log_score"]),
                "training_unit_count": int(training[["analysis_dataset_id", "analysis_unit_id"]].drop_duplicates().shape[0]),
                "heldout_unit_count": int(variant_rows.loc[variant_rows["cv_fold"].eq(heldout_fold), ["analysis_dataset_id", "analysis_unit_id"]].drop_duplicates().shape[0]),
            })
    return pd.DataFrame.from_records(records).sort_values(["variant", "heldout_fold"], ignore_index=True)


def out_of_fold_predictions(grid: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    selected = selections.rename(columns={"heldout_fold": "cv_fold", "selected_boundary": "boundary"})
    out = grid.merge(selected[["variant", "cv_fold", "boundary"]], on=["variant", "cv_fold", "boundary"], how="inner", validate="many_to_one")
    expected = grid[["variant", "analysis_dataset_id", "analysis_unit_id"]].drop_duplicates().groupby("variant").size()
    observed = out[["variant", "analysis_dataset_id", "analysis_unit_id"]].drop_duplicates().groupby("variant").size()
    if not observed.equals(expected):
        raise RuntimeError("Fold selection did not produce exactly one prediction per unit and variant")
    return out.sort_values(["variant", "analysis_dataset_id", "analysis_unit_id"], ignore_index=True)


def summarize_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (variant, row_id, label), rows in predictions.groupby(["variant", "analysis_dataset_id", "analysis_label"], sort=True):
        record = {"variant": variant, "analysis_dataset_id": row_id, "dataset": label, "N": len(rows)}
        for structure in STRUCTURES:
            x = rows[f"{structure}_probability"]
            y = rows["human_rate"]
            record[f"{structure}_pearson_r"] = float(x.corr(y)) if x.nunique() > 1 and y.nunique() > 1 else float("nan")
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_log_scores(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for (variant, row_id, label), rows in predictions.groupby(
        ["variant", "analysis_dataset_id", "analysis_label"], sort=True
    ):
        for structure in STRUCTURES:
            scores = item_log_score(rows["human_rate"], rows[f"{structure}_probability"])
            records.append({
                "variant": variant,
                "analysis_dataset_id": row_id,
                "dataset": label,
                "structure": structure,
                "N": len(rows),
                "mean_oof_log_score": float(scores.mean()),
            })
    by_dataset = pd.DataFrame.from_records(records)
    balanced = (
        by_dataset.groupby(["variant", "structure"], as_index=False)
        .agg(
            balanced_mean_oof_log_score=("mean_oof_log_score", "mean"),
            dataset_count=("analysis_dataset_id", "nunique"),
            total_unit_count=("N", "sum"),
        )
        .sort_values(["variant", "balanced_mean_oof_log_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return by_dataset, balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = pd.read_csv(args.prediction_grid)
    selections = select_boundaries(grid)
    predictions = out_of_fold_predictions(grid, selections)
    correlations = summarize_correlations(predictions)
    log_scores_by_dataset, balanced_log_scores = summarize_log_scores(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections.to_csv(args.output_dir / "fold_selections.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    correlations.to_csv(args.output_dir / "correlations.csv", index=False)
    log_scores_by_dataset.to_csv(
        args.output_dir / "oof_log_scores_by_dataset_and_structure.csv", index=False
    )
    balanced_log_scores.to_csv(
        args.output_dir / "oof_balanced_log_scores_by_structure.csv", index=False
    )
    print(
        "[complete] wrote fold selections, OOF predictions, correlations, and "
        f"OOF log-score summaries to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
