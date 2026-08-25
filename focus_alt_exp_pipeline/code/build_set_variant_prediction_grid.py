"""Turn cluster Qwen distribution scores into a Top-K / Top-p prediction grid.

The scorer samples one shared bank of weighted orderings per prompt.  Top-K
uses the first K sampled words; Top-p uses the shortest sampled prefix whose
original normalized candidate probabilities reach p.  The output is consumed
by ``evaluate_set_variant_grid.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROWS = PIPELINE_DIR / "scoring_manifests" / "set_variant_qwen" / "source_rows.csv"

ROW_LABELS = {
    "hu_rx22": "Hu: Ronai & Xiang (2022)",
    "hu_pvt21": "Hu: Pankratz & van Tiel (2021)",
    "hu_g18": "Hu: Gotzner et al. (2018)",
    "hu_vt16": "Hu: van Tiel et al. (2016)",
    "rnx_esi": "Ronai & Xiang (2024): ESI",
    "rnx_eweak": "Ronai & Xiang (2024): Eweak",
    "rnx_estrong": "Ronai & Xiang (2024): Estrong",
    "rnx_eonly": "Ronai & Xiang (2024): Eonly",
    "rnx_eonlystrong": "Ronai & Xiang (2024): Eonlystrong",
    "novel_focus": "Novel focus alternatives",
}


def _parse_values(raw: str, *, kind: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"No {kind} values were supplied")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {kind} values were supplied")
    return sorted(values)


def _dataset_id(row: pd.Series) -> str:
    family, dataset, condition = str(row.dataset_family), str(row.dataset), str(row.condition)
    if family == "hu_2023_benchmark":
        return f"hu_{dataset}"
    if family == "ronai_xiang_2024":
        return f"rnx_{condition.lower()}"
    if family == "novel_focus":
        return "novel_focus"
    raise ValueError(f"Unsupported dataset family: {family}")


def _load_vocab(vocab_manifest: Path) -> tuple[list[str], dict[str, int]]:
    manifest = json.loads(vocab_manifest.read_text(encoding="utf-8"))
    tokens: list[str] = []
    for source in manifest["sources"]:
        path = Path(source["path"])
        with path.open(encoding="utf-8") as stream:
            tokens.extend(line.rstrip("\n") for line in stream)
    lookup = {token.strip().lower(): index for index, token in enumerate(tokens) if token.strip()}
    if len(lookup) != len(tokens):
        raise ValueError("Candidate vocabulary contains duplicate or empty normalized tokens")
    return tokens, lookup


def _sample_prefixes(probabilities: np.ndarray, *, num_reps: int, prefix_size: int, rng: np.random.Generator) -> np.ndarray:
    if prefix_size <= 0 or prefix_size > len(probabilities):
        raise ValueError("prefix_size must be positive and no larger than candidate support")
    return np.vstack([rng.choice(len(probabilities), size=prefix_size, replace=False, p=probabilities) for _ in range(num_reps)])


def _probability_records(
    sampled: np.ndarray,
    probabilities: np.ndarray,
    *,
    query_index: int,
    trigger_index: int,
    k_values: list[int],
    p_values: list[float],
) -> list[dict[str, float]]:
    num_reps, prefix_size = sampled.shape
    position = np.full((num_reps, 2), prefix_size, dtype=np.int32)
    for row_index, row in enumerate(sampled):
        query_positions = np.flatnonzero(row == query_index)
        trigger_positions = np.flatnonzero(row == trigger_index)
        if len(query_positions):
            position[row_index, 0] = query_positions[0]
        if len(trigger_positions):
            position[row_index, 1] = trigger_positions[0]
    query_position, trigger_position = position[:, 0], position[:, 1]
    ordering_probability = float(probabilities[query_index] / (probabilities[query_index] + probabilities[trigger_index]))
    prefix_mass = probabilities[sampled].cumsum(axis=1)
    records: list[dict[str, float]] = []
    for value in k_values:
        in_set = query_position < value
        conjunction = in_set & (query_position < trigger_position)
        set_probability = float(in_set.mean())
        conjunction_probability = float(conjunction.mean())
        records.append({
            "variant": "top_k", "boundary": float(value),
            "set_probability": set_probability,
            "ordering_probability": ordering_probability,
            "conjunction_probability": conjunction_probability,
            "disjunction_probability": min(1.0, set_probability + ordering_probability - conjunction_probability),
        })
    for value in p_values:
        cutoff = np.argmax(prefix_mass >= value, axis=1)
        if not np.all(prefix_mass[np.arange(num_reps), cutoff] >= value):
            raise ValueError("Sampled prefix was too short to reach a requested p value")
        in_set = query_position <= cutoff
        conjunction = in_set & (query_position < trigger_position)
        set_probability = float(in_set.mean())
        conjunction_probability = float(conjunction.mean())
        records.append({
            "variant": "top_p", "boundary": float(value),
            "set_probability": set_probability,
            "ordering_probability": ordering_probability,
            "conjunction_probability": conjunction_probability,
            "disjunction_probability": min(1.0, set_probability + ordering_probability - conjunction_probability),
        })
    return records


def _assign_folds(units: pd.DataFrame, fold_count: int) -> pd.Series:
    assignments: dict[str, int] = {}
    groups = units[["dataset_family", "dataset", "cv_group_id"]].drop_duplicates()
    for _, stratum in groups.groupby(["dataset_family", "dataset"], sort=True):
        group_ids = sorted(stratum.cv_group_id.astype(str))
        if len(group_ids) < fold_count:
            raise ValueError("Every dataset stratum must contain at least fold_count groups")
        assignments.update({group_id: index % fold_count for index, group_id in enumerate(group_ids)})
    return units.cv_group_id.map(assignments).astype(int)


def _aggregate_to_units(raw: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    non_hu = raw.loc[~raw.dataset_family.eq("hu_2023_benchmark")].copy()
    non_hu["analysis_dataset_id"] = non_hu.apply(_dataset_id, axis=1)
    non_hu["analysis_unit_id"] = non_hu.item_id.astype(str)
    non_hu["cv_group_id"] = non_hu.dataset_family.astype(str) + "::" + non_hu.dataset.astype(str) + "::" + non_hu.group_id.astype(str)
    hu = raw.loc[raw.dataset_family.eq("hu_2023_benchmark") & raw.hu_original_analysis_included.astype(bool)].copy()
    hu["analysis_dataset_id"] = hu.apply(_dataset_id, axis=1)
    hu["analysis_unit_id"] = hu.dataset.astype(str) + "::" + hu.scale_id.astype(str)
    hu["cv_group_id"] = hu.dataset_family.astype(str) + "::" + hu.dataset.astype(str) + "::" + hu.scale_id.astype(str)
    probability_columns = [f"{structure}_probability" for structure in ("set", "ordering", "conjunction", "disjunction")]
    fixed_columns = ["variant", "boundary", "analysis_dataset_id", "analysis_unit_id", "cv_group_id"]
    hu = hu.groupby(fixed_columns, as_index=False).agg(human_rate=("human_rate", "mean"), dataset_family=("dataset_family", "first"), dataset=("dataset", "first"), **{column: (column, "mean") for column in probability_columns})
    non_hu = non_hu[fixed_columns + ["human_rate", "dataset_family", "dataset", *probability_columns]]
    units = pd.concat([non_hu, hu], ignore_index=True)
    units["analysis_label"] = units.analysis_dataset_id.map(ROW_LABELS)
    if units.analysis_label.isna().any():
        raise ValueError("Unknown analysis dataset ID")
    unit_keys = units[["analysis_dataset_id", "analysis_unit_id", "dataset_family", "dataset", "cv_group_id"]].drop_duplicates()
    unit_keys["cv_fold"] = _assign_folds(unit_keys, fold_count)
    return units.merge(
        unit_keys[["analysis_dataset_id", "analysis_unit_id", "cv_fold"]],
        on=["analysis_dataset_id", "analysis_unit_id"],
        how="left",
        validate="many_to_one",
    )


def build_prediction_grid(source_rows: pd.DataFrame, *, log_probs_dir: Path, k_values: list[int], p_values: list[float], num_reps: int, prefix_size: int, seed: int, fold_count: int) -> pd.DataFrame:
    _, token_lookup = _load_vocab(log_probs_dir / "vocab_manifest.json")
    rng = np.random.default_rng(seed)
    raw_records = []
    for prompt_id, rows in source_rows.groupby("prompt_id", sort=True):
        path = log_probs_dir / f"{prompt_id}.log_probs.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing Qwen scores for prompt {prompt_id}: {path}")
        log_probs = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64)
        finite = np.isfinite(log_probs)
        shifted = log_probs[finite] - log_probs[finite].max()
        support_probs = np.exp(shifted)
        support_probs /= support_probs.sum()
        support_indices = np.flatnonzero(finite)
        global_to_support = np.full(len(log_probs), -1, dtype=np.int64)
        global_to_support[support_indices] = np.arange(len(support_indices))
        sampled = _sample_prefixes(support_probs, num_reps=num_reps, prefix_size=min(prefix_size, len(support_probs)), rng=rng)
        for _, row in rows.iterrows():
            query = str(row.query).strip().lower()
            trigger = str(row.trigger).strip().lower()
            if query not in token_lookup or trigger not in token_lookup:
                raise ValueError(f"Candidate is absent from scored vocabulary: {trigger!r}, {query!r}")
            query_index, trigger_index = global_to_support[token_lookup[query]], global_to_support[token_lookup[trigger]]
            if query_index < 0 or trigger_index < 0:
                raise ValueError(f"Candidate received a non-finite Qwen score: {trigger!r}, {query!r}")
            for prediction in _probability_records(sampled, support_probs, query_index=int(query_index), trigger_index=int(trigger_index), k_values=k_values, p_values=p_values):
                raw_records.append({**row.to_dict(), **prediction})
    return _aggregate_to_units(pd.DataFrame.from_records(raw_records), fold_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--log-probs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k-values", default="1,2,3,5,10,20,30,50,75,100")
    parser.add_argument("--top-p-values", default="0.25,0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--num-reps", type=int, default=500)
    parser.add_argument("--max-prefix-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fold-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(value) for value in _parse_values(args.top_k_values, kind="Top-K")]
    p_values = _parse_values(args.top_p_values, kind="Top-p")
    if any(value <= 0 for value in k_values) or any(value <= 0 or value >= 1 for value in p_values):
        raise ValueError("K values must be positive and p values must be strictly between 0 and 1")
    if args.num_reps <= 0 or args.max_prefix_size <= 0:
        raise ValueError("num-reps and max-prefix-size must be positive")
    grid = build_prediction_grid(pd.read_csv(args.source_rows), log_probs_dir=args.log_probs_dir, k_values=k_values, p_values=p_values, num_reps=args.num_reps, prefix_size=max(args.max_prefix_size, max(k_values)), seed=args.seed, fold_count=args.fold_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.output, index=False)
    print(f"[complete] rows={len(grid)} output={args.output}")


if __name__ == "__main__":
    main()
