"""Stable experiment orchestration.

The runner is dataset-agnostic:
it gets samples from a sampler, runs registered models, and writes standard outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

try:
    from .models import ModelSpec, get_models
    from .samplers import ContextSampler
except ImportError:
    from models import ModelSpec, get_models
    from samplers import ContextSampler

RESULT_COLUMNS = [
    "set_boundary",
    "num_reps",
    "context",
    "trigger",
    "query",
    "neg",
    "log_likelihood",
    "negation_probability",
    "probability_query_observed",
]


def clean_word(word: str) -> str:
    token = str(word).strip().lower()
    if token.startswith("a "):
        return token[2:]
    if token.startswith("an "):
        return token[3:]
    return token


def prepare_experimental_data(experimental_data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized cleaned query/trigger columns."""
    df = experimental_data.copy()
    if "cleaned_query" not in df.columns and "query" in df.columns:
        df["cleaned_query"] = df["query"].apply(clean_word)
    if "cleaned_trigger" not in df.columns and "trigger" in df.columns:
        df["cleaned_trigger"] = df["trigger"].apply(clean_word)
    if "story" in df.columns:
        df = df.sort_values(by="story")
    return df


def _resolve_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Neither '{preferred}' nor '{fallback}' exists in experimental_data")


def _append_csv(rows: Sequence[list], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)


def run_experiment(
    experimental_data: pd.DataFrame,
    sampler: ContextSampler,
    *,
    set_boundaries: Iterable[int],
    num_reps: int,
    model_specs: Sequence[ModelSpec] | None = None,
    results_dir: str | Path | None = None,
    file_suffix: str = "",
) -> Dict[str, pd.DataFrame]:

    """Run the full experiment loop and return in-memory result DataFrames."""
    df = prepare_experimental_data(experimental_data)
    context_col = _resolve_col(df, "story", "context")
    query_col = _resolve_col(df, "cleaned_query", "query")
    trigger_col = _resolve_col(df, "cleaned_trigger", "trigger")
    neg_col = _resolve_col(df, "neg", None)

    models = list(model_specs) if model_specs is not None else get_models()
    all_rows: Dict[str, list] = {spec.name: [] for spec in models}
    all_missing: list = []

    out_dir = Path(results_dir) if results_dir is not None else None
    suffix = file_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    contexts = [str(c) for c in df[context_col].dropna().unique()]
    sampler.prepare_contexts(contexts)

    for set_boundary in set_boundaries:
        print(f"Set Boundary: {set_boundary}")
        context_samples = sampler.sample_contexts(contexts, set_boundary=set_boundary, num_reps=num_reps)
        boundary_rows: Dict[str, list] = {spec.name: [] for spec in models}
        boundary_missing: list = []

        for _, row in df.iterrows():
            context = str(row[context_col])
            query = str(row[query_col])
            trigger = str(row[trigger_col])
            query_negated = int(row[neg_col])

            if not sampler.supports_token(context, query):
                boundary_missing.append([set_boundary, context, trigger, query, "query_not_in_context"])
                continue

            needs_trigger = any(spec.requires_trigger for spec in models)
            if needs_trigger and not sampler.supports_token(context, trigger):
                boundary_missing.append([set_boundary, context, trigger, query, "trigger_not_in_context"])
                continue

            samples = context_samples[context]
            for spec in models:
                log_likelihood, negation_probability, prob_obs = spec.fn(
                    samples=samples,
                    query=query,
                    trigger=trigger,
                    query_negated=query_negated,
                )
                boundary_rows[spec.name].append(
                    [
                        set_boundary,
                        num_reps,
                        context,
                        trigger,
                        query,
                        query_negated,
                        log_likelihood,
                        negation_probability,
                        prob_obs,
                    ]
                )

        for name, rows in boundary_rows.items():
            all_rows[name].extend(rows)
            if out_dir is not None:
                _append_csv(rows, out_dir / f"{name}_results{suffix}.csv")

        all_missing.extend(boundary_missing)
        if out_dir is not None and boundary_missing:
            missing_path = out_dir / f"missing_trials{suffix}.csv"
            pd.DataFrame(
                boundary_missing,
                columns=["set_boundary", "context", "trigger", "query", "reason"],
            ).to_csv(missing_path, mode="a", header=not missing_path.exists(), index=False)

    results = {name: pd.DataFrame(rows, columns=RESULT_COLUMNS) for name, rows in all_rows.items()}
    results["missing_trials"] = pd.DataFrame(
        all_missing,
        columns=["set_boundary", "context", "trigger", "query", "reason"],
    )
    return results
