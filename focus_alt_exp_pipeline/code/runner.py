"""Stable experiment orchestration.

The runner is dataset-agnostic:
it gets samples from a sampler, runs registered models, and writes standard outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

try:
    from .data_utils import prepare_experimental_data, resolve_context_col
    from .models import ModelSpec, get_models, probability_to_model_result
    from .samplers import ContextSampler
except ImportError:
    from data_utils import prepare_experimental_data, resolve_context_col
    from models import ModelSpec, get_models, probability_to_model_result
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


def _clear_previous_outputs(
    out_dir: Path | None,
    *,
    models: Sequence[ModelSpec],
    suffix: str,
) -> None:
    if out_dir is None:
        return

    targets = [out_dir / f"{spec.name}_results{suffix}.csv" for spec in models]
    targets.append(out_dir / f"missing_trials{suffix}.csv")
    for path in targets:
        if path.exists():
            path.unlink()


def run_experiment(
    experimental_data: pd.DataFrame,
    sampler: ContextSampler,
    *,
    set_boundaries: Iterable[int],
    num_reps: int,
    model_specs: Sequence[ModelSpec] | None = None,
    results_dir: str | Path | None = None,
    file_suffix: str = "",
    overwrite_existing: bool = True,
) -> Dict[str, pd.DataFrame]:

    """Run the full experiment loop and return in-memory result DataFrames."""
    resolved_set_boundaries = [int(boundary) for boundary in set_boundaries]
    df = prepare_experimental_data(experimental_data)
    context_col = resolve_context_col(df)
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
    if overwrite_existing:
        _clear_previous_outputs(out_dir, models=models, suffix=suffix)

    contexts = [str(c) for c in df[context_col].dropna().unique()]
    sampler.prepare_contexts(contexts)
    sampler.prepare_run(
        contexts=contexts,
        set_boundaries=resolved_set_boundaries,
        num_reps=num_reps,
        model_names=[spec.name for spec in models],
    )

    for set_boundary in resolved_set_boundaries:
        print(f"Set Boundary: {set_boundary}")
        needs_sampled_contexts = any(
            spec.uses_samples
            and not sampler.supports_direct_model(spec.name)
            and not sampler.supports_exact_model(spec.name)
            for spec in models
        )
        context_samples = (
            sampler.sample_contexts(contexts, set_boundary=set_boundary, num_reps=num_reps)
            if needs_sampled_contexts
            else {}
        )
        boundary_rows: Dict[str, list] = {spec.name: [] for spec in models}
        boundary_missing: list = []
        allow_unsupported_query = (
            all(spec.name == "set" for spec in models)
            and sampler.allow_unsupported_query_for_model("set")
        )

        for _, row in df.iterrows():
            context = str(row[context_col])
            query = str(row[query_col])
            trigger = str(row[trigger_col])
            query_negated = int(row[neg_col])

            if not sampler.supports_token(context, query) and not allow_unsupported_query:
                boundary_missing.append([set_boundary, context, trigger, query, "query_not_in_context"])
                continue

            needs_trigger = any(spec.requires_trigger for spec in models)
            if needs_trigger and not sampler.supports_token(context, trigger):
                boundary_missing.append([set_boundary, context, trigger, query, "trigger_not_in_context"])
                continue

            for spec in models:
                direct_result = sampler.direct_model_result(
                    model_name=spec.name,
                    context=context,
                    query=query,
                    trigger=trigger,
                    query_negated=query_negated,
                    set_boundary=set_boundary,
                )
                if direct_result is not None:
                    log_likelihood, negation_probability, prob_obs = direct_result
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
                    continue

                exact_negation_probability = sampler.exact_negation_probability(
                    model_name=spec.name,
                    context=context,
                    query=query,
                    trigger=trigger,
                    set_boundary=set_boundary,
                )
                if exact_negation_probability is not None:
                    log_likelihood, negation_probability, prob_obs = probability_to_model_result(
                        exact_negation_probability,
                        query_negated,
                    )
                else:
                    if spec.uses_samples and context not in context_samples:
                        raise RuntimeError(
                            f"Missing sampled contexts for model '{spec.name}' at set boundary "
                            f"{set_boundary}"
                        )
                    samples = () if not spec.uses_samples else context_samples[context]
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
