"""CLI entrypoint for the focus_alt_exp pipeline.

Examples:
  python run_experiment.py --dataset cloze --set-boundaries 2,3,4 --num-reps 100
  python run_experiment.py --dataset frequency --frequency-background-vocab-size 10000 --model-names ordering --set-boundaries 5,10,15 --num-reps 500
  python run_experiment.py --dataset qwen --qwen-top-vocab-size 100000 --model-names ordering,set,conjunction,disjunction --num-reps 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .data_utils import normalize_unique_tokens, prepare_experimental_data, resolve_context_col
    from .models import get_models
    from .runner import run_experiment
    from .samplers import ClozeSampler, FrequencySampler, QwenSampler
except ImportError:
    from data_utils import normalize_unique_tokens, prepare_experimental_data, resolve_context_col
    from models import get_models
    from runner import run_experiment
    from samplers import ClozeSampler, FrequencySampler, QwenSampler


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_CLOZE_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "cloze_data" / "all_cloze_prob_data_preprocessed.csv"
DEFAULT_RESULTS_DIR = ROOT_DIR / "focus_alt_exp_pipeline" / "results"
DEFAULT_FREQUENCY_1GRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_1gram_counts.tsv"
DEFAULT_FREQUENCY_2GRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_2gram_counts.tsv"
DEFAULT_FREQUENCY_BACKGROUND_VOCAB_SIZE = 800_000
DEFAULT_QWEN_LOG_PROBS_DIR = ROOT_DIR.parent / "ngrams" / "qwen_context_balanced_log_probs"
DEFAULT_QWEN_TOP_VOCAB_SIZE = 100_000
RESULTS_SUBDIR_BY_DATASET = {
    "cloze": "cloze_probability",
    "frequency": "frequency",
    "qwen": "qwen",
}


def _parse_csv_list(raw: str) -> List[str]:
    values = [chunk.strip() for chunk in raw.split(",")]
    return [value for value in values if value]


def _resolve_set_boundaries(args: argparse.Namespace) -> List[int]:
    if args.set_boundaries:
        boundaries = [int(value) for value in _parse_csv_list(args.set_boundaries)]
        if not boundaries:
            raise ValueError("--set-boundaries was provided but no values were parsed.")
        return boundaries

    if args.set_step <= 0:
        raise ValueError("--set-step must be > 0")
    if args.set_stop <= args.set_start:
        raise ValueError("--set-stop must be greater than --set-start")
    return list(range(args.set_start, args.set_stop, args.set_step))


def _normalize_suffix(raw_suffix: str | None) -> str:
    suffix = "" if raw_suffix is None else raw_suffix.strip()
    if suffix and not suffix.startswith("_"):
        return f"_{suffix}"
    return suffix


def _default_results_subdir(dataset: str) -> str:
    return RESULTS_SUBDIR_BY_DATASET.get(dataset, dataset)


def _extract_frequency_required_tokens(experimental_data: pd.DataFrame) -> List[str]:
    prepared = prepare_experimental_data(experimental_data)
    query_col = "cleaned_query" if "cleaned_query" in prepared.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in prepared.columns else "trigger"
    values = pd.concat([prepared[query_col], prepared[trigger_col]], ignore_index=True).dropna()
    return normalize_unique_tokens(values.tolist())


def _extract_required_tokens_by_context(experimental_data: pd.DataFrame) -> dict[str, List[str]]:
    prepared = prepare_experimental_data(experimental_data)
    context_col = resolve_context_col(prepared)
    query_col = "cleaned_query" if "cleaned_query" in prepared.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in prepared.columns else "trigger"

    required_by_context: dict[str, List[str]] = {}
    for context, subset in prepared.groupby(context_col, sort=False):
        tokens = pd.concat([subset[query_col], subset[trigger_col]], ignore_index=True).dropna()
        required_by_context[str(context)] = normalize_unique_tokens(tokens.tolist())
    return required_by_context


def _build_sampler(args: argparse.Namespace, experimental_data: pd.DataFrame):
    if args.dataset == "cloze":
        cloze_df = pd.read_csv(args.cloze_data)
        sampler = ClozeSampler(
            cloze_df=cloze_df,
            context_col=args.cloze_context_col,
            word_col=args.cloze_word_col,
            prob_col=args.cloze_prob_col,
            seed=args.seed,
        )
        contexts = set(cloze_df[args.cloze_context_col].astype(str).unique())
        return sampler, contexts

    if args.dataset == "frequency":
        sampler = FrequencySampler(
            required_tokens=_extract_frequency_required_tokens(experimental_data),
            unigram_counts_path=args.frequency_1gram_counts,
            bigram_counts_path=args.frequency_2gram_counts,
            background_vocab_size=args.frequency_background_vocab_size,
            max_vocab_size=args.frequency_max_vocab_size,
            seed=args.seed,
        )
        return sampler, None
    if args.dataset == "qwen":
        sampler = QwenSampler(
            required_tokens_by_context=_extract_required_tokens_by_context(experimental_data),
            log_probs_dir=args.qwen_log_probs_dir,
            top_vocab_size=args.qwen_top_vocab_size,
            seed=args.seed,
        )
        return sampler, set(sampler.available_contexts)
    raise ValueError(f"Unsupported dataset '{args.dataset}'")


def _resolve_models(args: argparse.Namespace):
    if args.model_names:
        names = _parse_csv_list(args.model_names)
        return get_models(model_names=names)
    return get_models(include_baselines=args.include_baselines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focus_alt_exp pipeline")

    parser.add_argument(
        "--dataset",
        choices=["cloze", "frequency", "qwen"],
        required=True,
    )
    parser.add_argument("--experimental-data", type=Path, default=DEFAULT_EXPERIMENTAL_DATA)
    parser.add_argument(
        "--contexts",
        type=str,
        default="",
        help="Optional comma-separated explicit context names to keep.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Optional limit on number of contexts (keeps first N by row order).",
    )
    parser.add_argument("--num-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--set-boundaries", type=str, default="")
    parser.add_argument("--set-start", type=int, default=3)
    parser.add_argument("--set-stop", type=int, default=300)
    parser.add_argument("--set-step", type=int, default=3)

    parser.add_argument("--model-names", type=str, default="")
    parser.add_argument("--include-baselines", action="store_true")

    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="",
        help=(
            "Optional subdirectory under --results-dir for this run. "
            "For example: cloze_probability"
        ),
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default=None,
        help="Optional suffix for output filenames. Defaults to the selected dataset name.",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to existing output CSVs instead of overwriting them at the start of the run.",
    )
    parser.add_argument("--print-head", type=int, default=10)

    parser.add_argument("--cloze-data", type=Path, default=DEFAULT_CLOZE_DATA)
    parser.add_argument("--cloze-context-col", type=str, default="context")
    parser.add_argument("--cloze-word-col", type=str, default="word")
    parser.add_argument("--cloze-prob-col", type=str, default="cloze_probability")

    parser.add_argument(
        "--frequency-1gram-counts",
        type=Path,
        default=DEFAULT_FREQUENCY_1GRAM_COUNTS,
        help="Path to unigram count TSV used by --dataset frequency.",
    )
    parser.add_argument(
        "--frequency-2gram-counts",
        type=Path,
        default=DEFAULT_FREQUENCY_2GRAM_COUNTS,
        help="Path to bigram count TSV used by --dataset frequency.",
    )
    parser.add_argument(
        "--frequency-background-vocab-size",
        type=int,
        default=None,
        help=(
            "Optional top-K global Google Ngram vocabulary to include alongside all "
            "experimental query/trigger tokens that must remain in the frequency model. "
            f"Defaults to {DEFAULT_FREQUENCY_BACKGROUND_VOCAB_SIZE} when --dataset frequency "
            "and --frequency-max-vocab-size is not set."
        ),
    )
    parser.add_argument(
        "--frequency-max-vocab-size",
        type=int,
        default=None,
        help="Optional top-N frequency vocab cap. Currently only supported with --dataset frequency --model-names set.",
    )
    parser.add_argument(
        "--qwen-log-probs-dir",
        type=Path,
        default=DEFAULT_QWEN_LOG_PROBS_DIR,
        help="Directory containing precomputed per-context Qwen log-prob arrays and vocab_manifest.json.",
    )
    parser.add_argument(
        "--qwen-top-vocab-size",
        type=int,
        default=DEFAULT_QWEN_TOP_VOCAB_SIZE,
        help=(
            "Top-M Qwen continuations to keep per context for sampled set/conjunction estimation. "
            "Experimental query/trigger tokens for that context are force-included even when they "
            "fall outside the top-M support."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file_suffix is None:
        args.file_suffix = args.dataset
    if not args.results_subdir:
        args.results_subdir = _default_results_subdir(args.dataset)
    if (
        args.dataset == "frequency"
        and args.frequency_background_vocab_size is None
        and args.frequency_max_vocab_size is None
    ):
        args.frequency_background_vocab_size = DEFAULT_FREQUENCY_BACKGROUND_VOCAB_SIZE

    if args.num_reps <= 0:
        raise ValueError("--num-reps must be > 0")

    experimental_data = pd.read_csv(args.experimental_data)
    boundaries = _resolve_set_boundaries(args)
    models = _resolve_models(args)
    if (
        args.frequency_background_vocab_size is not None
        and args.frequency_max_vocab_size is not None
    ):
        raise ValueError(
            "--frequency-background-vocab-size and --frequency-max-vocab-size cannot be used together"
        )
    if args.frequency_background_vocab_size is not None:
        if args.frequency_background_vocab_size <= 0:
            raise ValueError("--frequency-background-vocab-size must be > 0")
        if args.dataset != "frequency":
            raise ValueError(
                "--frequency-background-vocab-size can only be used with --dataset frequency"
            )
    if args.frequency_max_vocab_size is not None:
        if args.frequency_max_vocab_size <= 0:
            raise ValueError("--frequency-max-vocab-size must be > 0")
        if args.dataset != "frequency":
            raise ValueError("--frequency-max-vocab-size can only be used with --dataset frequency")
        if len(models) != 1 or models[0].name != "set":
            raise ValueError(
                "--frequency-max-vocab-size is only supported with --dataset frequency --model-names set"
            )
    if args.qwen_top_vocab_size <= 0:
        raise ValueError("--qwen-top-vocab-size must be > 0")

    context_col = resolve_context_col(experimental_data)
    sampler = None
    available_contexts = None
    if args.dataset != "frequency":
        sampler, available_contexts = _build_sampler(args, experimental_data)

    if available_contexts is not None:
        original_rows = len(experimental_data)
        experimental_data = experimental_data[
            experimental_data[context_col].astype(str).isin(available_contexts)
        ].copy()
        filtered_rows = len(experimental_data)
        dropped_rows = original_rows - filtered_rows
        if dropped_rows > 0:
            print(
                f"Filtered experimental_data by available contexts: kept={filtered_rows}, dropped={dropped_rows}"
            )
        if filtered_rows == 0:
            raise ValueError("No experimental rows remain after context filtering.")

    if args.contexts:
        requested_contexts = set(_parse_csv_list(args.contexts))
        experimental_data = experimental_data[
            experimental_data[context_col].astype(str).isin(requested_contexts)
        ].copy()
        print(f"Applied explicit contexts filter: contexts={sorted(requested_contexts)}")

    if args.max_contexts is not None:
        if args.max_contexts <= 0:
            raise ValueError("--max-contexts must be > 0 when provided")
        kept_contexts = []
        seen = set()
        for ctx in experimental_data[context_col].astype(str):
            if ctx in seen:
                continue
            seen.add(ctx)
            kept_contexts.append(ctx)
            if len(kept_contexts) == args.max_contexts:
                break
        experimental_data = experimental_data[
            experimental_data[context_col].astype(str).isin(set(kept_contexts))
        ].copy()
        print(f"Applied max-contexts filter: max_contexts={args.max_contexts}, kept={kept_contexts}")

    if len(experimental_data) == 0:
        raise ValueError("No experimental rows remain after optional context filters.")

    if args.dataset == "frequency":
        sampler, _ = _build_sampler(args, experimental_data)

    results_dir = None
    if not args.no_write:
        results_dir = args.results_dir
        if args.results_subdir:
            results_dir = results_dir / args.results_subdir
    results = run_experiment(
        experimental_data=experimental_data,
        sampler=sampler,
        set_boundaries=boundaries,
        num_reps=args.num_reps,
        model_specs=models,
        results_dir=results_dir,
        file_suffix=args.file_suffix,
        overwrite_existing=not args.append_existing,
    )

    print("Run complete.")
    print(f"dataset={args.dataset}")
    print(f"set_boundaries={boundaries}")
    print(f"num_reps={args.num_reps}")
    print(f"models={[spec.name for spec in models]}")
    print(f"missing_trials={len(results['missing_trials'])}")

    if results_dir is not None:
        suffix = _normalize_suffix(args.file_suffix)
        print(f"results_dir={results_dir}")
        for spec in models:
            print(f"  wrote: {results_dir / f'{spec.name}_results{suffix}.csv'}")
        print(f"  wrote: {results_dir / f'missing_trials{suffix}.csv'}")
    else:
        print("results_dir=None (no CSVs written)")

    if args.print_head > 0:
        for key, df in results.items():
            print(f"\n[{key}] rows={len(df)}")
            if not df.empty:
                print(df.head(args.print_head).to_string(index=False))


if __name__ == "__main__":
    main()
