"""CLI entrypoint for the focus_alt_exp pipeline.

Examples:
  python run_experiment.py --dataset cloze --set-boundaries 2,3,4 --num-reps 100
  python run_experiment.py --dataset bert --set-start 3 --set-stop 300 --set-step 5 --num-reps 500
  python run_experiment.py --dataset bert_static --set-start 3 --set-stop 300 --set-step 5 --num-reps 500
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .models import get_models
    from .runner import run_experiment
    from .samplers import BertSampler, ClozeSampler, StaticBERTSampler
except ImportError:
    from models import get_models
    from runner import run_experiment
    from samplers import BertSampler, ClozeSampler, StaticBERTSampler


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_CLOZE_DATA = ROOT_DIR / "focus_alt_exp_pipline" / "cloze_data" / "all_cloze_prob.csv"
DEFAULT_BERT_PROMPTS = ROOT_DIR / "prompts" / "prompt_files" / "prompts_llm_only.csv"
DEFAULT_RESULTS_DIR = ROOT_DIR / "focus_alt_exp_pipline" / "results"


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


def _normalize_suffix(raw_suffix: str) -> str:
    suffix = raw_suffix.strip()
    if suffix and not suffix.startswith("_"):
        return f"_{suffix}"
    return suffix


def _build_sampler(args: argparse.Namespace):
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

    if args.dataset == "bert":
        prompts_df = pd.read_csv(args.bert_prompts_data)
        sampler = BertSampler(
            prompts_df=prompts_df,
            context_col=args.bert_context_col,
            prompt_col=args.bert_prompt_col,
            model_name=args.bert_model_name,
            seed=args.seed,
            device=args.device,
        )
        contexts = set(prompts_df[args.bert_context_col].astype(str).unique())
        return sampler, contexts

    sampler = StaticBERTSampler(
        model_name=args.bert_model_name,
        seed=args.seed,
        device=args.device,
    )
    return sampler, None


def _resolve_models(args: argparse.Namespace):
    if args.model_names:
        names = _parse_csv_list(args.model_names)
        return get_models(model_names=names)
    return get_models(include_baselines=args.include_baselines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focus_alt_exp pipeline")

    parser.add_argument("--dataset", choices=["cloze", "bert", "bert_static"], required=True)
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
    parser.add_argument("--set-stop", type=int, default=25)
    parser.add_argument("--set-step", type=int, default=1)

    parser.add_argument("--model-names", type=str, default="")
    parser.add_argument("--include-baselines", action="store_true")

    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--file-suffix", type=str, default="")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--print-head", type=int, default=10)

    parser.add_argument("--cloze-data", type=Path, default=DEFAULT_CLOZE_DATA)
    parser.add_argument("--cloze-context-col", type=str, default="context")
    parser.add_argument("--cloze-word-col", type=str, default="word")
    parser.add_argument("--cloze-prob-col", type=str, default="cloze_probability")

    parser.add_argument("--bert-prompts-data", type=Path, default=DEFAULT_BERT_PROMPTS)
    parser.add_argument("--bert-context-col", type=str, default="story")
    parser.add_argument("--bert-prompt-col", type=str, default="prompt")
    parser.add_argument(
        "--bert-model-name",
        type=str,
        default="google-bert/bert-large-uncased-whole-word-masking",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--hf-offline",
        action="store_true",
        help="Set HF/Transformers offline env vars to avoid network retries.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Enabled Hugging Face offline mode.")

    if args.num_reps <= 0:
        raise ValueError("--num-reps must be > 0")

    experimental_data = pd.read_csv(args.experimental_data)
    boundaries = _resolve_set_boundaries(args)
    models = _resolve_models(args)
    sampler, available_contexts = _build_sampler(args)

    context_col = "story" if "story" in experimental_data.columns else "context"
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

    results_dir = None if args.no_write else args.results_dir
    results = run_experiment(
        experimental_data=experimental_data,
        sampler=sampler,
        set_boundaries=boundaries,
        num_reps=args.num_reps,
        model_specs=models,
        results_dir=results_dir,
        file_suffix=args.file_suffix,
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
