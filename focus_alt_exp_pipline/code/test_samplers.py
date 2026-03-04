"""Quick validation script for samplers.py.

This script prints concrete sample outputs for:
1) ClozeSampler (real cloze data)
2) BertSampler (real model if available, otherwise deterministic mock fallback)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence
from unittest.mock import patch

import pandas as pd

from samplers import BertSampler, ClozeSampler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CLOZE_PATH = DATA_DIR / "inside_the_set" / "word_freq_and_cloze_prob.csv"
PROMPTS_PATH = DATA_DIR / "prompts" / "prompts_BERT.csv"


def _parse_boundaries(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("No set boundaries parsed")
    return values


def _pick_contexts(values: Iterable[str], limit: int) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
        if len(unique) == limit:
            break
    return unique


def _print_samples(
    label: str,
    context_samples: dict[str, list[tuple[list[str], list[str]]]],
    *,
    show_reps: int,
    show_tokens: int,
) -> None:
    print(f"\n[{label}]")
    for context, samples in context_samples.items():
        print(f"  context='{context}' total_reps={len(samples)}")
        for i, (ordering, sampled_set) in enumerate(samples[:show_reps]):
            print(f"    rep={i} ordering[:{show_tokens}]={ordering[:show_tokens]}")
            print(f"    rep={i} sampled_set={sampled_set}")


def run_cloze_demo(args: argparse.Namespace) -> None:
    df_cloze = pd.read_csv(CLOZE_PATH)
    contexts = _pick_contexts(df_cloze["context"], args.contexts)
    sampler = ClozeSampler(df_cloze, seed=args.seed)

    context_samples = sampler.sample_contexts(
        contexts,
        set_boundary=args.set_boundary,
        num_reps=args.num_reps,
    )

    print("=== ClozeSampler Demo ===")
    print(f"data_file={CLOZE_PATH}")
    print(f"contexts={contexts}")
    print(f"set_boundary={args.set_boundary} num_reps={args.num_reps}")
    _print_samples(
        "cloze",
        context_samples,
        show_reps=args.show_reps,
        show_tokens=args.show_tokens,
    )

    for context in contexts:
        subset = df_cloze[df_cloze["context"] == context]
        example_token = str(subset.iloc[0]["word"])
        print(
            f"  supports_token(context='{context}', token='{example_token}') -> "
            f"{sampler.supports_token(context, example_token)}"
        )
    print(
        "  supports_token(context=first_context, token='__not_in_vocab__') -> "
        f"{sampler.supports_token(contexts[0], '__not_in_vocab__')}"
    )


class _FakeBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self["input_ids"] = self.input_ids
        return self


class _FakeTokenizer:
    def __init__(self):
        import torch

        self._torch = torch
        self.vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "bread",
            "cake",
            "water",
            "juice",
            "apple",
        ]
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.mask_token_id = self.token_to_id["[MASK]"]

    def __call__(self, text: str, return_tensors: str = "pt"):
        if return_tensors != "pt":
            raise ValueError("Fake tokenizer only supports return_tensors='pt'")
        tokens = text.replace(".", " ").split()
        ids = []
        for tok in tokens:
            if tok == "[MASK]":
                ids.append(self.mask_token_id)
            else:
                ids.append(self.token_to_id.get(tok.lower(), self.token_to_id["[UNK]"]))
        input_ids = self._torch.tensor([ids], dtype=self._torch.long)
        return _FakeBatch(input_ids)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.vocab[i] if 0 <= i < len(self.vocab) else "[UNK]" for i in ids]


class _FakeModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._device = "cpu"

    def to(self, device: str):
        self._device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        import torch

        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            -8.0,
            dtype=torch.float32,
            device=input_ids.device,
        )

        context_signal = int(input_ids.sum().item()) % 2
        if context_signal == 0:
            logits[:, :, 5] = 6.0  # bread
            logits[:, :, 6] = 5.0  # cake
            logits[:, :, 9] = 2.0  # apple
        else:
            logits[:, :, 7] = 6.0  # water
            logits[:, :, 8] = 5.0  # juice
            logits[:, :, 9] = 2.0  # apple

        class _Output:
            def __init__(self, values):
                self.logits = values

        return _Output(logits)


def _build_mock_bert_sampler(seed: int) -> tuple[BertSampler, pd.DataFrame]:
    prompts_df = pd.DataFrame(
        {
            "story": ["snack_shop", "drink_stand"],
            "prompt": [
                "In the snack shop they only have [MASK].",
                "At the drink stand they only have [MASK].",
            ],
        }
    )

    fake_tokenizer = _FakeTokenizer()
    fake_model = _FakeModel(vocab_size=len(fake_tokenizer.vocab))

    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=fake_tokenizer,
    ), patch(
        "transformers.AutoModelForMaskedLM.from_pretrained",
        return_value=fake_model,
    ):
        sampler = BertSampler(
            prompts_df=prompts_df,
            model_name="mock-bert",
            seed=seed,
            device="cpu",
        )

    return sampler, prompts_df


def run_bert_demo(args: argparse.Namespace) -> None:
    boundaries = _parse_boundaries(args.bert_set_boundaries)

    print("\n=== BertSampler Demo ===")
    print(f"bert_mode={args.bert_mode}")
    print(f"set_boundaries={boundaries} num_reps={args.num_reps}")

    sampler: BertSampler
    prompts_df: pd.DataFrame
    mode_used = args.bert_mode

    if args.bert_mode == "mock":
        sampler, prompts_df = _build_mock_bert_sampler(seed=args.seed)
    else:
        prompts_df = pd.read_csv(PROMPTS_PATH)
        contexts = _pick_contexts(prompts_df["story"], args.contexts)
        prompts_df = prompts_df[prompts_df["story"].isin(contexts)].copy()
        try:
            sampler = BertSampler(
                prompts_df=prompts_df,
                model_name=args.model_name,
                seed=args.seed,
                device=args.device,
            )
        except Exception as exc:
            if args.bert_mode == "real":
                raise
            print("  real BERT load failed, falling back to deterministic mock.")
            print(f"  reason: {type(exc).__name__}: {exc}")
            sampler, prompts_df = _build_mock_bert_sampler(seed=args.seed)
            mode_used = "mock_fallback"

    contexts = _pick_contexts(prompts_df["story"], args.contexts)
    print(f"contexts={contexts}")
    print(f"mode_used={mode_used}")

    sampler.prepare_contexts(contexts)
    cache_size = len(getattr(sampler, "_distribution_cache", {}))
    print(f"distribution_cache_size_after_prepare={cache_size}")

    for set_boundary in boundaries:
        context_samples = sampler.sample_contexts(
            contexts,
            set_boundary=set_boundary,
            num_reps=args.num_reps,
        )
        print(f"\n  boundary={set_boundary}")
        print(f"  distribution_cache_size={len(getattr(sampler, '_distribution_cache', {}))}")
        _print_samples(
            f"bert_{set_boundary}",
            context_samples,
            show_reps=args.show_reps,
            show_tokens=args.show_tokens,
        )

    dist_cache = getattr(sampler, "_distribution_cache", {})
    for context in contexts:
        dist = dist_cache.get(context, [])
        top = sorted(dist, key=lambda x: x[1], reverse=True)[:5]
        print(f"  top5_distribution(context='{context}')={top}")
        if top:
            token = top[0][0]
            print(
                f"  supports_token(context='{context}', token='{token}') -> "
                f"{sampler.supports_token(context, token)}"
            )
    print(
        "  supports_token(context=first_context, token='__not_in_vocab__') -> "
        f"{sampler.supports_token(contexts[0], '__not_in_vocab__')}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ClozeSampler and BertSampler outputs")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--contexts", type=int, default=2, help="Number of contexts to demo")
    parser.add_argument("--num-reps", type=int, default=3, help="Repetitions per context")
    parser.add_argument("--set-boundary", type=int, default=5, help="Set boundary for cloze demo")
    parser.add_argument(
        "--bert-set-boundaries",
        type=str,
        default="3,8",
        help="Comma-separated set boundaries for bert demo",
    )
    parser.add_argument(
        "--bert-mode",
        choices=["auto", "real", "mock"],
        default="auto",
        help="Use real BERT, mock BERT, or auto fallback",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google-bert/bert-large-uncased-whole-word-masking",
        help="HF model name used when bert-mode is real/auto",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for BertSampler (default: auto)",
    )
    parser.add_argument(
        "--show-reps",
        type=int,
        default=2,
        help="How many repetitions to print per context",
    )
    parser.add_argument(
        "--show-tokens",
        type=int,
        default=10,
        help="How many ordering tokens to print",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cloze_demo(args)
    run_bert_demo(args)


if __name__ == "__main__":
    main()
