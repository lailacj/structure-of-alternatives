#!/usr/bin/env python3
"""Build a neutral, context-balanced bigram support set for Qwen precompute."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PIPELINE_CODE_DIR = ROOT_DIR / "focus_alt_exp_pipeline" / "code"
if str(PIPELINE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE_DIR))

from data_utils import clean_word, prepare_experimental_data, resolve_context_col  # noqa: E402


DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR.parent / "ngrams" / "qwen_bigram_support"
DEFAULT_UNIGRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_1gram_counts.tsv"
DEFAULT_BACKGROUND_BIGRAM_COUNTS = ROOT_DIR.parent / "ngrams" / "vocab_2gram_counts.tsv"
DEFAULT_NGRAM_DIR = ROOT_DIR.parent / "ngrams" / "downloaded_files"
DEFAULT_UNIGRAM_GLOB = "googlebooks-eng-all-1gram-*.gz"
DEFAULT_BIGRAM_GLOB = "googlebooks-eng-all-2gram-*.gz"

TOKEN_POLICY_REGEX = {
    "letters_only": r"^[A-Za-z]+$",
    "letters_apostrophe_hyphen": r"^[A-Za-z][A-Za-z'-]*$",
}


@dataclass(frozen=True)
class BigramCandidate:
    token: str
    count: int
    conditional_score: float
    first_word: str


def _safe_stem(text: str) -> str:
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _compile_token_regex(policy_name: str, custom_regex: str | None) -> re.Pattern[str]:
    pattern = custom_regex if custom_regex else TOKEN_POLICY_REGEX[policy_name]
    return re.compile(pattern)


def _normalize_experiment_token(
    raw_value: object,
    *,
    token_re: re.Pattern[str],
    strip_indefinite_articles: bool,
) -> tuple[str | None, str | None]:
    token = str(raw_value).strip().lower()
    if not token:
        return None, None
    if strip_indefinite_articles:
        token = clean_word(token)
    parts = token.split()
    if len(parts) == 1 and token_re.fullmatch(parts[0]):
        return parts[0], "1gram"
    if len(parts) == 2 and token_re.fullmatch(parts[0]) and token_re.fullmatch(parts[1]):
        return f"{parts[0]} {parts[1]}", "2gram"
    return None, None


def _normalize_bigram_token(raw_token: str, token_re: re.Pattern[str]) -> str | None:
    parts = raw_token.strip().lower().split()
    if len(parts) != 2:
        return None
    if not token_re.fullmatch(parts[0]) or not token_re.fullmatch(parts[1]):
        return None
    return f"{parts[0]} {parts[1]}"


def _load_unigram_counts(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            token, sep, count_str = raw_line.rstrip("\n").partition("\t")
            if not sep:
                continue
            counts[token] = int(count_str)
    return counts


def _iter_1gram(gz_path: Path, token_re: re.Pattern[str]) -> Iterable[tuple[str, int]]:
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            token = parts[0].lower()
            if not token_re.fullmatch(token):
                continue
            try:
                count = int(parts[2])
            except ValueError:
                continue
            yield token, count


def _count_missing_unigrams_from_raw(
    *,
    missing_tokens: Sequence[str],
    ngram_dir: Path,
    unigram_glob: str,
    token_re: re.Pattern[str],
) -> Dict[str, int]:
    remaining = set(missing_tokens)
    if not remaining:
        return {}

    counts = {token: 0 for token in remaining}
    gz_files = sorted(Path(path) for path in glob(str(ngram_dir / unigram_glob)))
    if not gz_files:
        raise FileNotFoundError(f"No 1gram shards matched: {ngram_dir / unigram_glob}")

    for gz_path in gz_files:
        for token, count in _iter_1gram(gz_path, token_re):
            if token in remaining:
                counts[token] += count
    return counts


def _shard_suffix_for_first_word(word: str) -> str:
    if len(word) == 1:
        return f"{word}_"
    return word[:2]


def _build_shard_index(paths: Sequence[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in paths:
        suffix = path.name.rsplit("-", 1)[-1]
        if not suffix.endswith(".gz"):
            continue
        out[suffix[:-3]] = path
    return out


def _scan_family_bigram_counts(
    *,
    family_first_words: Sequence[str],
    ngram_dir: Path,
    bigram_glob: str,
    token_re: re.Pattern[str],
) -> Dict[str, int]:
    wanted = {word.lower() for word in family_first_words}
    if not wanted:
        return {}

    gz_files = sorted(Path(path) for path in glob(str(ngram_dir / bigram_glob)))
    if not gz_files:
        raise FileNotFoundError(f"No 2gram shards matched: {ngram_dir / bigram_glob}")

    shard_index = _build_shard_index(gz_files)
    shard_targets: Dict[str, set[str]] = defaultdict(set)
    for word in wanted:
        shard_targets[_shard_suffix_for_first_word(word)].add(word)

    missing_shards = sorted(suffix for suffix in shard_targets if suffix not in shard_index)
    if missing_shards:
        raise FileNotFoundError(f"Missing 2gram shard files for suffixes: {missing_shards}")

    counts: Dict[str, int] = defaultdict(int)
    for suffix in sorted(shard_targets):
        wanted_words = shard_targets[suffix]
        with gzip.open(shard_index[suffix], "rt", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                token = parts[0].lower()
                normalized = _normalize_bigram_token(token, token_re)
                if normalized is None:
                    continue
                first_word = normalized.split(" ", 1)[0]
                if first_word not in wanted_words:
                    continue
                try:
                    count = int(parts[2])
                except ValueError:
                    continue
                counts[normalized] += count
    return dict(counts)


def _extract_context_requirements(
    *,
    experimental_data_path: Path,
    token_re: re.Pattern[str],
    strip_indefinite_articles: bool,
    allow_unsupported_human_tokens: bool,
) -> dict[str, dict]:
    prepared = prepare_experimental_data(pd.read_csv(experimental_data_path))
    context_col = resolve_context_col(prepared)

    if strip_indefinite_articles:
        query_col = "cleaned_query" if "cleaned_query" in prepared.columns else "query"
        trigger_col = "cleaned_trigger" if "cleaned_trigger" in prepared.columns else "trigger"
    else:
        query_col = "query" if "query" in prepared.columns else "cleaned_query"
        trigger_col = "trigger" if "trigger" in prepared.columns else "cleaned_trigger"

    context_specs: dict[str, dict] = {}
    unsupported: Dict[str, List[str]] = defaultdict(list)

    for context, subset in prepared.groupby(context_col, sort=False):
        required_tokens: List[str] = []
        required_bigrams: List[str] = []
        family_first_words: List[str] = []
        seen_tokens = set()
        seen_bigrams = set()
        seen_first_words = set()

        values = pd.concat([subset[query_col], subset[trigger_col]], ignore_index=True).dropna()
        for raw_value in values.tolist():
            normalized, kind = _normalize_experiment_token(
                raw_value,
                token_re=token_re,
                strip_indefinite_articles=strip_indefinite_articles,
            )
            if normalized is None:
                unsupported[str(context)].append(str(raw_value))
                continue

            if normalized not in seen_tokens:
                required_tokens.append(normalized)
                seen_tokens.add(normalized)

            first_word = normalized.split(" ", 1)[0]
            if first_word not in seen_first_words:
                family_first_words.append(first_word)
                seen_first_words.add(first_word)

            if kind == "2gram" and normalized not in seen_bigrams:
                required_bigrams.append(normalized)
                seen_bigrams.add(normalized)

        context_specs[str(context)] = {
            "required_tokens": required_tokens,
            "required_bigrams": required_bigrams,
            "family_first_words": family_first_words,
        }

    if unsupported and not allow_unsupported_human_tokens:
        preview_context = sorted(unsupported)[0]
        preview_values = unsupported[preview_context][:10]
        raise ValueError(
            "Unsupported human tokens remain after normalization under the selected token policy. "
            f"First context={preview_context!r}, examples={preview_values}"
        )

    for context, values in unsupported.items():
        context_specs[context]["unsupported_tokens"] = values
    return context_specs


def _build_family_candidates_by_first_word(
    *,
    bigram_counts: Dict[str, int],
    first_word_counts: Dict[str, int],
) -> Dict[str, List[BigramCandidate]]:
    grouped: Dict[str, List[BigramCandidate]] = defaultdict(list)
    for token, count in bigram_counts.items():
        first_word = token.split(" ", 1)[0]
        first_word_count = first_word_counts.get(first_word)
        if first_word_count is None or first_word_count <= 0:
            continue
        grouped[first_word].append(
            BigramCandidate(
                token=token,
                count=int(count),
                conditional_score=float(count) / float(first_word_count),
                first_word=first_word,
            )
        )

    for first_word, candidates in grouped.items():
        candidates.sort(key=lambda item: (-item.conditional_score, -item.count, item.token))
        grouped[first_word] = candidates
    return dict(grouped)


def _build_background_candidates(
    *,
    bigram_counts_path: Path,
    token_re: re.Pattern[str],
    first_word_counts: Dict[str, int],
    limit: int,
) -> List[BigramCandidate]:
    heap: List[tuple[float, int, str]] = []
    with bigram_counts_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            token, sep, count_str = raw_line.rstrip("\n").partition("\t")
            if not sep:
                continue
            normalized = _normalize_bigram_token(token, token_re)
            if normalized is None:
                continue
            first_word = normalized.split(" ", 1)[0]
            first_word_count = first_word_counts.get(first_word)
            if first_word_count is None or first_word_count <= 0:
                continue
            count = int(count_str)
            score = float(count) / float(first_word_count)
            entry = (score, count, normalized)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    candidates = [
        BigramCandidate(
            token=token,
            count=count,
            conditional_score=score,
            first_word=token.split(" ", 1)[0],
        )
        for score, count, token in heap
    ]
    candidates.sort(key=lambda item: (-item.conditional_score, -item.count, item.token))
    return candidates


def _select_context_bigrams(
    *,
    context: str,
    context_spec: dict,
    family_candidates_by_first_word: Dict[str, List[BigramCandidate]],
    background_candidates: Sequence[BigramCandidate],
    bigram_budget: int,
    shared_background_pool_size: int,
) -> List[dict]:
    required_bigrams = list(context_spec["required_bigrams"])
    if len(required_bigrams) > bigram_budget:
        raise ValueError(
            f"Context {context!r} requires {len(required_bigrams)} bigrams, exceeding budget {bigram_budget}."
        )

    selected_rows: List[dict] = []
    seen = set()

    for token in required_bigrams:
        selected_rows.append(
            {
                "token": token,
                "selection_source": "required",
                "count": None,
                "conditional_score": None,
                "background_rank": None,
            }
        )
        seen.add(token)

    family_candidates: Dict[str, BigramCandidate] = {}
    for first_word in context_spec["family_first_words"]:
        for candidate in family_candidates_by_first_word.get(first_word, []):
            existing = family_candidates.get(candidate.token)
            if existing is None:
                family_candidates[candidate.token] = candidate

    ranked_family_candidates = sorted(
        family_candidates.values(),
        key=lambda item: (-item.conditional_score, -item.count, item.token),
    )
    for candidate in ranked_family_candidates:
        if candidate.token in seen:
            continue
        selected_rows.append(
            {
                "token": candidate.token,
                "selection_source": "family",
                "count": candidate.count,
                "conditional_score": candidate.conditional_score,
                "background_rank": None,
            }
        )
        seen.add(candidate.token)
        if len(selected_rows) >= bigram_budget:
            return selected_rows

    for rank, candidate in enumerate(background_candidates, start=1):
        if candidate.token in seen:
            continue
        selected_rows.append(
            {
                "token": candidate.token,
                "selection_source": "background_core" if rank <= shared_background_pool_size else "background_overflow",
                "count": candidate.count,
                "conditional_score": candidate.conditional_score,
                "background_rank": rank,
            }
        )
        seen.add(candidate.token)
        if len(selected_rows) >= bigram_budget:
            return selected_rows

    raise ValueError(
        f"Context {context!r} could not reach bigram budget {bigram_budget}. "
        f"Selected only {len(selected_rows)} items."
    )


def _write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a neutral, context-balanced Qwen bigram support set.")
    parser.add_argument("--experimental-data", type=Path, default=DEFAULT_EXPERIMENTAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unigram-counts", type=Path, default=DEFAULT_UNIGRAM_COUNTS)
    parser.add_argument("--background-bigram-counts", type=Path, default=DEFAULT_BACKGROUND_BIGRAM_COUNTS)
    parser.add_argument("--ngram-dir", type=Path, default=DEFAULT_NGRAM_DIR)
    parser.add_argument("--unigram-glob", type=str, default=DEFAULT_UNIGRAM_GLOB)
    parser.add_argument("--bigram-glob", type=str, default=DEFAULT_BIGRAM_GLOB)
    parser.add_argument("--token-policy", choices=sorted(TOKEN_POLICY_REGEX), default="letters_only")
    parser.add_argument("--custom-token-regex", type=str, default="")
    parser.add_argument("--keep-indefinite-articles", action="store_true")
    parser.add_argument("--allow-unsupported-human-tokens", action="store_true")
    parser.add_argument("--context-bigram-budget", type=int, default=1500)
    parser.add_argument("--shared-background-pool-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.context_bigram_budget <= 0:
        raise ValueError("--context-bigram-budget must be > 0")
    if args.shared_background_pool_size <= 0:
        raise ValueError("--shared-background-pool-size must be > 0")

    token_re = _compile_token_regex(
        args.token_policy,
        args.custom_token_regex.strip() or None,
    )
    strip_indefinite_articles = not args.keep_indefinite_articles

    context_specs = _extract_context_requirements(
        experimental_data_path=args.experimental_data,
        token_re=token_re,
        strip_indefinite_articles=strip_indefinite_articles,
        allow_unsupported_human_tokens=args.allow_unsupported_human_tokens,
    )

    family_first_words = sorted(
        {
            first_word
            for context_spec in context_specs.values()
            for first_word in context_spec["family_first_words"]
        }
    )

    unigram_counts = _load_unigram_counts(args.unigram_counts)
    missing_first_words = [word for word in family_first_words if word not in unigram_counts]
    if missing_first_words:
        unigram_counts.update(
            _count_missing_unigrams_from_raw(
                missing_tokens=missing_first_words,
                ngram_dir=args.ngram_dir,
                unigram_glob=args.unigram_glob,
                token_re=token_re,
            )
        )

    unresolved_first_words = [word for word in family_first_words if unigram_counts.get(word, 0) <= 0]
    if unresolved_first_words:
        raise ValueError(
            "Missing unigram counts for required context family first words: "
            f"{unresolved_first_words}"
        )

    family_bigram_counts = _scan_family_bigram_counts(
        family_first_words=family_first_words,
        ngram_dir=args.ngram_dir,
        bigram_glob=args.bigram_glob,
        token_re=token_re,
    )
    family_candidates_by_first_word = _build_family_candidates_by_first_word(
        bigram_counts=family_bigram_counts,
        first_word_counts=unigram_counts,
    )

    background_candidate_limit = max(args.context_bigram_budget, args.shared_background_pool_size)
    background_candidates = _build_background_candidates(
        bigram_counts_path=args.background_bigram_counts,
        token_re=token_re,
        first_word_counts=unigram_counts,
        limit=background_candidate_limit,
    )

    selected_by_context: Dict[str, List[dict]] = {}
    union_counts: Dict[str, int | None] = {}
    for context, context_spec in context_specs.items():
        selected_rows = _select_context_bigrams(
            context=context,
            context_spec=context_spec,
            family_candidates_by_first_word=family_candidates_by_first_word,
            background_candidates=background_candidates,
            bigram_budget=args.context_bigram_budget,
            shared_background_pool_size=args.shared_background_pool_size,
        )
        selected_by_context[context] = selected_rows
        for row in selected_rows:
            token = row["token"]
            count = row["count"]
            existing = union_counts.get(token)
            if existing is None or count is None:
                union_counts[token] = count

    union_tokens = sorted(union_counts)
    line_no_by_token = {token: index for index, token in enumerate(union_tokens, start=1)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir = args.output_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = args.output_dir / "vocab_2gram.txt"
    vocab_counts_path = args.output_dir / "vocab_2gram_counts.tsv"
    background_path = args.output_dir / "background_bigrams.tsv"
    selection_manifest_path = args.output_dir / "selection_manifest.json"

    with vocab_path.open("w", encoding="utf-8") as handle:
        for token in union_tokens:
            handle.write(token + "\n")

    with vocab_counts_path.open("w", encoding="utf-8") as handle:
        for token in union_tokens:
            count = union_counts[token]
            count_str = "" if count is None else str(count)
            handle.write(f"{token}\t{count_str}\n")

    background_rows = []
    for rank, candidate in enumerate(background_candidates, start=1):
        background_rows.append(
            {
                "rank": rank,
                "token": candidate.token,
                "count": candidate.count,
                "conditional_score": candidate.conditional_score,
                "is_core": int(rank <= args.shared_background_pool_size),
            }
        )
    _write_tsv(
        background_path,
        background_rows,
        ["rank", "token", "count", "conditional_score", "is_core"],
    )

    manifest_contexts: Dict[str, dict] = {}
    for context, context_spec in context_specs.items():
        rows = []
        family_count = 0
        background_count = 0
        for row in selected_by_context[context]:
            selection_source = row["selection_source"]
            if selection_source == "family":
                family_count += 1
            elif selection_source.startswith("background"):
                background_count += 1
            rows.append(
                {
                    "line_no": line_no_by_token[row["token"]],
                    "token": row["token"],
                    "count": "" if row["count"] is None else row["count"],
                    "conditional_score": "" if row["conditional_score"] is None else row["conditional_score"],
                    "selection_source": selection_source,
                    "background_rank": "" if row["background_rank"] is None else row["background_rank"],
                }
            )
        rows.sort(key=lambda item: int(item["line_no"]))

        selection_path = contexts_dir / f"{_safe_stem(context)}.selected_bigrams.tsv"
        _write_tsv(
            selection_path,
            rows,
            ["line_no", "token", "count", "conditional_score", "selection_source", "background_rank"],
        )
        manifest_contexts[context] = {
            "required_tokens": context_spec["required_tokens"],
            "required_bigrams": context_spec["required_bigrams"],
            "family_first_words": context_spec["family_first_words"],
            "selected_bigrams_path": str(selection_path),
            "selected_bigram_count": len(rows),
            "family_bigram_count": family_count,
            "background_bigram_count": background_count,
            "unsupported_tokens": context_spec.get("unsupported_tokens", []),
        }

    manifest = {
        "version": 1,
        "experimental_data_path": str(args.experimental_data),
        "token_policy": {
            "name": args.token_policy,
            "token_regex": token_re.pattern,
            "strip_indefinite_articles": strip_indefinite_articles,
        },
        "selection_strategy": {
            "type": "context_balanced_support",
            "context_bigram_budget": args.context_bigram_budget,
            "shared_background_pool_size": args.shared_background_pool_size,
            "background_usage_rule": "fallback_only",
            "background_candidate_limit": background_candidate_limit,
            "family_source": "all_required_tokens_by_context",
            "scoring": "count(w1 w2) / count(w1)",
        },
        "inputs": {
            "unigram_counts_path": str(args.unigram_counts),
            "background_bigram_counts_path": str(args.background_bigram_counts),
            "ngram_dir": str(args.ngram_dir),
            "unigram_glob": args.unigram_glob,
            "bigram_glob": args.bigram_glob,
        },
        "outputs": {
            "global_vocab_path": str(vocab_path),
            "global_counts_path": str(vocab_counts_path),
            "background_pool_path": str(background_path),
            "contexts_dir": str(contexts_dir),
            "selection_manifest_path": str(selection_manifest_path),
        },
        "contexts": manifest_contexts,
        "global_vocab_size": len(union_tokens),
    }
    with selection_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[contexts] {len(manifest_contexts)}")
    print(f"[family_first_words] {len(family_first_words)}")
    print(f"[background_candidates] {len(background_candidates)}")
    print(f"[global_vocab_size] {len(union_tokens)}")
    print(f"[wrote] {selection_manifest_path}")


if __name__ == "__main__":
    main()
