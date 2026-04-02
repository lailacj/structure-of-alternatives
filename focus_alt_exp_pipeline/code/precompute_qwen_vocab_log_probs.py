"""Precompute Qwen continuation log-probabilities for the focus-alternative pipeline.

By default, this script writes a sparse per-context artifact that is sufficient for the
Qwen focus-alternative model with a chosen top-M support size:
  - all unigrams are scored
  - the exact top remaining bigrams are scored
  - any required experimental bigrams for that context are force-scored

If `--dense-output` is provided, the script falls back to scoring the full 2-gram
vocabulary as well.

Each context still writes one float32 .npy aligned to a shared vocab manifest. Entries
that were not scored remain NaN.

The output is designed to be resumable:
  - <context>.log_probs.npy stores the values
  - <context>.progress.json tracks where scoring has reached
  - vocab_manifest.json stores the shared vocab layout
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import pandas as pd

try:
    from .data_utils import normalize_unique_tokens, prepare_experimental_data, resolve_context_col
except ImportError:
    from data_utils import normalize_unique_tokens, prepare_experimental_data, resolve_context_col


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_CSV = ROOT_DIR / "prompts" / "prompt_files" / "prompts_llm_next_word.csv"
DEFAULT_EXPERIMENTAL_DATA = ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
DEFAULT_VOCAB_1GRAM = ROOT_DIR.parent / "ngrams" / "vocab_1gram.txt"
DEFAULT_VOCAB_2GRAM = ROOT_DIR.parent / "ngrams" / "vocab_2gram.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR.parent / "ngrams" / "qwen_full_vocab_log_probs"
DEFAULT_MODEL_PATH = ROOT_DIR.parent / "hf-cache" / "models--Qwen--Qwen2-7B"
DEFAULT_TARGET_VOCAB_SIZE = 100_000


@dataclass
class PromptState:
    past_key_values: object
    next_log_probs: object
    device: object


def _parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _safe_stem(text: str) -> str:
    keep = []
    for char in str(text):
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists():
        return model_path

    if (path / "tokenizer.json").exists() and (
        (path / "model.safetensors.index.json").exists() or any(path.glob("*.safetensors"))
    ):
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return model_path

    ref_main = path / "refs" / "main"
    if ref_main.exists():
        revision = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / revision
        if candidate.exists():
            return str(candidate)

    snapshot_dirs = sorted(child for child in snapshots_dir.iterdir() if child.is_dir())
    if len(snapshot_dirs) == 1:
        return str(snapshot_dirs[0])

    return model_path


def _iter_vocab(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            word = raw_line.rstrip("\n")
            if not word:
                continue
            yield line_no, word


def _pick_dtype(torch_module, requested: str):
    if requested == "float32":
        return torch_module.float32
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16

    if torch_module.cuda.is_available():
        return torch_module.bfloat16 if torch_module.cuda.is_bf16_supported() else torch_module.float16
    return torch_module.float32


def _load_prompts(
    prompts_csv: Path,
    *,
    context_col: str,
    prompt_col: str,
    contexts: Iterable[str] | None = None,
    max_contexts: int | None = None,
) -> List[tuple[str, str]]:
    df = pd.read_csv(prompts_csv)
    required = {context_col, prompt_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prompts CSV: {sorted(missing)}")

    selected = df[[context_col, prompt_col]].drop_duplicates().copy()
    if contexts:
        keep = {str(value) for value in contexts}
        selected = selected[selected[context_col].astype(str).isin(keep)].copy()

    rows: List[tuple[str, str]] = []
    for _, row in selected.iterrows():
        rows.append((str(row[context_col]), str(row[prompt_col])))
        if max_contexts is not None and len(rows) >= max_contexts:
            break

    if not rows:
        raise ValueError("No contexts remain after prompt filtering.")
    return rows


def _extract_required_bigrams_by_context(experimental_data_path: Path) -> Dict[str, set[str]]:
    experimental_data = pd.read_csv(experimental_data_path)
    prepared = prepare_experimental_data(experimental_data)
    context_col = resolve_context_col(prepared)
    query_col = "cleaned_query" if "cleaned_query" in prepared.columns else "query"
    trigger_col = "cleaned_trigger" if "cleaned_trigger" in prepared.columns else "trigger"

    required_by_context: Dict[str, set[str]] = {}
    for context, subset in prepared.groupby(context_col, sort=False):
        values = pd.concat([subset[query_col], subset[trigger_col]], ignore_index=True).dropna()
        tokens = normalize_unique_tokens(values.tolist())
        required_by_context[str(context)] = {token for token in tokens if " " in token}
    return required_by_context


def _load_or_create_manifest(
    output_dir: Path,
    *,
    vocab_1gram: Path,
    vocab_2gram: Path,
) -> dict:
    manifest_path = output_dir / "vocab_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    count_1gram = _count_lines(vocab_1gram)
    count_2gram = _count_lines(vocab_2gram)
    manifest = {
        "dtype": "float32",
        "sources": [
            {
                "name": "1gram",
                "path": str(vocab_1gram),
                "count": count_1gram,
                "offset": 0,
            },
            {
                "name": "2gram",
                "path": str(vocab_2gram),
                "count": count_2gram,
                "offset": count_1gram,
            },
        ],
        "total_count": count_1gram + count_2gram,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _load_progress(progress_path: Path, source_names: Iterable[str]) -> dict:
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "sources": {
            name: {
                "last_line": 0,
                "done": False,
            }
            for name in source_names
        }
    }


def _save_progress(progress_path: Path, progress: dict) -> None:
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)


def _init_output_array(output_path: Path, total_count: int, overwrite: bool) -> np.memmap:
    if overwrite and output_path.exists():
        output_path.unlink()

    if output_path.exists():
        return np.lib.format.open_memmap(output_path, mode="r+")

    arr = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(total_count,))
    arr[:] = np.nan
    arr.flush()
    return arr


def _write_context_metadata(meta_path: Path, payload: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_model(model_path: str, *, dtype: str, device_map: str, local_files_only: bool):
    import inspect
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_path = _resolve_model_path(model_path)
    torch_dtype = _pick_dtype(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", tokenizer.unk_token)

    from_pretrained_sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_kwarg = "dtype" if "dtype" in from_pretrained_sig.parameters else "torch_dtype"
    model_kwargs = {
        dtype_kwarg: torch_dtype,
        "device_map": device_map,
        "local_files_only": local_files_only,
    }
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        **model_kwargs,
    ).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return torch, tokenizer, model, resolved_model_path


def _prepare_prompt_state(torch_module, tokenizer, model, prompt: str) -> PromptState:
    import torch.nn.functional as F

    device = next(model.parameters()).device
    prompt_text = str(prompt).rstrip() + " "
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(device)
    with torch_module.no_grad():
        output = model(**encoded, use_cache=True)
    next_log_probs = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)[0]
    return PromptState(
        past_key_values=output.past_key_values,
        next_log_probs=next_log_probs,
        device=device,
    )


def _score_continuation_log_prob(
    torch_module,
    tokenizer,
    model,
    prompt_state: PromptState,
    continuation: str,
) -> float:
    import torch.nn.functional as F

    token_ids = tokenizer(" " + str(continuation), add_special_tokens=False).input_ids
    if not token_ids:
        return float("nan")

    total_log_prob = 0.0
    next_log_probs = prompt_state.next_log_probs
    past_key_values = prompt_state.past_key_values

    with torch_module.no_grad():
        for idx, token_id in enumerate(token_ids):
            total_log_prob += float(next_log_probs[token_id].item())
            if idx == len(token_ids) - 1:
                break

            next_token = torch_module.tensor([[token_id]], dtype=torch_module.long, device=prompt_state.device)
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_log_probs = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)[0]

    return total_log_prob


def _process_source(
    *,
    torch_module,
    tokenizer,
    model,
    prompt_state: PromptState,
    source_info: dict,
    output_array: np.memmap,
    progress: dict,
    progress_path: Path,
    flush_every: int,
    limit: int | None,
) -> None:
    source_name = str(source_info["name"])
    source_path = Path(source_info["path"])
    source_offset = int(source_info["offset"])
    source_count = int(source_info["count"])
    source_progress = progress["sources"].setdefault(source_name, {"last_line": 0, "done": False})

    if source_progress.get("done") and (limit is None or source_progress["last_line"] >= limit):
        print(f"[skip] source={source_name} already complete")
        return
    if limit is not None and source_progress["last_line"] >= limit:
        print(f"[skip] source={source_name} already reached requested limit={limit}")
        return

    last_line = int(source_progress.get("last_line", 0))
    processed_since_flush = 0
    processed_total = last_line
    last_seen_line = last_line

    print(
        f"[start] source={source_name} resume_line={last_line} "
        f"target_lines={limit if limit is not None else source_count}"
    )

    for line_no, word in _iter_vocab(source_path):
        if line_no <= last_line:
            continue
        if limit is not None and line_no > limit:
            break

        global_index = source_offset + line_no - 1
        output_array[global_index] = np.float32(
            _score_continuation_log_prob(torch_module, tokenizer, model, prompt_state, word)
        )
        processed_since_flush += 1
        processed_total = line_no
        last_seen_line = line_no

        if processed_since_flush >= flush_every:
            output_array.flush()
            source_progress["last_line"] = line_no
            source_progress["done"] = False
            _save_progress(progress_path, progress)
            print(f"[progress] source={source_name} line={line_no}")
            processed_since_flush = 0

    output_array.flush()
    source_progress["last_line"] = last_seen_line
    source_progress["done"] = last_seen_line >= source_count
    _save_progress(progress_path, progress)
    print(f"[done] source={source_name} line={processed_total} complete={source_progress['done']}")


def _read_unigram_log_probs(source_info: dict, output_array: np.memmap) -> Dict[str, float]:
    source_path = Path(source_info["path"])
    source_offset = int(source_info["offset"])
    source_count = int(source_info["count"])
    values = np.asarray(output_array[source_offset : source_offset + source_count], dtype=np.float32)

    unigram_log_probs: Dict[str, float] = {}
    for line_no, word in _iter_vocab(source_path):
        value = float(values[line_no - 1])
        if np.isfinite(value):
            unigram_log_probs[word] = value
    return unigram_log_probs


def _serialize_top_heap(top_heap: Sequence[tuple[float, int]]) -> List[dict]:
    return [
        {
            "log_prob": float(log_prob),
            "line_no": int(line_no),
        }
        for log_prob, line_no in sorted(top_heap, reverse=True)
    ]


def _deserialize_top_heap(raw_heap: Sequence[dict], max_size: int) -> List[tuple[float, int]]:
    heap: List[tuple[float, int]] = []
    for entry in raw_heap:
        if len(heap) >= max_size:
            break
        heap.append((float(entry["log_prob"]), int(entry["line_no"])))
    heapq.heapify(heap)
    return heap


def _process_sparse_bigram_source(
    *,
    torch_module,
    tokenizer,
    model,
    prompt_state: PromptState,
    source_info: dict,
    output_array: np.memmap,
    progress: dict,
    progress_path: Path,
    flush_every: int,
    limit: int | None,
    unigram_log_probs: Dict[str, float],
    required_bigrams: set[str],
    target_bigram_count: int,
) -> None:
    source_name = str(source_info["name"])
    if source_name != "2gram":
        raise ValueError("_process_sparse_bigram_source only supports the 2gram source")

    source_path = Path(source_info["path"])
    source_offset = int(source_info["offset"])
    source_count = int(source_info["count"])
    source_progress = progress["sources"].setdefault(source_name, {"last_line": 0, "done": False})
    source_progress["selection_mode"] = "sparse_top_support"
    source_progress["target_bigram_count"] = int(target_bigram_count)
    source_progress["target_vocab_size"] = int(target_bigram_count + len(unigram_log_probs))
    source_progress["required_total"] = len(required_bigrams)

    if source_progress.get("done") and (limit is None or int(source_progress.get("last_line", 0)) >= limit):
        print(f"[skip] source={source_name} already complete")
        return
    if limit is not None and int(source_progress.get("last_line", 0)) >= limit:
        print(f"[skip] source={source_name} already reached requested limit={limit}")
        return

    if target_bigram_count <= 0 and not required_bigrams:
        source_progress["last_line"] = min(source_count, limit if limit is not None else source_count)
        source_progress["done"] = limit is None or source_progress["last_line"] >= source_count
        source_progress["top_heap"] = []
        source_progress["required_scored"] = []
        _save_progress(progress_path, progress)
        print(f"[done] source={source_name} no bigram scoring needed")
        return

    last_line = int(source_progress.get("last_line", 0))
    top_heap = _deserialize_top_heap(source_progress.get("top_heap", []), target_bigram_count)
    required_scored = set(str(value) for value in source_progress.get("required_scored", []))
    scored_candidates = int(source_progress.get("scored_candidates", 0))

    processed_since_flush = 0
    processed_total = last_line
    last_seen_line = last_line

    print(
        f"[start] source={source_name} resume_line={last_line} "
        f"target_lines={limit if limit is not None else source_count} "
        f"target_bigram_count={target_bigram_count} required_bigrams={len(required_bigrams)}"
    )

    with source_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if line_no <= last_line:
                continue
            if limit is not None and line_no > limit:
                break

            phrase = raw_line.rstrip("\n")
            if not phrase:
                continue

            last_seen_line = line_no
            processed_total = line_no
            global_index = source_offset + line_no - 1
            first_word = phrase.split(" ", 1)[0]
            upper_bound = unigram_log_probs.get(first_word)
            is_required = phrase in required_bigrams
            current_threshold = top_heap[0][0] if len(top_heap) >= target_bigram_count and top_heap else float("-inf")

            if not is_required and upper_bound is not None and len(top_heap) >= target_bigram_count and upper_bound <= current_threshold:
                continue
            if not is_required and upper_bound is None:
                continue

            log_prob = _score_continuation_log_prob(
                torch_module,
                tokenizer,
                model,
                prompt_state,
                phrase,
            )
            output_array[global_index] = np.float32(log_prob)
            processed_since_flush += 1
            scored_candidates += 1

            if is_required:
                required_scored.add(phrase)

            if target_bigram_count > 0 and np.isfinite(log_prob):
                if len(top_heap) < target_bigram_count:
                    heapq.heappush(top_heap, (float(log_prob), line_no))
                elif float(log_prob) > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (float(log_prob), line_no))

            if processed_since_flush >= flush_every:
                output_array.flush()
                source_progress["last_line"] = line_no
                source_progress["done"] = False
                source_progress["scored_candidates"] = scored_candidates
                source_progress["required_scored"] = sorted(required_scored)
                source_progress["required_scored_count"] = len(required_scored)
                source_progress["top_heap"] = _serialize_top_heap(top_heap)
                _save_progress(progress_path, progress)
                print(
                    f"[progress] source={source_name} line={line_no} "
                    f"scored_candidates={scored_candidates} heap_size={len(top_heap)}"
                )
                processed_since_flush = 0

    output_array.flush()
    source_progress["last_line"] = last_seen_line
    source_progress["done"] = last_seen_line >= source_count and limit is None
    source_progress["scored_candidates"] = scored_candidates
    source_progress["required_scored"] = sorted(required_scored)
    source_progress["required_scored_count"] = len(required_scored)
    source_progress["top_heap"] = _serialize_top_heap(top_heap)
    missing_required = sorted(required_bigrams.difference(required_scored))
    if missing_required:
        source_progress["required_missing"] = missing_required
        print(f"[warn] source={source_name} missing required bigrams: {missing_required}")
    _save_progress(progress_path, progress)
    print(
        f"[done] source={source_name} line={processed_total} complete={source_progress['done']} "
        f"scored_candidates={scored_candidates} heap_size={len(top_heap)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Qwen log-probabilities over the full ngram vocab")
    parser.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    parser.add_argument("--experimental-data", type=Path, default=DEFAULT_EXPERIMENTAL_DATA)
    parser.add_argument("--prompt-context-col", type=str, default="story")
    parser.add_argument("--prompt-col", type=str, default="prompt")
    parser.add_argument("--contexts", type=str, default="")
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--vocab-1gram", type=Path, default=DEFAULT_VOCAB_1GRAM)
    parser.add_argument("--vocab-2gram", type=Path, default=DEFAULT_VOCAB_2GRAM)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=DEFAULT_TARGET_VOCAB_SIZE,
        help=(
            "Total number of finite Qwen entries to preserve per context in sparse mode. "
            "The script keeps all unigrams, the exact top remaining bigrams, and any "
            "required experimental bigrams for that context."
        ),
    )
    parser.add_argument(
        "--dense-output",
        action="store_true",
        help="Score the full 2-gram vocabulary instead of the sparse top-support output.",
    )
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--limit-1gram", type=int, default=None)
    parser.add_argument("--limit-2gram", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    return parser.parse_args()


def _source_by_name(manifest: dict, name: str) -> dict:
    for source in manifest["sources"]:
        if str(source["name"]) == name:
            return source
    raise KeyError(f"Missing source '{name}' in manifest")


def _reset_source_state(
    *,
    output_array: np.memmap,
    source_info: dict,
    progress: dict,
) -> None:
    offset = int(source_info["offset"])
    count = int(source_info["count"])
    output_array[offset : offset + count] = np.nan
    progress["sources"][str(source_info["name"])] = {
        "last_line": 0,
        "done": False,
    }


def _ensure_bigram_mode_compatibility(
    *,
    output_array: np.memmap,
    manifest: dict,
    progress: dict,
    dense_output: bool,
    target_vocab_size: int,
) -> None:
    bigram_source = _source_by_name(manifest, "2gram")
    source_progress = progress["sources"].setdefault("2gram", {"last_line": 0, "done": False})
    existing_mode = source_progress.get("selection_mode")

    if dense_output:
        if existing_mode == "sparse_top_support":
            print("[reset] Clearing sparse 2gram state before dense output run")
            _reset_source_state(output_array=output_array, source_info=bigram_source, progress=progress)
        return

    existing_target = source_progress.get("target_vocab_size")
    if existing_mode != "sparse_top_support":
        if int(source_progress.get("last_line", 0)) > 0 or bool(source_progress.get("done")):
            print("[reset] Clearing legacy/dense 2gram state before sparse output run")
            _reset_source_state(output_array=output_array, source_info=bigram_source, progress=progress)
        return

    if existing_target is None or int(existing_target) != int(target_vocab_size):
        print(
            "[reset] Clearing sparse 2gram state because target_vocab_size changed: "
            f"existing={existing_target} requested={target_vocab_size}"
        )
        _reset_source_state(output_array=output_array, source_info=bigram_source, progress=progress)


def main() -> None:
    args = parse_args()

    if args.flush_every <= 0:
        raise ValueError("--flush-every must be > 0")
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("--max-contexts must be > 0")
    if args.target_vocab_size <= 0:
        raise ValueError("--target-vocab-size must be > 0")

    if args.hf_offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(
        args.output_dir,
        vocab_1gram=args.vocab_1gram,
        vocab_2gram=args.vocab_2gram,
    )
    unigram_source = _source_by_name(manifest, "1gram")
    bigram_source = _source_by_name(manifest, "2gram")
    unigram_count = int(unigram_source["count"])
    target_bigram_count = max(int(args.target_vocab_size) - unigram_count, 0)

    selected_contexts = _load_prompts(
        args.prompts_csv,
        context_col=args.prompt_context_col,
        prompt_col=args.prompt_col,
        contexts=_parse_csv_list(args.contexts),
        max_contexts=args.max_contexts,
    )
    required_bigrams_by_context = _extract_required_bigrams_by_context(args.experimental_data)

    torch_module, tokenizer, model, resolved_model_path = _load_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )

    limits = {
        "1gram": args.limit_1gram,
        "2gram": args.limit_2gram,
    }

    for context, prompt in selected_contexts:
        stem = _safe_stem(context)
        output_path = args.output_dir / f"{stem}.log_probs.npy"
        progress_path = args.output_dir / f"{stem}.progress.json"
        meta_path = args.output_dir / f"{stem}.meta.json"

        if args.overwrite:
            for path in [output_path, progress_path, meta_path]:
                if path.exists():
                    path.unlink()

        print(f"[context] {context}")
        output_array = _init_output_array(output_path, int(manifest["total_count"]), overwrite=False)
        progress = _load_progress(progress_path, [source["name"] for source in manifest["sources"]])
        _ensure_bigram_mode_compatibility(
            output_array=output_array,
            manifest=manifest,
            progress=progress,
            dense_output=args.dense_output,
            target_vocab_size=int(args.target_vocab_size),
        )
        _write_context_metadata(
            meta_path,
            {
                "context": context,
                "prompt": prompt,
                "model_path": resolved_model_path,
                "manifest_path": str(args.output_dir / "vocab_manifest.json"),
                "output_path": str(output_path),
                "selection_mode": "dense_full_vocab" if args.dense_output else "sparse_top_support",
                "target_vocab_size": int(args.target_vocab_size),
                "target_bigram_count": int(target_bigram_count),
                "required_bigrams": sorted(required_bigrams_by_context.get(str(context), set())),
            },
        )

        prompt_state = _prepare_prompt_state(torch_module, tokenizer, model, prompt)
        _process_source(
            torch_module=torch_module,
            tokenizer=tokenizer,
            model=model,
            prompt_state=prompt_state,
            source_info=unigram_source,
            output_array=output_array,
            progress=progress,
            progress_path=progress_path,
            flush_every=args.flush_every,
            limit=limits.get("1gram"),
        )

        if args.dense_output:
            _process_source(
                torch_module=torch_module,
                tokenizer=tokenizer,
                model=model,
                prompt_state=prompt_state,
                source_info=bigram_source,
                output_array=output_array,
                progress=progress,
                progress_path=progress_path,
                flush_every=args.flush_every,
                limit=limits.get("2gram"),
            )
        else:
            unigram_log_probs = _read_unigram_log_probs(unigram_source, output_array)
            _process_sparse_bigram_source(
                torch_module=torch_module,
                tokenizer=tokenizer,
                model=model,
                prompt_state=prompt_state,
                source_info=bigram_source,
                output_array=output_array,
                progress=progress,
                progress_path=progress_path,
                flush_every=args.flush_every,
                limit=limits.get("2gram"),
                unigram_log_probs=unigram_log_probs,
                required_bigrams=required_bigrams_by_context.get(str(context), set()),
                target_bigram_count=target_bigram_count,
            )

        del output_array
        del prompt_state
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    print("[complete] all requested contexts processed")


if __name__ == "__main__":
    main()
