"""Precompute Qwen continuation log-probabilities over a context-balanced vocab."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_CSV = ROOT_DIR / "prompts" / "prompt_files" / "prompts_llm_next_word.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR.parent / "ngrams" / "qwen_context_balanced_log_probs"
DEFAULT_VOCAB_1GRAM = ROOT_DIR.parent / "ngrams" / "vocab_1gram.txt"
DEFAULT_BIGRAM_SUPPORT_DIR = ROOT_DIR.parent / "ngrams" / "qwen_bigram_support"
DEFAULT_BIGRAM_SUPPORT_MANIFEST = DEFAULT_BIGRAM_SUPPORT_DIR / "selection_manifest.json"
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


def _load_bigram_support_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing bigram support manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    outputs = manifest.get("outputs", {})
    contexts = manifest.get("contexts", {})
    global_vocab_path = outputs.get("global_vocab_path")
    if not global_vocab_path:
        raise ValueError(f"Bigram support manifest is missing outputs.global_vocab_path: {path}")
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError(f"Bigram support manifest is missing context specs: {path}")
    return manifest


def _load_or_create_vocab_manifest(
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


def _source_by_name(manifest: dict, name: str) -> dict:
    for source in manifest["sources"]:
        if str(source["name"]) == name:
            return source
    raise KeyError(f"Missing source '{name}' in manifest")


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
        arr = np.lib.format.open_memmap(output_path, mode="r+")
        if tuple(arr.shape) != (total_count,):
            raise ValueError(
                f"Existing output array has shape {arr.shape}, expected {(total_count,)}. "
                "Use --overwrite or a fresh output directory."
            )
        return arr

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


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None

    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        pass

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        cloned_legacy_cache = tuple(
            (key_states.clone(), value_states.clone())
            for key_states, value_states in legacy_cache
        )
        from_legacy_cache = getattr(past_key_values.__class__, "from_legacy_cache", None)
        if callable(from_legacy_cache):
            return from_legacy_cache(cloned_legacy_cache)
        return cloned_legacy_cache

    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            (key_states.clone(), value_states.clone())
            for key_states, value_states in past_key_values
        )

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


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
    if len(token_ids) == 1:
        return float(next_log_probs[token_ids[0]].item())

    past_key_values = _clone_past_key_values(prompt_state.past_key_values)

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


def _reset_source_state(*, output_array: np.memmap, source_info: dict, progress: dict) -> None:
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
    selected_bigrams_path: Path,
    selection_manifest_path: Path,
    target_vocab_size: int,
) -> None:
    bigram_source = _source_by_name(manifest, "2gram")
    source_progress = progress["sources"].setdefault("2gram", {"last_line": 0, "done": False})
    existing_mode = source_progress.get("selection_mode")
    existing_target = source_progress.get("target_vocab_size")
    existing_selection_path = source_progress.get("selected_bigrams_path")
    existing_manifest_path = source_progress.get("selection_manifest_path")

    if existing_mode != "context_balanced_support":
        if int(source_progress.get("last_line", 0)) > 0 or bool(source_progress.get("done")):
            print("[reset] Clearing legacy/dense 2gram state before context-balanced run")
            _reset_source_state(output_array=output_array, source_info=bigram_source, progress=progress)
        return

    if (
        existing_selection_path != str(selected_bigrams_path)
        or existing_manifest_path != str(selection_manifest_path)
        or existing_target is None
        or int(existing_target) != int(target_vocab_size)
    ):
        print("[reset] Clearing 2gram state because the selected support changed")
        _reset_source_state(output_array=output_array, source_info=bigram_source, progress=progress)


def _process_unigram_source(
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
    source_progress["done"] = last_seen_line >= source_count and (limit is None or last_seen_line >= limit)
    _save_progress(progress_path, progress)
    print(f"[done] source={source_name} line={processed_total} complete={source_progress['done']}")


def _load_context_bigram_entries(context_info: dict) -> List[dict]:
    selected_path = Path(str(context_info["selected_bigrams_path"]))
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected bigram file: {selected_path}")

    entries: List[dict] = []
    with selected_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entries.append(
                {
                    "line_no": int(row["line_no"]),
                    "token": row["token"],
                    "selection_source": row["selection_source"],
                }
            )
    entries.sort(key=lambda item: item["line_no"])
    return entries


def _process_selected_bigram_source(
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
    selected_entries: Sequence[dict],
    required_bigrams: Sequence[str],
    unigram_count: int,
    selected_bigrams_path: Path,
    selection_manifest_path: Path,
) -> None:
    source_name = str(source_info["name"])
    source_count = int(source_info["count"])
    source_offset = int(source_info["offset"])
    source_progress = progress["sources"].setdefault(source_name, {"last_line": 0, "done": False})
    target_bigram_count = len(selected_entries)
    target_vocab_size = int(unigram_count + target_bigram_count)
    required_bigram_set = {str(token) for token in required_bigrams}

    source_progress["selection_mode"] = "context_balanced_support"
    source_progress["target_bigram_count"] = int(target_bigram_count)
    source_progress["target_vocab_size"] = target_vocab_size
    source_progress["required_total"] = len(required_bigram_set)
    source_progress["selected_bigrams_path"] = str(selected_bigrams_path)
    source_progress["selection_manifest_path"] = str(selection_manifest_path)

    if source_progress.get("done") and (limit is None or source_progress.get("scored_candidates", 0) >= limit):
        print(f"[skip] source={source_name} already complete")
        return

    last_line = int(source_progress.get("last_line", 0))
    required_scored = set(str(value) for value in source_progress.get("required_scored", []))
    scored_candidates = int(source_progress.get("scored_candidates", 0))
    processed_since_flush = 0
    last_seen_line = last_line

    print(
        f"[start] source={source_name} resume_line={last_line} "
        f"selected_bigrams={target_bigram_count} required_bigrams={len(required_bigram_set)}"
    )

    processed_entries = 0
    for entry in selected_entries:
        line_no = int(entry["line_no"])
        token = str(entry["token"])
        if line_no <= last_line:
            processed_entries += 1
            continue
        if line_no > source_count:
            raise ValueError(f"Selected bigram line {line_no} exceeds source size {source_count}")
        if limit is not None and processed_entries >= limit:
            break

        global_index = source_offset + line_no - 1
        output_array[global_index] = np.float32(
            _score_continuation_log_prob(torch_module, tokenizer, model, prompt_state, token)
        )
        if token in required_bigram_set:
            required_scored.add(token)

        scored_candidates += 1
        processed_since_flush += 1
        processed_entries += 1
        last_seen_line = line_no

        if processed_since_flush >= flush_every:
            output_array.flush()
            source_progress["last_line"] = line_no
            source_progress["done"] = False
            source_progress["scored_candidates"] = scored_candidates
            source_progress["required_scored"] = sorted(required_scored)
            source_progress["required_scored_count"] = len(required_scored)
            _save_progress(progress_path, progress)
            print(
                f"[progress] source={source_name} line={line_no} "
                f"scored_candidates={scored_candidates}"
            )
            processed_since_flush = 0

    output_array.flush()
    source_progress["last_line"] = last_seen_line
    source_progress["done"] = processed_entries >= len(selected_entries) and limit is None
    source_progress["scored_candidates"] = scored_candidates
    source_progress["required_scored"] = sorted(required_scored)
    source_progress["required_scored_count"] = len(required_scored)
    missing_required = sorted(required_bigram_set.difference(required_scored))
    if missing_required:
        source_progress["required_missing"] = missing_required
        print(f"[warn] source={source_name} missing required bigrams: {missing_required}")
    else:
        source_progress.pop("required_missing", None)
    _save_progress(progress_path, progress)
    print(
        f"[done] source={source_name} line={last_seen_line} complete={source_progress['done']} "
        f"scored_candidates={scored_candidates}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Qwen log-probabilities over a context-balanced vocab")
    parser.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    parser.add_argument("--prompt-context-col", type=str, default="story")
    parser.add_argument("--prompt-col", type=str, default="prompt")
    parser.add_argument("--contexts", type=str, default="")
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--vocab-1gram", type=Path, default=DEFAULT_VOCAB_1GRAM)
    parser.add_argument("--bigram-support-manifest", type=Path, default=DEFAULT_BIGRAM_SUPPORT_MANIFEST)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-vocab-size", type=int, default=DEFAULT_TARGET_VOCAB_SIZE)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--limit-1gram", type=int, default=None)
    parser.add_argument("--limit-2gram", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    return parser.parse_args()


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

    bigram_support_manifest = _load_bigram_support_manifest(args.bigram_support_manifest)
    vocab_2gram = Path(str(bigram_support_manifest["outputs"]["global_vocab_path"]))
    if not vocab_2gram.exists():
        raise FileNotFoundError(f"Missing global bigram support vocab: {vocab_2gram}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_manifest = _load_or_create_vocab_manifest(
        args.output_dir,
        vocab_1gram=args.vocab_1gram,
        vocab_2gram=vocab_2gram,
    )
    unigram_source = _source_by_name(vocab_manifest, "1gram")
    bigram_source = _source_by_name(vocab_manifest, "2gram")
    unigram_count = int(unigram_source["count"])

    selected_contexts = _load_prompts(
        args.prompts_csv,
        context_col=args.prompt_context_col,
        prompt_col=args.prompt_col,
        contexts=_parse_csv_list(args.contexts),
        max_contexts=args.max_contexts,
    )

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

        if context not in bigram_support_manifest["contexts"]:
            raise KeyError(f"Context {context!r} is missing from bigram support manifest {args.bigram_support_manifest}")

        context_info = bigram_support_manifest["contexts"][context]
        selected_entries = _load_context_bigram_entries(context_info)
        actual_target_vocab_size = unigram_count + len(selected_entries)
        if actual_target_vocab_size < args.target_vocab_size:
            raise ValueError(
                f"Context {context!r} only has {actual_target_vocab_size} total supported tokens, "
                f"below requested target {args.target_vocab_size}."
            )
        selected_bigrams_path = Path(str(context_info["selected_bigrams_path"]))

        if args.overwrite:
            for path in [output_path, progress_path, meta_path]:
                if path.exists():
                    path.unlink()

        print(f"[context] {context}")
        output_array = _init_output_array(output_path, int(vocab_manifest["total_count"]), overwrite=False)
        progress = _load_progress(progress_path, [source["name"] for source in vocab_manifest["sources"]])
        _ensure_bigram_mode_compatibility(
            output_array=output_array,
            manifest=vocab_manifest,
            progress=progress,
            selected_bigrams_path=selected_bigrams_path,
            selection_manifest_path=args.bigram_support_manifest,
            target_vocab_size=actual_target_vocab_size,
        )
        _write_context_metadata(
            meta_path,
            {
                "context": context,
                "prompt": prompt,
                "model_path": resolved_model_path,
                "manifest_path": str(args.output_dir / "vocab_manifest.json"),
                "output_path": str(output_path),
                "selection_mode": "context_balanced_support",
                "target_vocab_size": int(actual_target_vocab_size),
                "requested_target_vocab_size": int(args.target_vocab_size),
                "target_bigram_count": len(selected_entries),
                "required_bigrams": context_info["required_bigrams"],
                "selected_bigrams_path": str(selected_bigrams_path),
                "selection_manifest_path": str(args.bigram_support_manifest),
            },
        )

        prompt_state = _prepare_prompt_state(torch_module, tokenizer, model, prompt)
        _process_unigram_source(
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
        _process_selected_bigram_source(
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
            selected_entries=selected_entries,
            required_bigrams=context_info["required_bigrams"],
            unigram_count=unigram_count,
            selected_bigrams_path=selected_bigrams_path,
            selection_manifest_path=args.bigram_support_manifest,
        )

        del output_array
        del prompt_state
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    print("[complete] all requested contexts processed")


if __name__ == "__main__":
    main()
