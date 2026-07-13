"""Resumably score trigger and query continuations in a Qwen manifest.

The output contains one row per manifest row and records the token-level,
summed, and mean continuation log probabilities for both candidates.  A
checkpoint CSV is replaced atomically while scoring so interrupted cluster
jobs can be resumed safely.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .scoring_manifest import (
        MANIFEST_COLUMNS,
        MANIFEST_KEY_COLUMNS,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )
except ImportError:
    from scoring_manifest import (
        MANIFEST_COLUMNS,
        MANIFEST_KEY_COLUMNS,
        summarize_scoring_manifest,
        validate_scoring_manifest,
    )


SCORER_VERSION: Final[str] = "1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "focus_alt_exp_pipeline"
    / "scoring_manifests"
    / "hu_rnx_no_frame_manifest.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "focus_alt_exp_pipeline"
    / "model_scores"
    / "hu_rnx_no_frame_qwen_scores.csv"
)
DEFAULT_MODEL_PATH = REPO_ROOT.parent / "hf-cache" / "models--Qwen--Qwen2-7B"

SCORE_STEMS: Final[tuple[str, ...]] = ("trigger", "query")
SCORE_SUFFIXES: Final[tuple[str, ...]] = (
    "token_ids_json",
    "tokens_json",
    "token_logprobs_json",
    "token_count",
    "logprob_sum",
    "logprob_mean",
    "tokenization_mode",
)
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "scoring_row_key",
    "scorer_version",
    "manifest_sha256",
    "model_identifier",
    "model_revision",
    "model_revision_source",
    "resolved_model_path",
    "tokenizer_name_or_path",
    "tokenizer_class",
    "model_class",
    "requested_dtype",
    "actual_model_dtype",
    "device_map",
    "local_files_only",
    "torch_version",
    "transformers_version",
    "scored_at_utc",
)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *MANIFEST_COLUMNS,
    *(f"{stem}_{suffix}" for stem in SCORE_STEMS for suffix in SCORE_SUFFIXES),
    "query_minus_trigger_logprob_sum",
    "query_minus_trigger_logprob_mean",
    *METADATA_COLUMNS,
)


@dataclass(frozen=True)
class ModelLocation:
    requested: str
    resolved: str
    revision: str | None
    revision_source: str | None


@dataclass
class PromptState:
    past_key_values: object
    next_log_probs: object
    device: object


@dataclass(frozen=True)
class ContinuationScore:
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    tokenization_mode: str

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def logprob_sum(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def logprob_mean(self) -> float:
        if not self.token_logprobs:
            return float("nan")
        return float(self.logprob_sum / self.token_count)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scoring_row_key(row: Mapping[str, object]) -> str:
    return json.dumps(
        [str(row[column]) for column in MANIFEST_KEY_COLUMNS],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def partial_output_path(output: Path) -> Path:
    if output.suffix:
        return output.with_name(f"{output.stem}.partial{output.suffix}")
    return output.with_name(f"{output.name}.partial.csv")


def resolve_model_location(model_path: str | Path) -> ModelLocation:
    requested = str(model_path)
    path = Path(requested).expanduser()
    if not path.exists():
        return ModelLocation(requested, requested, None, None)

    path = path.resolve()
    if path.parent.name == "snapshots":
        return ModelLocation(requested, str(path), path.name, "snapshot_directory")

    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        ref_main = path / "refs" / "main"
        if ref_main.is_file():
            revision = ref_main.read_text(encoding="utf-8").strip()
            candidate = snapshots_dir / revision
            if revision and candidate.is_dir():
                return ModelLocation(
                    requested,
                    str(candidate.resolve()),
                    revision,
                    "huggingface_cache_ref_main",
                )

        snapshots = sorted(child for child in snapshots_dir.iterdir() if child.is_dir())
        if len(snapshots) == 1:
            return ModelLocation(
                requested,
                str(snapshots[0].resolve()),
                snapshots[0].name,
                "single_huggingface_cache_snapshot",
            )
        if len(snapshots) > 1:
            raise ValueError(
                f"Model cache {path} has multiple snapshots and no usable refs/main. "
                "Pass the exact snapshots/<revision> directory with --model-path."
            )

    return ModelLocation(requested, str(path), None, None)


def split_prompt_boundary(prompt: str, candidate: str) -> tuple[str, str]:
    prompt_text = str(prompt)
    candidate_text = str(candidate).strip()
    if not candidate_text:
        raise ValueError("Cannot score an empty continuation")
    match = re.search(r"\s+$", prompt_text)
    if match is None:
        raise ValueError("The scoring prompt must end in whitespace")
    prefix = prompt_text[: match.start()]
    boundary = prompt_text[match.start() :]
    if not prefix:
        raise ValueError("The scoring prompt cannot contain only whitespace")
    return prefix, boundary + candidate_text


def continuation_token_ids(
    tokenizer,
    *,
    prefix: str,
    continuation: str,
) -> tuple[list[int], str]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + continuation, add_special_tokens=False).input_ids
    if not prefix_ids:
        raise ValueError("The tokenized prompt prefix is empty")
    if len(full_ids) < len(prefix_ids) or full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Tokenizer merged across the prompt/continuation boundary; exact continuation "
            "scoring is unavailable for this row."
        )
    token_ids = [int(token_id) for token_id in full_ids[len(prefix_ids) :]]
    if not token_ids:
        raise ValueError("The tokenized continuation is empty")
    return token_ids, "exact_concat"


def _pick_dtype(torch_module, requested: str):
    if requested == "float32":
        return torch_module.float32
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16
    if requested != "auto":
        raise ValueError(f"Unsupported dtype: {requested}")
    if torch_module.cuda.is_available():
        return (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
    return torch_module.float32


def _revision_from_loaded_objects(model, tokenizer) -> tuple[str | None, str | None]:
    candidates = [
        (getattr(model.config, "_commit_hash", None), "model_config_commit_hash"),
        (getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"), "tokenizer_commit_hash"),
    ]
    for revision, source in candidates:
        if revision:
            return str(revision), source
    return None, None


def _revisions_agree(left: str, right: str) -> bool:
    return left == right or left.startswith(right) or right.startswith(left)


def load_model(
    model_path: str,
    *,
    dtype: str,
    device_map: str,
    local_files_only: bool,
):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    location = resolve_model_location(model_path)
    torch_dtype = _pick_dtype(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        location.resolved,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", tokenizer.unk_token)

    signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_argument = "dtype" if "dtype" in signature.parameters else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        location.resolved,
        **{
            dtype_argument: torch_dtype,
            "device_map": device_map,
            "local_files_only": local_files_only,
        },
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    loaded_revision, loaded_source = _revision_from_loaded_objects(model, tokenizer)
    if location.revision and loaded_revision and not _revisions_agree(
        location.revision, loaded_revision
    ):
        raise ValueError(
            "Resolved snapshot revision and loaded model revision disagree: "
            f"{location.revision!r} versus {loaded_revision!r}"
        )
    revision = location.revision or loaded_revision
    revision_source = location.revision_source or loaded_source
    if not revision:
        raise ValueError(
            "Could not determine an exact model revision. Pass a Hugging Face cache root "
            "with refs/main or an exact snapshots/<revision> directory."
        )

    first_parameter = next(model.parameters())
    metadata = {
        "model_identifier": location.requested,
        "model_revision": revision,
        "model_revision_source": revision_source,
        "resolved_model_path": location.resolved,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "requested_dtype": dtype,
        "actual_model_dtype": str(first_parameter.dtype).replace("torch.", ""),
        "device_map": device_map,
        "local_files_only": local_files_only,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }
    return torch, tokenizer, model, metadata


def prepare_prompt_state(torch_module, tokenizer, model, prefix: str) -> PromptState:
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    encoded = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").to(device)
    if encoded.input_ids.shape[1] == 0:
        raise ValueError("Cannot score an empty tokenized prompt")
    with torch_module.no_grad():
        output = model(**encoded, use_cache=True)
    return PromptState(
        past_key_values=output.past_key_values,
        next_log_probs=functional.log_softmax(output.logits[:, -1, :].float(), dim=-1)[0],
        device=device,
    )


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    try:
        return copy.deepcopy(past_key_values)
    except Exception:
        pass
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        cloned = tuple(
            (key_states.clone(), value_states.clone())
            for key_states, value_states in legacy
        )
        restore = getattr(past_key_values.__class__, "from_legacy_cache", None)
        return restore(cloned) if callable(restore) else cloned
    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            (key_states.clone(), value_states.clone())
            for key_states, value_states in past_key_values
        )
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def score_continuation(
    torch_module,
    tokenizer,
    model,
    prompt_state: PromptState,
    *,
    prefix: str,
    continuation: str,
) -> ContinuationScore:
    import torch.nn.functional as functional

    token_ids, tokenization_mode = continuation_token_ids(
        tokenizer,
        prefix=prefix,
        continuation=continuation,
    )
    logprobs: list[float] = []
    next_log_probs = prompt_state.next_log_probs
    past_key_values = clone_past_key_values(prompt_state.past_key_values)
    with torch_module.no_grad():
        for index, token_id in enumerate(token_ids):
            logprobs.append(float(next_log_probs[token_id].item()))
            if index == len(token_ids) - 1:
                break
            next_token = torch_module.tensor(
                [[token_id]],
                dtype=torch_module.long,
                device=prompt_state.device,
            )
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_log_probs = functional.log_softmax(
                output.logits[:, -1, :].float(), dim=-1
            )[0]
    tokens = tuple(str(token) for token in tokenizer.convert_ids_to_tokens(token_ids))
    return ContinuationScore(
        token_ids=tuple(token_ids),
        tokens=tokens,
        token_logprobs=tuple(logprobs),
        tokenization_mode=tokenization_mode,
    )


def _score_fields(stem: str, score: ContinuationScore) -> dict[str, object]:
    return {
        f"{stem}_token_ids_json": json.dumps(score.token_ids),
        f"{stem}_tokens_json": json.dumps(score.tokens, ensure_ascii=False),
        f"{stem}_token_logprobs_json": json.dumps(score.token_logprobs),
        f"{stem}_token_count": score.token_count,
        f"{stem}_logprob_sum": score.logprob_sum,
        f"{stem}_logprob_mean": score.logprob_mean,
        f"{stem}_tokenization_mode": score.tokenization_mode,
    }


def _read_partial(
    path: Path,
    *,
    manifest_hash: str,
    valid_keys: set[str],
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    partial = pd.read_csv(path)
    missing = set(OUTPUT_COLUMNS).difference(partial.columns)
    if missing:
        raise ValueError(f"Partial score file is missing columns: {sorted(missing)}")
    if not partial["manifest_sha256"].astype(str).eq(manifest_hash).all():
        raise ValueError("Partial score file was created from a different manifest")
    if not partial["scorer_version"].astype(str).eq(SCORER_VERSION).all():
        raise ValueError("Partial score file uses a different scorer version")
    keys = partial["scoring_row_key"].astype(str)
    if keys.duplicated().any():
        raise ValueError("Partial score file contains duplicate scoring row keys")
    unexpected = set(keys).difference(valid_keys)
    if unexpected:
        raise ValueError("Partial score file contains rows absent from this manifest")
    return partial.loc[:, OUTPUT_COLUMNS].to_dict(orient="records")


def _validate_resume_metadata(
    completed: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    if not completed:
        return
    for column in [
        "model_revision",
        "requested_dtype",
        "actual_model_dtype",
        "device_map",
        "local_files_only",
    ]:
        existing = {str(row[column]) for row in completed}
        if existing != {str(metadata[column])}:
            raise ValueError(
                f"Cannot resume with changed {column}: partial={sorted(existing)}, "
                f"current={metadata[column]!r}"
            )


def _write_checkpoint(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _candidate_scores_for_prompt(
    rows: pd.DataFrame,
    *,
    torch_module,
    tokenizer,
    model,
) -> dict[str, ContinuationScore]:
    prompt = str(rows.iloc[0]["generation_prompt"])
    candidates = sorted(
        set(rows["trigger"].astype(str)).union(rows["query"].astype(str))
    )
    split = {candidate: split_prompt_boundary(prompt, candidate) for candidate in candidates}
    prefixes = {prefix for prefix, _ in split.values()}
    if len(prefixes) != 1:
        raise ValueError("Candidates for one manifest prompt yielded inconsistent prefixes")
    prefix = prefixes.pop()
    prompt_state = prepare_prompt_state(torch_module, tokenizer, model, prefix)
    return {
        candidate: score_continuation(
            torch_module,
            tokenizer,
            model,
            prompt_state,
            prefix=prefix,
            continuation=continuation,
        )
        for candidate, (_, continuation) in split.items()
    }


def score_manifest(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    validate_scoring_manifest(manifest)
    manifest_hash = sha256_file(args.manifest)
    summary = summarize_scoring_manifest(manifest)
    model_location = resolve_model_location(args.model_path)
    output = args.output.resolve()
    partial = partial_output_path(output)

    print("[manifest]")
    for key, value in asdict(summary).items():
        print(f"  {key}={value}")
    print(f"  sha256={manifest_hash}")
    print("[model]")
    print(f"  requested={model_location.requested}")
    print(f"  resolved={model_location.resolved}")
    print(f"  revision={model_location.revision or 'resolved only after model load'}")
    print(f"[output]\n  final={output}\n  checkpoint={partial}")
    if args.dry_run:
        print("[dry-run complete] Manifest is valid; no model was loaded and no files changed.")
        return

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Final output already exists: {output}")
    if args.overwrite:
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif partial.exists() and not args.resume:
        raise FileExistsError(
            f"Checkpoint exists: {partial}. Pass --resume or choose another --output."
        )

    manifest = manifest.copy()
    manifest["scoring_row_key"] = manifest.apply(scoring_row_key, axis=1)
    valid_keys = set(manifest["scoring_row_key"])
    completed = (
        _read_partial(partial, manifest_hash=manifest_hash, valid_keys=valid_keys)
        if args.resume
        else []
    )
    completed_keys = {str(row["scoring_row_key"]) for row in completed}
    print(f"[resume] completed_rows={len(completed)} remaining_rows={len(manifest)-len(completed)}")

    torch_module, tokenizer, model, model_metadata = load_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=not args.allow_downloads,
    )
    run_metadata = {
        "scorer_version": SCORER_VERSION,
        "manifest_sha256": manifest_hash,
        **model_metadata,
    }
    _validate_resume_metadata(completed, run_metadata)
    print(
        f"[loaded] revision={run_metadata['model_revision']} "
        f"dtype={run_metadata['actual_model_dtype']}"
    )

    rows_since_checkpoint = 0
    remaining = manifest.loc[~manifest["scoring_row_key"].isin(completed_keys)].copy()
    prompt_groups = list(remaining.groupby("generation_prompt", sort=False, dropna=False))
    for group_number, (_, group) in enumerate(prompt_groups, start=1):
        candidate_scores = _candidate_scores_for_prompt(
            group,
            torch_module=torch_module,
            tokenizer=tokenizer,
            model=model,
        )
        scored_at = datetime.now(timezone.utc).isoformat()
        for _, row in group.iterrows():
            trigger_score = candidate_scores[str(row["trigger"])]
            query_score = candidate_scores[str(row["query"])]
            record = {column: row[column] for column in MANIFEST_COLUMNS}
            record.update(_score_fields("trigger", trigger_score))
            record.update(_score_fields("query", query_score))
            record.update(
                {
                    "query_minus_trigger_logprob_sum": (
                        query_score.logprob_sum - trigger_score.logprob_sum
                    ),
                    "query_minus_trigger_logprob_mean": (
                        query_score.logprob_mean - trigger_score.logprob_mean
                    ),
                    **run_metadata,
                    "scoring_row_key": row["scoring_row_key"],
                    "scored_at_utc": scored_at,
                }
            )
            completed.append(record)
            rows_since_checkpoint += 1

        if rows_since_checkpoint >= args.checkpoint_every or group_number == len(prompt_groups):
            _write_checkpoint(completed, partial)
            rows_since_checkpoint = 0
            print(
                f"[checkpoint] rows={len(completed)}/{len(manifest)} "
                f"prompt_groups={group_number}/{len(prompt_groups)}"
            )

    if len(completed) != len(manifest):
        raise RuntimeError(
            f"Scoring stopped with {len(completed)} rows; expected {len(manifest)}"
        )
    os.replace(partial, output)
    print(f"[complete] wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Hu/R&X no-frame trigger and query continuations with Qwen."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Transformers to access/download model files; local-only is the default.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


if __name__ == "__main__":
    score_manifest(parse_args())
