"""Score Qwen trigger/query prompt correlations.

This script compares Qwen continuation log probabilities for each unique
`(story, trigger, query)` pair under:

    1. the base next-word prompt
    2. the same prompt after appending "{trigger} but not"

It also aggregates trigger-conditioned scores to unique `(story, query)` rows.
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - scipy is listed in requirements.txt.
    scipy_stats = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_CSV = ROOT_DIR / "prompts" / "prompt_files" / "prompts_llm_next_word.csv"
DEFAULT_EXPERIMENTAL_DATA = (
    ROOT_DIR / "focus_alt_exp_pipeline" / "human_exp_data" / "sca_dataframe.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "trigger_analysis" / "results"
DEFAULT_MODEL_PATH = ROOT_DIR.parent / "hf-cache" / "models--Qwen--Qwen2-7B"


@dataclass
class PromptState:
    past_key_values: object
    next_log_probs: object
    device: object


@dataclass
class ContinuationScore:
    logprob_sum: float
    token_count: int
    tokenization_mode: str

    @property
    def logprob_mean(self) -> float:
        if self.token_count <= 0 or not np.isfinite(self.logprob_sum):
            return float("nan")
        return float(self.logprob_sum / self.token_count)


def clean_word(word: object) -> str:
    token = str(word).strip().lower()
    if token.startswith("a "):
        return token[2:]
    if token.startswith("an "):
        return token[3:]
    return token


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


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


def _pick_dtype(torch_module, requested: str):
    if requested == "float32":
        return torch_module.float32
    if requested == "float16":
        return torch_module.float16
    if requested == "bfloat16":
        return torch_module.bfloat16

    if torch_module.cuda.is_available():
        return (
            torch_module.bfloat16
            if torch_module.cuda.is_bf16_supported()
            else torch_module.float16
        )
    return torch_module.float32


def _load_model(
    model_path: str,
    *,
    dtype: str,
    device_map: str,
    local_files_only: bool,
):
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


def _prepare_prompt_state(torch_module, tokenizer, model, prefix: str) -> PromptState:
    import torch.nn.functional as F

    device = next(model.parameters()).device
    prefix_text = str(prefix).rstrip()
    encoded = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt").to(device)
    if encoded.input_ids.shape[1] == 0:
        raise ValueError("Cannot score an empty prompt prefix.")

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


def _continuation_token_ids(
    tokenizer,
    *,
    prefix: str,
    continuation: str,
) -> tuple[list[int], str]:
    prefix_text = str(prefix).rstrip()
    continuation_text = " " + str(continuation).strip()

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix_text + continuation_text, add_special_tokens=False).input_ids
    if len(full_ids) >= len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :], "exact_concat"

    # This fallback keeps the run moving if tokenizer boundary merging means the
    # concatenated string cannot be cleanly split at the prompt boundary.
    fallback_ids = tokenizer(continuation_text, add_special_tokens=False).input_ids
    return fallback_ids, "separate_continuation_fallback"


def _score_continuation_logprob(
    torch_module,
    model,
    prompt_state: PromptState,
    token_ids: Sequence[int],
) -> float:
    import torch.nn.functional as F

    if not token_ids:
        return float("nan")

    next_log_probs = prompt_state.next_log_probs
    if len(token_ids) == 1:
        return float(next_log_probs[int(token_ids[0])].item())

    total_logprob = 0.0
    past_key_values = _clone_past_key_values(prompt_state.past_key_values)

    with torch_module.no_grad():
        for idx, token_id in enumerate(token_ids):
            token_id = int(token_id)
            total_logprob += float(next_log_probs[token_id].item())
            if idx == len(token_ids) - 1:
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
            next_log_probs = F.log_softmax(output.logits[:, -1, :].float(), dim=-1)[0]

    return float(total_logprob)


def score_continuation(
    torch_module,
    tokenizer,
    model,
    prompt_state: PromptState,
    *,
    prefix: str,
    continuation: str,
) -> ContinuationScore:
    token_ids, tokenization_mode = _continuation_token_ids(
        tokenizer,
        prefix=prefix,
        continuation=continuation,
    )
    logprob_sum = _score_continuation_logprob(
        torch_module,
        model,
        prompt_state,
        token_ids,
    )
    return ContinuationScore(
        logprob_sum=logprob_sum,
        token_count=len(token_ids),
        tokenization_mode=tokenization_mode,
    )


def _load_prompt_map(prompts_csv: Path) -> dict[str, str]:
    prompts = pd.read_csv(prompts_csv)
    required = {"story", "prompt"}
    missing = required.difference(prompts.columns)
    if missing:
        raise ValueError(f"Missing columns in prompts CSV: {sorted(missing)}")

    prompts = prompts[["story", "prompt"]].drop_duplicates().copy()
    prompts["story"] = prompts["story"].astype(str).str.strip()
    if prompts["story"].duplicated().any():
        duplicated = sorted(prompts.loc[prompts["story"].duplicated(), "story"].unique())
        raise ValueError(f"Duplicate stories in prompts CSV: {duplicated}")

    return dict(zip(prompts["story"], prompts["prompt"]))


def _load_unique_pairs(
    experimental_data: Path,
    *,
    string_mode: str,
    contexts: Iterable[str] | None,
) -> pd.DataFrame:
    data = pd.read_csv(experimental_data)
    if "story" not in data.columns:
        raise ValueError("Experimental data must include a 'story' column.")

    if string_mode == "cleaned":
        trigger_col = "cleaned_trigger" if "cleaned_trigger" in data.columns else "trigger"
        query_col = "cleaned_query" if "cleaned_query" in data.columns else "query"
    elif string_mode == "raw":
        trigger_col = "trigger"
        query_col = "query"
    else:
        raise ValueError(f"Unsupported string mode: {string_mode}")

    required = {"story", trigger_col, query_col}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns in experimental data: {sorted(missing)}")

    pairs = data[["story", trigger_col, query_col]].dropna().copy()
    pairs = pairs.rename(columns={trigger_col: "trigger", query_col: "query"})
    pairs["story"] = pairs["story"].astype(str).str.strip()

    if string_mode == "cleaned":
        pairs["trigger"] = pairs["trigger"].apply(clean_word)
        pairs["query"] = pairs["query"].apply(clean_word)
    else:
        pairs["trigger"] = pairs["trigger"].astype(str).str.strip()
        pairs["query"] = pairs["query"].astype(str).str.strip()

    pairs = pairs[(pairs["trigger"] != "") & (pairs["query"] != "")].copy()

    context_set = set(contexts or [])
    if context_set:
        pairs = pairs[pairs["story"].isin(context_set)].copy()

    pairs = pairs.drop_duplicates(ignore_index=True)
    return pairs.sort_values(["story", "trigger", "query"], kind="stable").reset_index(drop=True)


def _attach_prompt_prefixes(pairs: pd.DataFrame, prompt_map: dict[str, str]) -> pd.DataFrame:
    missing = sorted(set(pairs["story"]).difference(prompt_map))
    if missing:
        raise ValueError(f"Missing prompt rows for stories: {missing}")

    prepared = pairs.copy()
    prepared["base_prefix"] = prepared["story"].map(prompt_map)
    prepared["trigger_prefix"] = prepared.apply(
        lambda row: f"{str(row['base_prefix']).rstrip()} {row['trigger']} but not",
        axis=1,
    )
    return prepared


def _clear_prompt_state(torch_module) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _score_pairs(
    pairs: pd.DataFrame,
    *,
    torch_module,
    tokenizer,
    model,
) -> pd.DataFrame:
    base_scores: dict[tuple[str, str], ContinuationScore] = {}
    trigger_scores: dict[tuple[str, str, str], ContinuationScore] = {}

    for story, subset in pairs.groupby("story", sort=False):
        prefix = str(subset["base_prefix"].iloc[0])
        queries = list(dict.fromkeys(subset["query"].tolist()))
        print(f"[base] {story}: scoring {len(queries)} unique queries")
        prompt_state = _prepare_prompt_state(torch_module, tokenizer, model, prefix)
        for query in queries:
            base_scores[(str(story), str(query))] = score_continuation(
                torch_module,
                tokenizer,
                model,
                prompt_state,
                prefix=prefix,
                continuation=str(query),
            )
        del prompt_state
        _clear_prompt_state(torch_module)

    for (story, trigger), subset in pairs.groupby(["story", "trigger"], sort=False):
        prefix = str(subset["trigger_prefix"].iloc[0])
        queries = list(dict.fromkeys(subset["query"].tolist()))
        print(f"[trigger] {story} / {trigger}: scoring {len(queries)} queries")
        prompt_state = _prepare_prompt_state(torch_module, tokenizer, model, prefix)
        for query in queries:
            trigger_scores[(str(story), str(trigger), str(query))] = score_continuation(
                torch_module,
                tokenizer,
                model,
                prompt_state,
                prefix=prefix,
                continuation=str(query),
            )
        del prompt_state
        _clear_prompt_state(torch_module)

    records = []
    for _, row in pairs.iterrows():
        story = str(row["story"])
        trigger = str(row["trigger"])
        query = str(row["query"])
        base = base_scores[(story, query)]
        but_not = trigger_scores[(story, trigger, query)]
        records.append(
            {
                "story": story,
                "trigger": trigger,
                "query": query,
                "base_prefix": row["base_prefix"],
                "trigger_prefix": row["trigger_prefix"],
                "base_logprob_sum": base.logprob_sum,
                "base_logprob_mean": base.logprob_mean,
                "base_query_token_count": base.token_count,
                "base_tokenization_mode": base.tokenization_mode,
                "but_not_logprob_sum": but_not.logprob_sum,
                "but_not_logprob_mean": but_not.logprob_mean,
                "but_not_query_token_count": but_not.token_count,
                "but_not_tokenization_mode": but_not.tokenization_mode,
                "delta_logprob_sum": but_not.logprob_sum - base.logprob_sum,
                "delta_logprob_mean": but_not.logprob_mean - base.logprob_mean,
            }
        )

    return pd.DataFrame.from_records(records)


def _build_query_level_scores(pair_scores: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (story, query), subset in pair_scores.groupby(["story", "query"], sort=False):
        base_sum = float(subset["base_logprob_sum"].iloc[0])
        base_mean = float(subset["base_logprob_mean"].iloc[0])
        but_not_sum = subset["but_not_logprob_sum"].astype(float)
        but_not_mean = subset["but_not_logprob_mean"].astype(float)

        records.append(
            {
                "story": story,
                "query": query,
                "n_triggers": int(subset["trigger"].nunique()),
                "triggers": ",".join(sorted(subset["trigger"].astype(str).unique())),
                "base_logprob_sum": base_sum,
                "base_logprob_mean": base_mean,
                "base_query_token_count": int(subset["base_query_token_count"].iloc[0]),
                "mean_but_not_logprob_sum": float(but_not_sum.mean()),
                "median_but_not_logprob_sum": float(but_not_sum.median()),
                "sd_but_not_logprob_sum": float(but_not_sum.std(ddof=1)),
                "min_but_not_logprob_sum": float(but_not_sum.min()),
                "max_but_not_logprob_sum": float(but_not_sum.max()),
                "mean_but_not_logprob_mean": float(but_not_mean.mean()),
                "median_but_not_logprob_mean": float(but_not_mean.median()),
                "sd_but_not_logprob_mean": float(but_not_mean.std(ddof=1)),
                "min_but_not_logprob_mean": float(but_not_mean.min()),
                "max_but_not_logprob_mean": float(but_not_mean.max()),
                "mean_delta_logprob_sum": float(but_not_sum.mean() - base_sum),
                "mean_delta_logprob_mean": float(but_not_mean.mean() - base_mean),
            }
        )

    return pd.DataFrame.from_records(records)


def _finite_xy(frame: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    xy = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return xy.astype(float)


def _correlation(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    method: str,
) -> tuple[float, float, int, int, int, str]:
    xy = _finite_xy(frame, x_col, y_col)
    n = len(xy)
    x_unique = int(xy[x_col].nunique()) if n else 0
    y_unique = int(xy[y_col].nunique()) if n else 0

    if n < 2:
        return float("nan"), float("nan"), n, x_unique, y_unique, "fewer_than_two_points"
    if x_unique < 2:
        return float("nan"), float("nan"), n, x_unique, y_unique, "constant_x"
    if y_unique < 2:
        return float("nan"), float("nan"), n, x_unique, y_unique, "constant_y"

    if scipy_stats is None:
        value = float(xy[x_col].corr(xy[y_col], method=method))
        return value, float("nan"), n, x_unique, y_unique, ""

    if method == "pearson":
        result = scipy_stats.pearsonr(xy[x_col], xy[y_col])
        return float(result.statistic), float(result.pvalue), n, x_unique, y_unique, ""
    if method == "spearman":
        result = scipy_stats.spearmanr(xy[x_col], xy[y_col])
        return float(result.statistic), float(result.pvalue), n, x_unique, y_unique, ""
    raise ValueError(f"Unsupported correlation method: {method}")


def _correlation_records(
    frame: pd.DataFrame,
    *,
    analysis_level: str,
    context: str,
    x_col: str,
    y_col: str,
    score_scale: str,
) -> list[dict[str, object]]:
    records = []
    for method in ["pearson", "spearman"]:
        value, p_value, n, x_unique, y_unique, reason = _correlation(
            frame,
            x_col=x_col,
            y_col=y_col,
            method=method,
        )
        records.append(
            {
                "analysis_level": analysis_level,
                "context": context,
                "score_scale": score_scale,
                "x_col": x_col,
                "y_col": y_col,
                "method": method,
                "n": n,
                "x_unique": x_unique,
                "y_unique": y_unique,
                "correlation": value,
                "p_value": p_value,
                "undefined_reason": reason,
            }
        )
    return records


def _build_pair_correlations(pair_scores: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("sum_logprob", "base_logprob_sum", "but_not_logprob_sum"),
        ("mean_logprob_per_token", "base_logprob_mean", "but_not_logprob_mean"),
    ]
    records: list[dict[str, object]] = []
    for score_scale, x_col, y_col in specs:
        records.extend(
            _correlation_records(
                pair_scores,
                analysis_level="pair",
                context="ALL",
                x_col=x_col,
                y_col=y_col,
                score_scale=score_scale,
            )
        )
        for story, subset in pair_scores.groupby("story", sort=False):
            records.extend(
                _correlation_records(
                    subset,
                    analysis_level="pair",
                    context=str(story),
                    x_col=x_col,
                    y_col=y_col,
                    score_scale=score_scale,
                )
            )
    return pd.DataFrame.from_records(records)


def _build_query_correlations(query_scores: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("sum_logprob", "base_logprob_sum", "mean_but_not_logprob_sum"),
        ("mean_logprob_per_token", "base_logprob_mean", "mean_but_not_logprob_mean"),
    ]
    records: list[dict[str, object]] = []
    for score_scale, x_col, y_col in specs:
        records.extend(
            _correlation_records(
                query_scores,
                analysis_level="unique_query",
                context="ALL",
                x_col=x_col,
                y_col=y_col,
                score_scale=score_scale,
            )
        )
        for story, subset in query_scores.groupby("story", sort=False):
            records.extend(
                _correlation_records(
                    subset,
                    analysis_level="unique_query",
                    context=str(story),
                    x_col=x_col,
                    y_col=y_col,
                    score_scale=score_scale,
                )
            )
    return pd.DataFrame.from_records(records)


def _print_dry_run_summary(pairs: pd.DataFrame) -> None:
    print(f"unique_story_trigger_query_pairs={len(pairs)}")
    print(f"unique_story_query_rows={pairs[['story', 'query']].drop_duplicates().shape[0]}")
    print(f"contexts={pairs['story'].nunique()}")
    print(f"base_prompt_states={pairs['story'].nunique()}")
    print(f"trigger_prompt_states={pairs[['story', 'trigger']].drop_duplicates().shape[0]}")
    trigger_counts = pairs.groupby(["story", "query"])["trigger"].nunique()
    print(f"triggers_per_story_query_min={int(trigger_counts.min())}")
    print(f"triggers_per_story_query_max={int(trigger_counts.max())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Qwen base-vs-trigger query logprob correlations."
    )
    parser.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    parser.add_argument("--experimental-data", type=Path, default=DEFAULT_EXPERIMENTAL_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--string-mode",
        choices=["cleaned", "raw"],
        default="cleaned",
        help="Use cleaned query/trigger strings by default, matching the active pipeline.",
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default="",
        help="Optional comma-separated story names to keep.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare rows and print counts without loading Qwen or writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    contexts = _parse_csv_list(args.contexts)
    prompt_map = _load_prompt_map(args.prompts_csv)
    pairs = _load_unique_pairs(
        args.experimental_data,
        string_mode=args.string_mode,
        contexts=contexts,
    )
    pairs = _attach_prompt_prefixes(pairs, prompt_map)

    if pairs.empty:
        raise ValueError("No unique trigger/query pairs remain after filtering.")

    if args.dry_run:
        _print_dry_run_summary(pairs)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] loading {args.model_path}")
    torch_module, tokenizer, model, resolved_model_path = _load_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    print(f"[model] resolved path: {resolved_model_path}")

    pair_scores = _score_pairs(
        pairs,
        torch_module=torch_module,
        tokenizer=tokenizer,
        model=model,
    )
    query_scores = _build_query_level_scores(pair_scores)
    pair_correlations = _build_pair_correlations(pair_scores)
    query_correlations = _build_query_correlations(query_scores)

    pair_scores_path = args.output_dir / "qwen_trigger_query_pair_logprobs.csv"
    query_scores_path = args.output_dir / "qwen_unique_query_logprobs.csv"
    pair_corr_path = args.output_dir / "qwen_trigger_query_pair_correlations.csv"
    query_corr_path = args.output_dir / "qwen_unique_query_correlations.csv"

    pair_scores.to_csv(pair_scores_path, index=False)
    query_scores.to_csv(query_scores_path, index=False)
    pair_correlations.to_csv(pair_corr_path, index=False)
    query_correlations.to_csv(query_corr_path, index=False)

    print("[complete] wrote outputs")
    print(f"  pair_scores={pair_scores_path}")
    print(f"  query_scores={query_scores_path}")
    print(f"  pair_correlations={pair_corr_path}")
    print(f"  query_correlations={query_corr_path}")


if __name__ == "__main__":
    main()
