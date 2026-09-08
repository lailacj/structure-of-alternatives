"""Validate completeness and candidate coverage of set-variant Qwen scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PIPELINE_DIR.parent
DEFAULT_MANIFEST_DIR = PIPELINE_DIR / "scoring_manifests" / "set_variant_qwen"
DEFAULT_LOG_PROBS_DIR = ROOT_DIR.parent / "ngrams" / "qwen_set_variant_log_probs"


def _load_vocab(log_probs_dir: Path) -> tuple[dict, dict[str, int]]:
    path = log_probs_dir / "vocab_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing vocabulary manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokens: list[str] = []
    for source in manifest.get("sources", []):
        source_path = Path(source["path"])
        values = source_path.read_text(encoding="utf-8").splitlines()
        if len(values) != int(source["count"]):
            raise ValueError(f"Vocabulary count changed for {source_path}")
        tokens.extend(values)
    lookup = {token.strip().lower(): index for index, token in enumerate(tokens)}
    if len(lookup) != len(tokens) or len(tokens) != int(manifest["total_count"]):
        raise ValueError("Scored vocabulary contains duplicates or has an inconsistent total")
    return manifest, lookup


def validate_scores(
    *, manifest_dir: Path, log_probs_dir: Path, allow_incomplete: bool
) -> dict:
    prompts = pd.read_csv(manifest_dir / "prompts.csv")
    source_rows = pd.read_csv(manifest_dir / "source_rows.csv")
    if prompts["prompt_id"].duplicated().any():
        raise ValueError("Prompt manifest contains duplicate prompt IDs")
    vocab_manifest, lookup = _load_vocab(log_probs_dir)
    total_count = int(vocab_manifest["total_count"])
    required_global = {
        str(value).strip().lower()
        for value in pd.concat([source_rows["trigger"], source_rows["query"]])
    }
    absent_global = sorted(required_global.difference(lookup))

    complete: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    problems: list[dict] = []
    for row in prompts.itertuples(index=False):
        prompt_id = str(row.prompt_id)
        paths = {
            "array": log_probs_dir / f"{prompt_id}.log_probs.npy",
            "progress": log_probs_dir / f"{prompt_id}.progress.json",
            "metadata": log_probs_dir / f"{prompt_id}.meta.json",
        }
        absent_files = [name for name, path in paths.items() if not path.exists()]
        if absent_files:
            missing.append(prompt_id)
            if len(absent_files) != len(paths):
                problems.append({"prompt_id": prompt_id, "issue": "partial_file_set", "details": absent_files})
            continue
        try:
            progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
            sources = progress["sources"]
            is_complete = bool(sources["1gram"].get("done") and sources["2gram"].get("done"))
            if not is_complete:
                partial.append(prompt_id)
                continue
            array = np.load(paths["array"], mmap_mode="r")
            if array.shape != (total_count,) or array.dtype != np.float32:
                problems.append({"prompt_id": prompt_id, "issue": "array_schema", "details": [list(array.shape), str(array.dtype)]})
                continue
            nonfinite_count = int((~np.isfinite(array)).sum())
            if nonfinite_count:
                problems.append({"prompt_id": prompt_id, "issue": "nonfinite_scores", "details": nonfinite_count})
            required = {
                str(value).strip().lower()
                for value in pd.concat([
                    source_rows.loc[source_rows["prompt_id"].eq(prompt_id), "trigger"],
                    source_rows.loc[source_rows["prompt_id"].eq(prompt_id), "query"],
                ])
            }
            nonfinite_required = sorted(
                token for token in required if token in lookup and not np.isfinite(array[lookup[token]])
            )
            if nonfinite_required:
                problems.append({"prompt_id": prompt_id, "issue": "nonfinite_required_candidates", "details": nonfinite_required})
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("context") != prompt_id or int(metadata.get("target_vocab_size", -1)) != total_count:
                problems.append({"prompt_id": prompt_id, "issue": "metadata_mismatch", "details": metadata})
            complete.append(prompt_id)
        except Exception as error:
            problems.append({"prompt_id": prompt_id, "issue": "validation_exception", "details": repr(error)})

    report = {
        "expected_prompts": int(len(prompts)),
        "complete_prompts": len(complete),
        "partial_prompts": len(partial),
        "missing_prompts": len(missing),
        "candidate_count": total_count,
        "required_candidate_count": len(required_global),
        "absent_required_candidates": absent_global,
        "problem_count": len(problems),
        "partial_prompt_ids": sorted(partial),
        "missing_prompt_ids": sorted(missing),
        "problems": problems,
        "ready": not absent_global and not problems and not partial and not missing,
    }
    if not allow_incomplete and not report["ready"]:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--log-probs-dir", type=Path, default=DEFAULT_LOG_PROBS_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_scores(
        manifest_dir=args.manifest_dir,
        log_probs_dir=args.log_probs_dir,
        allow_incomplete=args.allow_incomplete,
    )
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
