"""Rebuild and verify the versioned Day 1 cross-dataset result snapshot.

The command runs both existing builders into an isolated output root, then
compares every generated CSV with its committed counterpart by row count and
SHA-256 checksum.  It refuses to write into a non-empty output directory so a
verification run cannot silently mix old and new artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "focus_alt_exp_pipeline"
    / "analysis_configs"
    / "day1_absolute_threshold_v1.json"
)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "schema_version",
        "config_id",
        "config_version",
        "model",
        "expectedness_score",
        "aggregation_grain",
        "cross_validation",
        "calibration",
        "paths",
        "expected_row_counts",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing)}")
    if config["schema_version"] != "focus-alternatives-analysis-config/v1":
        raise ValueError(f"Unsupported schema_version: {config['schema_version']}")
    if config["expectedness_score"]["primary"] != "summed_continuation_log_probability":
        raise ValueError("The current Day 1 builder supports summed log probability only")
    if config["cross_validation"]["seed"] is not None:
        raise ValueError("The current deterministic fold builder does not consume a seed")
    return config


def _prepare_clean_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Clean output root is not empty: {path}. "
            "Pass --output-root with a new directory for another rebuild."
        )
    path.mkdir(parents=True, exist_ok=True)


def _run(command: list[str], *, label: str) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(
        part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
    )
    if output:
        print(output)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    return output


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _comparison_record(
    *,
    clean: Path,
    committed: Path,
    expected_rows: int,
) -> dict[str, Any]:
    if not clean.is_file():
        raise FileNotFoundError(f"Expected generated file is missing: {clean}")
    if not committed.is_file():
        raise FileNotFoundError(f"Committed comparison file is missing: {committed}")
    clean_rows = _csv_row_count(clean)
    committed_rows = _csv_row_count(committed)
    if clean_rows != expected_rows:
        raise ValueError(
            f"Unexpected row count for {clean}: expected={expected_rows}, observed={clean_rows}"
        )
    clean_hash = _sha256(clean)
    committed_hash = _sha256(committed)
    return {
        "generated_path": _display_path(clean),
        "committed_path": _display_path(committed),
        "expected_rows": expected_rows,
        "generated_rows": clean_rows,
        "committed_rows": committed_rows,
        "row_count_match": clean_rows == committed_rows,
        "generated_sha256": clean_hash,
        "committed_sha256": committed_hash,
        "sha256_match": clean_hash == committed_hash,
        "exact_match": clean_rows == committed_rows and clean_hash == committed_hash,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the config's clean output root; the directory must be empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    paths = config["paths"]
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else _resolve_repo_path(paths["clean_output_root"])
    )
    _prepare_clean_root(output_root)

    canonical_dir = output_root / "canonical"
    table_dir = output_root / "big_table"
    canonical_builder = _resolve_repo_path(paths["canonical_builder"])
    table_builder = _resolve_repo_path(paths["table_builder"])
    canonical_input = canonical_dir / "all_observations.csv"
    fold_count = int(config["cross_validation"]["fold_count"])

    commands = [
        [
            sys.executable,
            str(canonical_builder),
            "--output-dir",
            str(canonical_dir),
        ],
        [
            sys.executable,
            str(table_builder),
            "--input",
            str(canonical_input),
            "--output-dir",
            str(table_dir),
            "--fold-count",
            str(fold_count),
        ],
    ]

    log_parts = []
    for label, command in zip(("canonical build", "development table build"), commands):
        log_parts.append(f"$ {' '.join(command)}")
        log_parts.append(_run(command, label=label))
    (output_root / "build.log").write_text("\n\n".join(log_parts) + "\n", encoding="utf-8")
    shutil.copyfile(config_path, output_root / "analysis_config.json")

    committed_canonical = _resolve_repo_path(paths["committed_canonical_dir"])
    committed_table = _resolve_repo_path(paths["committed_table_dir"])
    comparisons = []
    for relative, expected_rows in config["expected_row_counts"].items():
        clean = output_root / relative
        relative_path = Path(relative)
        if relative_path.parts[0] == "canonical":
            committed = committed_canonical / relative_path.name
        elif relative_path.parts[0] == "big_table":
            committed = committed_table / relative_path.name
        else:
            raise ValueError(f"Unsupported expected output group: {relative}")
        comparisons.append(
            _comparison_record(
                clean=clean,
                committed=committed,
                expected_rows=int(expected_rows),
            )
        )

    generated_revision = None
    with canonical_input.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        revisions = {row["model_revision"] for row in reader}
    if len(revisions) != 1:
        raise ValueError(f"Expected one generated model revision, found {sorted(revisions)}")
    generated_revision = next(iter(revisions))
    if generated_revision != config["model"]["revision"]:
        raise ValueError(
            "Generated model revision does not match the versioned configuration: "
            f"{generated_revision} != {config['model']['revision']}"
        )

    manifest = {
        "manifest_schema_version": "focus-alternatives-rebuild-manifest/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": config["config_id"],
        "config_version": config["config_version"],
        "config_sha256": _sha256(config_path),
        "output_root": _display_path(output_root),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "repository": {
            "commit": _git_value("rev-parse", "HEAD"),
            "status_short": _git_value("status", "--short"),
        },
        "model_revision": generated_revision,
        "commands": commands,
        "comparisons": comparisons,
        "all_expected_row_counts_match": all(
            record["generated_rows"] == record["expected_rows"]
            for record in comparisons
        ),
        "all_committed_row_counts_match": all(
            record["row_count_match"] for record in comparisons
        ),
        "all_committed_sha256_match": all(
            record["sha256_match"] for record in comparisons
        ),
        "all_committed_outputs_exact_match": all(
            record["exact_match"] for record in comparisons
        ),
    }
    manifest_path = output_root / "verification_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[complete] verification manifest: {manifest_path}")
    print(f"[verify] expected row counts match: {manifest['all_expected_row_counts_match']}")
    print(f"[verify] committed row counts match: {manifest['all_committed_row_counts_match']}")
    print(f"[verify] committed SHA-256 match: {manifest['all_committed_sha256_match']}")
    if not manifest["all_committed_outputs_exact_match"]:
        raise RuntimeError(
            "Clean rebuild completed, but at least one output differs from the committed artifact. "
            "Inspect verification_manifest.json."
        )


if __name__ == "__main__":
    main()
