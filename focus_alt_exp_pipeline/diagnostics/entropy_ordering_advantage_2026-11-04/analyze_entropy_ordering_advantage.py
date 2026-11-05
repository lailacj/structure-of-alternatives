#!/usr/bin/env python3
"""Tabulate full-support entropy and frozen Ordering-minus-Set log scores.

Requires Python 3 and NumPy. All scoring, model-selection, and vocabulary
artifacts are read-only. No sampling, fitting, or inferential analysis is run.
Existing CSV/provenance outputs are never overwritten; use a fresh output dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


RESULTS_NAME = "set_variant_qwen_corrected_rnx_contrastive_2026-10-23"
BOUNDARY = "single-space-exact-concat-v1"
REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"
N_CANDIDATES = 121301
N_CONTEXTS = 16
N_PAIRS = 30
PROBABILITY_ATOL = 1e-12
FIELDS = [
    "context", "prompt_id", "n_candidates", "n_trigger_query_pairs",
    "entropy_bits", "set_log_score", "ordering_log_score",
    "ordering_advantage", "better_of_set_and_ordering",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(rows: list[dict[str, str]], field: str, context: str) -> str:
    values = {row[field] for row in rows}
    require(len(values) == 1 and "" not in values,
            f"{context}: expected one nonempty {field}; found {values}")
    return next(iter(values))


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_vocabulary(arrays_dir: Path, parent: dict) -> tuple[dict, dict, list]:
    manifest_path = arrays_dir / "vocab_manifest.json"
    require(sha256(manifest_path) == parent["vocabulary_manifest_sha256"],
            "Vocabulary manifest differs from the corrected scoring provenance")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["total_count"] == N_CANDIDATES, "Unexpected vocabulary size")
    require([part["name"] for part in manifest["sources"]] == ["1gram", "2gram"],
            "Expected complete unigram then bigram support")
    saved_hashes = {Path(part["path"]).resolve(): part["sha256"]
                    for part in parent["vocabulary_files"]}
    tokens, records = [], [file_record(manifest_path)]
    for words, part in enumerate(manifest["sources"], start=1):
        path = Path(part["path"]).resolve()
        require(path in saved_hashes and sha256(path) == saved_hashes[path],
                f"Vocabulary hash mismatch: {path}")
        values = path.read_text(encoding="utf-8").splitlines()
        require(part["offset"] == len(tokens) and len(values) == part["count"],
                f"Vocabulary count/offset mismatch: {path}")
        require(all(len(value.split()) == words for value in values),
                f"Unexpected candidate word length in {path}")
        tokens.extend(value.strip().lower() for value in values)
        records.append({**file_record(path), "candidate_count": len(values),
                        "offset": part["offset"], "words_per_candidate": words})
    require(len(tokens) == N_CANDIDATES and len(set(tokens)) == N_CANDIDATES,
            "Vocabulary is incomplete or contains duplicate candidates")
    return manifest, {token: index for index, token in enumerate(tokens)}, records


def analyze(args: argparse.Namespace) -> tuple[list[dict], dict]:
    repo = args.repo_root.resolve()
    results = (args.results_dir or repo / "focus_alt_exp_pipeline/results" / RESULTS_NAME).resolve()
    arrays_dir = (args.log_probs_dir or repo.parent / "ngrams/qwen_set_variant_log_probs_single_space").resolve()
    paths = {
        "source_rows": results / "inputs/source_rows.csv",
        "context_scores": results / "linking_structure_tables/log_scores_by_focus_context_and_linking_structure.csv",
        "fold_selections": results / "cv_results/fold_selections.csv",
        "oof_predictions": results / "cv_results/oof_predictions.csv",
        "summed_scores": results / "linking_structure_tables/summed_log_scores.csv",
        "score_reporting": results / "linking_structure_tables/log_score_reporting.json",
        "prompts": results / "inputs/prompts.csv",
        "corrected_parent_provenance": results / "inputs/parent_score_artifact_provenance.json",
    }
    inputs = {name: file_record(path) for name, path in paths.items()}
    source = [row for row in read_csv(paths["source_rows"])
              if row["dataset_family"] == "novel_focus"]
    groups: dict[str, list] = {}
    for row in source:
        groups.setdefault(row["context_id"], []).append(row)
    require(len(groups) == N_CONTEXTS and len(source) == N_CONTEXTS * N_PAIRS,
            f"Expected 16 contexts / 480 rows, found {len(groups)} / {len(source)}")
    scores = read_csv(paths["context_scores"])
    score_map = {row["Context"]: row for row in scores}
    require(len(scores) == N_CONTEXTS and set(score_map) == set(groups),
            "Per-context score table must contain the same 16 unique contexts")

    # Inspect existing selections only; do not call the evaluator or refit K.
    folds = [row for row in read_csv(paths["fold_selections"]) if row["variant"] == "top_k"]
    require(len(folds) == 10 and {int(row["heldout_fold"]) for row in folds} == set(range(10)),
            "Expected one retained Top-K selection for each of ten folds")
    require(all(float(row["selected_boundary"]) == 20.0 for row in folds),
            "The frozen selected K is not 20 in every fold")
    oof = read_csv(paths["oof_predictions"])
    require(all(row["source_rows_sha256"] == inputs["source_rows"]["sha256"] for row in oof),
            "Frozen predictions do not identify this corrected source_rows.csv")
    oof = [row for row in oof if row["dataset_family"] == "novel_focus" and row["variant"] == "top_k"]
    oof_map = {row["analysis_unit_id"]: row for row in oof}
    require(len(oof) == len(oof_map) == len(source), "Frozen predictions contain missing/duplicate focus pairs")
    require(set(oof_map) == {row["item_id"] for row in source}, "Frozen/source pair IDs differ")
    for row in source:
        frozen = oof_map[row["item_id"]]
        require(frozen["analysis_dataset_id"] == "novel_focus" and float(frozen["boundary"]) == 20.0,
                f"Unexpected frozen dataset or boundary: {row['item_id']}")
        require(int(frozen["cv_fold"]) in range(10), "Invalid frozen fold")
        require(math.isclose(float(frozen["human_rate"]), float(row["human_rate"]), rel_tol=0, abs_tol=1e-12),
                f"Frozen/source human rates differ: {row['item_id']}")

    reporting = json.loads(paths["score_reporting"].read_text())
    require(reporting["aggregation"] == "sum_of_item_log_scores"
            and reporting["participant_count_weighting"] is False
            and reporting["item_formula"] == "y*ln(p)+(1-y)*ln(1-p)",
            "Frozen score reporting is not unweighted summed proper log score")
    long_scores = [row for row in read_csv(paths["summed_scores"])
                   if row["dataset_id"] == "novel_focus" and row["context"] != "all"
                   and row["model"] in ("Set Top-K", "Ordering")]
    long_map = {(row["context"], row["model"]): row for row in long_scores}
    require(len(long_scores) == len(long_map) == N_CONTEXTS * 2,
            "Frozen long-form score coverage differs from the 16 context comparison")
    for context, row in score_map.items():
        for model in ("Set Top-K", "Ordering"):
            saved = long_map[(context, model)]
            require(int(saved["n"]) == N_PAIRS and float(saved["sum_log_score"]) == float(row[model]),
                    f"{context}: frozen {model} sum/count mismatch")

    prompts = read_csv(paths["prompts"])
    prompt_map = {row["prompt_id"]: row["generation_prompt"] for row in prompts}
    require(len(prompts) == len(prompt_map), "Duplicate prompt manifest IDs")
    parent = json.loads(paths["corrected_parent_provenance"].read_text())
    require(parent["ready"] is True and parent["model_revision"] == REVISION
            and parent["scoring_boundary_version"] == BOUNDARY,
            "Corrected parent provenance is incompatible")
    parent_arrays = {row["prompt_id"]: row for row in parent["arrays"]}
    manifest, lookup, vocab_records = load_vocabulary(arrays_dir, parent)
    output, validations = [], []
    seen_prompts, seen_arrays = set(), set()
    for context, rows in sorted(groups.items()):
        require(len(rows) == N_PAIRS and len({(r["trigger"], r["query"]) for r in rows}) == N_PAIRS,
                f"{context}: expected 30 unique trigger-query pairs")
        alternatives = {r["trigger"] for r in rows} | {r["query"] for r in rows}
        require(len(alternatives) == 6 and {(r["trigger"], r["query"]) for r in rows}
                == {(a, b) for a in alternatives for b in alternatives if a != b},
                f"{context}: expected all 30 directed pairs among six alternatives")
        prompt_id = unique(rows, "prompt_id", context)
        prompt = unique(rows, "generation_prompt", context)
        require(unique(rows, "generation_frame", context) == "no_frame", f"{context}: non-neutral frame")
        require(prompt_id == "prompt_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
                and prompt_map.get(prompt_id) == prompt, f"{context}: neutral prompt identity mismatch")
        require(all(r["x_but_not_y_prompt"] != prompt for r in rows), f"{context}: framed prompt selected")
        require(unique(rows, "model_revision", context) == REVISION
                and unique(rows, "neutral_scoring_boundary_version", context) == BOUNDARY,
                f"{context}: model revision or corrected boundary mismatch")
        array_path = Path(unique(rows, "source_score_file", context)).resolve()
        require(array_path == arrays_dir / f"{prompt_id}.log_probs.npy",
                f"{context}: source array is outside the specified corrected directory: {array_path}")
        require(prompt_id not in seen_prompts and array_path not in seen_arrays,
                f"{context}: neutral prompt/array is shared by multiple contexts")
        seen_prompts.add(prompt_id)
        seen_arrays.add(array_path)
        expected_hash = unique(rows, "neutral_score_array_sha256", context)
        # Hash and load the same bytes, so the loaded distribution is exactly the audited array.
        array_bytes = array_path.read_bytes()
        actual_hash = hashlib.sha256(array_bytes).hexdigest()
        saved = parent_arrays[prompt_id]
        require(actual_hash == expected_hash == saved["array_sha256"], f"{context}: neutral array SHA-256 mismatch")
        metadata_path = arrays_dir / f"{prompt_id}.meta.json"
        progress_path = arrays_dir / f"{prompt_id}.progress.json"
        require(sha256(metadata_path) == saved["metadata_sha256"]
                and sha256(progress_path) == saved["progress_sha256"],
                f"{context}: corrected metadata/progress hash mismatch")
        metadata = json.loads(metadata_path.read_text())
        expected_metadata = {"context": prompt_id, "prompt": prompt,
                             "scoring_prefix": prompt.rstrip(), "continuation_prefix": " ",
                             "scoring_boundary_version": BOUNDARY, "target_vocab_size": N_CANDIDATES}
        require(all(metadata.get(key) == value for key, value in expected_metadata.items()),
                f"{context}: array metadata does not describe the corrected neutral prompt/support")
        require(Path(metadata["model_path"]).name == REVISION
                and Path(metadata["output_path"]).resolve() == array_path
                and Path(metadata["manifest_path"]).resolve() == arrays_dir / "vocab_manifest.json",
                f"{context}: array metadata model/output/vocabulary mismatch")
        progress = json.loads(progress_path.read_text())["sources"]
        require(set(progress) == {"1gram", "2gram"}
                and all(progress[part["name"]].get("done") is True
                        and progress[part["name"]]["last_line"] == part["count"] for part in manifest["sources"]),
                f"{context}: candidate scoring is incomplete")
        saved_scores = np.load(io.BytesIO(array_bytes), allow_pickle=False)
        require(saved_scores.shape == (N_CANDIDATES,) and saved_scores.dtype == np.float32
                and bool(np.isfinite(saved_scores).all()) and bool((saved_scores <= 0).all()),
                f"{context}: expected 121301 finite saved float32 log scores")
        log_scores = saved_scores.astype(np.float64)
        weights = np.exp(log_scores - np.max(log_scores))
        probabilities = weights / np.sum(weights, dtype=np.float64)
        probability_sum = float(np.sum(probabilities, dtype=np.float64))
        require(math.isclose(probability_sum, 1.0, rel_tol=0, abs_tol=PROBABILITY_ATOL),
                f"{context}: normalized probability sum is {probability_sum}")
        for row in rows:
            for role in ("trigger", "query"):
                index = lookup[row[role].strip().lower()]
                require(math.isclose(float(log_scores[index]), float(row[f"{role}_logprob_sum"]), rel_tol=0, abs_tol=1e-12),
                        f"{context}: {role} score does not match the neutral array")
                require(math.isclose(float(probabilities[index]), float(row[f"{role}_sampling_probability"]), rel_tol=1e-12, abs_tol=0),
                        f"{context}: full-support probabilities differ from the linking-model source")
        positive = probabilities > 0
        entropy = float(-np.sum(probabilities[positive] * np.log2(probabilities[positive]), dtype=np.float64))
        require(math.isfinite(entropy) and 0 <= entropy <= math.log2(N_CANDIDATES) + PROBABILITY_ATOL,
                f"{context}: invalid entropy")
        set_score = float(score_map[context]["Set Top-K"])
        ordering_score = float(score_map[context]["Ordering"])
        require(math.isfinite(set_score) and math.isfinite(ordering_score), f"{context}: invalid frozen log score")
        advantage = ordering_score - set_score
        output.append(dict(zip(FIELDS, [
            context, prompt_id, N_CANDIDATES, len(rows), entropy, set_score, ordering_score,
            advantage, "Ordering" if advantage > 0 else "Set" if advantage < 0 else "Tie",
        ])))
        validations.append({"context": context, "prompt_id": prompt_id,
                            "array_path": str(array_path), "array_sha256": actual_hash,
                            "metadata_sha256": sha256(metadata_path), "progress_sha256": sha256(progress_path),
                            "n_trigger_query_pairs": len(rows), "n_candidates": int(log_scores.size),
                            "finite_score_count": int(np.isfinite(log_scores).sum()),
                            "zero_probability_count": int((~positive).sum()), "probability_sum": probability_sum,
                            "passed": True})
    output.sort(key=lambda row: (row["entropy_bits"], row["context"]))
    by_context = {row["context"]: row for row in output}
    arithmetic = {}
    for context, expected in (("hot", -28.4984), ("mall", 133.4571)):
        actual = by_context[context]["ordering_advantage"]
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=0.00005),
                f"{context}: advantage {actual} fails supplied arithmetic check {expected}")
        arithmetic[context] = {"actual": actual, "expected_approximately": expected, "passed": True}
    report = {
        "validation_passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo), "git_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "hostname": socket.gethostname(), "python_version": sys.version,
        "numpy_version": np.__version__, "working_directory": str(Path.cwd().resolve()),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "script": file_record(Path(__file__)), "inputs": inputs, "vocabulary": vocab_records,
        "calculation": {"filter": "dataset_family == novel_focus", "grain": "one entropy per context",
                        "normalization": "float64; weights=exp(log_scores-max(log_scores)); p=weights/sum(weights)",
                        "entropy": "-sum(p*log2(p)) over every positive p; zero contributions are zero",
                        "support": "all 98509 one-word and 22792 two-word saved candidates; no subset renormalization",
                        "probability_sum_absolute_tolerance": PROBABILITY_ATOL,
                        "set_score_column": "Set Top-K", "ordering_score_column": "Ordering",
                        "log_score_aggregation": reporting,
                        "ordering_advantage": "ordering_log_score - set_log_score",
                        "comparison_scope": "pairwise Set/Ordering only",
                        "csv_precision": "Python float round-trip representation; no rounding"},
        "validation": {"n_contexts": len(groups), "source_pairs": len(source),
                       "frozen_oof_pairs": len(oof), "selected_k_by_fold": {
                           row["heldout_fold"]: float(row["selected_boundary"]) for row in folds},
                       "neutral_prompts_arrays_and_source_probabilities_match": True,
                       "complete_vocabulary_and_saved_hashes_match": True,
                       "frozen_sums_each_cover_same_30_pairs": True,
                       "arithmetic_checks": arithmetic},
        "arrays": validations,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--results-dir", type=Path, help="Frozen corrected results directory")
    parser.add_argument("--log-probs-dir", type=Path, help="Corrected array directory identified by source metadata")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory without existing CSV/provenance outputs")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "entropy_ordering_advantage.csv"
    provenance_path = output_dir / "provenance_validation.json"
    for path in (csv_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    rows, report = analyze(args)  # Validate all contexts before writing any results.
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report["output_csv"] = file_record(csv_path)
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Validated {len(rows)} contexts; K=20 in all 10 folds.")
    print(f"CSV: {csv_path}")
    print(f"Provenance: {provenance_path}")
    print("context | entropy_bits | set_log_score | ordering_log_score | ordering_advantage | better")
    for row in rows:
        print(f"{row['context']:10} | {row['entropy_bits']:.6f} | {row['set_log_score']:.4f} | "
              f"{row['ordering_log_score']:.4f} | {row['ordering_advantage']:+.4f} | {row['better_of_set_and_ordering']}")


if __name__ == "__main__":
    main()
