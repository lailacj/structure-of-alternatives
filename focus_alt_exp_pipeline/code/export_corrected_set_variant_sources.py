"""Export direct neutral scores from the exact arrays used by linking models.

Fail closed on legacy/incomplete arrays, changed prompts/support/model, or a
failed audit of the existing query-only X-but-not-Y scores. This never loads
model weights or modifies a score array. Historical inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from score_qwen_scoring_manifest import SCORING_BOUNDARY_VERSION, continuation_token_ids, split_prompt_boundary
from validate_set_variant_qwen_scores import _load_vocab, validate_scores

REVISION = "453ed1575b739b5b03ce3758b23befdb0967f40e"
PIPELINE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_ids(tokenizer, prompt, candidate):
    prefix, continuation = split_prompt_boundary(prompt, candidate)
    return continuation_token_ids(tokenizer, prefix=prefix, continuation=continuation)[0]


def audit_framed(source, scores, tokenizer, *, revision=REVISION):
    """Validate historical framed scores rather than relabel or recompute them.

    Version 1.1 retained all trailing prompt whitespace. A one-space prompt,
    exact matching token IDs, and recorded per-token scores establish that the
    existing rows already used the corrected boundary convention. This is a
    tokenization/provenance audit, not an independent model-forward replication.
    """
    framed = scores.loc[scores.generation_frame.eq("x_but_not_y")].copy()
    expected = source.loc[source.x_but_not_y_logprob_sum.notna()].copy()
    if not len(framed) or not len(expected):
        raise ValueError("Framed audit requires the existing scored rows")
    indexed = {}
    for _, row in framed.iterrows():
        prompt, query = str(row.generation_prompt), str(row.query)
        prefix, continuation = split_prompt_boundary(prompt, query)
        if prompt != prefix + " ":
            raise ValueError("Historical framed prompt did not have exactly one trailing space")
        if str(row.model_revision) != revision or row.query_tokenization_mode != "exact_concat":
            raise ValueError("Framed model revision/tokenization convention mismatch")
        if row.scored_candidate_roles != "query_only" or pd.notna(row.trigger_logprob_sum):
            raise ValueError("Framed rows must score only the query")
        ids = candidate_ids(tokenizer, prompt, query)
        recorded_ids = json.loads(row.query_token_ids_json)
        logs = np.asarray(json.loads(row.query_token_logprobs_json), dtype=float)
        tokens = json.loads(row.query_tokens_json)
        if ids != recorded_ids or len(ids) != int(row.query_token_count):
            raise ValueError(f"Framed candidate token IDs disagree for {query!r}")
        if tokens != list(tokenizer.convert_ids_to_tokens(ids)):
            raise ValueError("Framed tokenizer token strings disagree")
        if len(logs) != len(ids) or not np.isfinite(logs).all() or (logs > 0).any():
            raise ValueError("Invalid recorded framed token scores")
        if not np.isclose(logs.sum(), row.query_logprob_sum, rtol=0, atol=1e-8):
            raise ValueError("Framed summed score does not match token scores")
        if not np.isclose(logs.mean(), row.query_logprob_mean, rtol=0, atol=1e-8):
            raise ValueError("Framed mean score does not match token scores")
        key = (prompt, query)
        value = (float(row.query_logprob_sum), float(row.query_logprob_mean), len(ids))
        if key in indexed and indexed[key] != value:
            raise ValueError("Conflicting repeated framed prompt/candidate scores")
        indexed[key] = value
    matched = set()
    for _, row in expected.iterrows():
        query = row.get("x_but_not_y_query")
        if pd.isna(query) or not str(query).strip():
            query = row["query"]
        key = (str(row.x_but_not_y_prompt), str(query))
        if key not in indexed:
            raise ValueError(f"Missing framed artifact row for {key!r}")
        value, mean, count = indexed[key]
        if not np.isclose(value, row.x_but_not_y_logprob_sum, atol=1e-8, rtol=0):
            raise ValueError("Source-row framed score disagrees with original artifact")
        if not np.isclose(mean, row.x_but_not_y_logprob_mean, atol=1e-8, rtol=0) or count != int(row.x_but_not_y_token_count):
            raise ValueError("Source-row framed token metadata disagrees")
        matched.add(key)
    if matched != set(indexed):
        raise ValueError("Framed source coverage does not match the audited artifact")
    return {"passed": True, "framed_score_rows": len(framed), "source_rows": len(expected),
            "unique_prompt_candidate_pairs": len(indexed), "model_revision": revision,
            "equivalent_boundary": SCORING_BOUNDARY_VERSION,
            "action": "reuse verified existing framed scores unchanged",
            "scope": "exact tokenization, prompt boundary, model revision, recorded sums/means and source mapping; no model-forward rerun"}


def export_neutral(source, prompts, directory, tokenizer, *, revision=REVISION):
    manifest, lookup = _load_vocab(directory)
    offset = 0
    for part in manifest["sources"]:
        if int(part["offset"]) != offset:
            raise ValueError("Vocabulary offsets are not contiguous")
        offset += int(part["count"])
    if offset != int(manifest["total_count"]) or "" in lookup:
        raise ValueError("Invalid vocabulary support")
    prompt_map = dict(zip(prompts.prompt_id, prompts.generation_prompt))
    if len(prompt_map) != len(prompts) or set(prompt_map) != set(source.prompt_id):
        raise ValueError("Source rows and prompt manifest have different prompt coverage")
    corrected = source.copy()
    provenance, target_rows = [], []
    for pid, rows in source.groupby("prompt_id", sort=True):
        path = directory / f"{pid}.log_probs.npy"
        meta_path = directory / f"{pid}.meta.json"
        progress_path = directory / f"{pid}.progress.json"
        metadata = json.loads(meta_path.read_text())
        progress = json.loads(progress_path.read_text())
        prefix, _ = split_prompt_boundary(prompt_map[pid], "placeholder")
        checks = {"context": pid, "prompt": prompt_map[pid], "scoring_prefix": prefix,
                  "continuation_prefix": " ", "scoring_boundary_version": SCORING_BOUNDARY_VERSION,
                  "target_vocab_size": offset}
        for key, value in checks.items():
            if metadata.get(key) != value:
                raise ValueError(f"{pid}: incompatible {key}")
        if Path(metadata.get("model_path", "")).name != revision:
            raise ValueError(f"{pid}: model snapshot revision mismatch")
        if set(progress["sources"]) != {"1gram", "2gram"} or not all(x.get("done") for x in progress["sources"].values()):
            raise ValueError(f"{pid}: incomplete array")
        scores = np.load(path, allow_pickle=False)
        if scores.dtype != np.float32 or scores.shape != (offset,) or not np.isfinite(scores).all() or (scores > 0).any():
            raise ValueError(f"{pid}: invalid score array")
        if not rows.generation_prompt.eq(prompt_map[pid]).all() or not rows.model_revision.eq(revision).all():
            raise ValueError("Source prompt or model revision changed")
        log_z = float(scores.max()) + np.log(np.exp(scores.astype(float) - float(scores.max())).sum())
        digest = sha(path)
        per_word = {}
        for word in sorted(set(rows.trigger) | set(rows["query"])):
            index = lookup[str(word).strip().lower()]
            ids = candidate_ids(tokenizer, prompt_map[pid], word)
            logp = float(scores[index])
            probability = float(np.exp(logp - log_z))
            if not 0 < probability <= 1:
                raise ValueError("Required normalized candidate probability is invalid")
            per_word[word] = (logp, len(ids), probability)
            target_rows.append({"prompt_id": pid, "candidate": word, "array_index": index,
                "logprob_sum": logp, "logprob_mean": logp / len(ids), "token_count": len(ids),
                "token_ids_json": json.dumps(ids), "sampling_probability": probability,
                "log_normalizer": log_z, "array_path": str(path.resolve()), "array_sha256": digest,
                "scoring_boundary_version": SCORING_BOUNDARY_VERSION, "model_revision": revision})
        for role in ("trigger", "query"):
            for idx, row in rows.iterrows():
                logp, count, probability = per_word[row[role]]
                corrected.loc[idx, f"{role}_logprob_sum"] = logp
                corrected.loc[idx, f"{role}_logprob_mean"] = logp / count
                corrected.loc[idx, f"{role}_token_count"] = count
                corrected.loc[idx, f"{role}_tokenization_mode"] = "exact_concat"
                corrected.loc[idx, f"{role}_sampling_probability"] = probability
        corrected.loc[rows.index, "source_score_file"] = str(path.resolve())
        corrected.loc[rows.index, "neutral_score_array_sha256"] = digest
        corrected.loc[rows.index, "neutral_scoring_boundary_version"] = SCORING_BOUNDARY_VERSION
        provenance.append({"prompt_id": pid, "array": str(path.resolve()), "array_sha256": digest,
                           "metadata_sha256": sha(meta_path), "progress_sha256": sha(progress_path)})
    # Human outcomes, analysis membership, prompts, IDs, and framed scores are
    # immutable. Only declared neutral-score/provenance fields may change.
    mutable = {f"{role}_{suffix}" for role in ("trigger", "query") for suffix in
               ("logprob_sum", "logprob_mean", "token_count", "tokenization_mode")}
    unchanged = [c for c in source.columns if c not in mutable | {"source_score_file"}]
    pd.testing.assert_frame_equal(source[unchanged], corrected[unchanged])
    return corrected, pd.DataFrame(target_rows), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--log-probs-dir", type=Path)
    parser.add_argument("--framed-scores", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.model_path.name != REVISION:
        raise ValueError("Pass the exact pinned Qwen snapshot directory")
    if args.output_dir.resolve() == args.manifest_dir.resolve():
        raise ValueError("Choose a new output directory; never overwrite the source manifest")
    for protected in (PIPELINE / "results/set_variant_qwen", PIPELINE / "canonical_data", PIPELINE / "model_scores"):
        if args.output_dir.resolve() == protected.resolve() or protected.resolve() in args.output_dir.resolve().parents:
            raise ValueError("Output must not be inside a historical result/data directory")
    output = args.output_dir / "source_rows.csv"
    if output.exists() and not args.audit_only:
        raise FileExistsError(f"Corrected source already exists; refusing overwrite: {output}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=True, local_files_only=True)
    source = pd.read_csv(args.manifest_dir / "source_rows.csv")
    scores = pd.read_csv(args.framed_scores)
    audit = audit_framed(source, scores, tokenizer)
    audit.update(framed_scores=str(args.framed_scores.resolve()), framed_scores_sha256=sha(args.framed_scores),
                 source_rows_sha256=sha(args.manifest_dir / "source_rows.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "framed_scoring_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    if args.log_probs_dir is None:
        raise ValueError("--log-probs-dir is required for neutral score extraction")
    validation = validate_scores(manifest_dir=args.manifest_dir, log_probs_dir=args.log_probs_dir, allow_incomplete=False)
    prompts = pd.read_csv(args.manifest_dir / "prompts.csv")
    corrected, targets, arrays = export_neutral(source, prompts, args.log_probs_dir, tokenizer)
    corrected["x_but_not_y_score_artifact"] = np.where(corrected.x_but_not_y_logprob_sum.notna(), str(args.framed_scores.resolve()), "")
    corrected.to_csv(output, index=False)
    targets.to_csv(args.output_dir / "neutral_candidate_scores.csv", index=False)
    for name in ("prompts.csv", "selection_manifest.json", "required_candidates.txt"):
        original = args.manifest_dir / name
        if original.exists():
            destination = args.output_dir / name
            if destination.exists() and sha(destination) != sha(original):
                raise ValueError(f"Refusing to overwrite changed {destination}")
            shutil.copyfile(original, destination)
    report = {"ready": True, "model_revision": REVISION, "scoring_boundary_version": SCORING_BOUNDARY_VERSION,
        "source_rows": len(corrected), "unique_neutral_candidates": len(targets), "validation": validation,
        "original_source_sha256": sha(args.manifest_dir / "source_rows.csv"), "corrected_source_sha256": sha(output),
        "vocabulary_manifest_sha256": sha(args.log_probs_dir / "vocab_manifest.json"),
        "vocabulary_files": [{"path": x["path"], "sha256": sha(x["path"])} for x in json.loads((args.log_probs_dir / "vocab_manifest.json").read_text())["sources"]],
        "framed_audit": audit, "arrays": arrays,
        "direct_baseline_definition": "exp(query_logprob_sum) from the same float32 array used by all linking rules; sampling_probability is additionally exported without renormalizing any subset"}
    (args.output_dir / "score_artifact_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[complete] corrected source rows={len(corrected)}, prompt/candidate pairs={len(targets)}, arrays={len(arrays)}")


if __name__ == "__main__":
    main()
