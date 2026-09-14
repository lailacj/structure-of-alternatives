"""Build an offline, self-contained viewer from the active sampled-prefix results.

The default build uses only Python's standard library. Optional full vocabulary
exports require NumPy and the original score arrays and vocabulary manifest.
No model inference, fitting, or result-file changes are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
DATASETS = [
    ("hu_vt16", "van Tiel et al. (2016)"),
    ("hu_g18", "Gotzner et al. (2018)"),
    ("hu_pvt21", "Pankratz & van Tiel (2021)"),
    ("hu_rx22", "Ronai & Xiang (2022)"),
    ("rnx_esi", "R&X ESI"), ("rnx_eweak", "R&X Eweak"),
    ("rnx_estrong", "R&X Estrong"), ("rnx_eonly", "R&X Eonly"),
    ("rnx_eonlystrong", "R&X Eonlystrong"),
    ("novel_focus", "Novel Focus Alternative Study"),
]
MODELS = ["No linking structure", "X but not Y", "Set Top-K", "Set Top-p",
          "Ordering", "Conjunction Top-K", "Conjunction Top-p",
          "Disjunction Top-K", "Disjunction Top-p"]
SAMPLED = {2: ("top_k", "set"), 3: ("top_p", "set"),
           4: ("top_k", "ordering"), 5: ("top_k", "conjunction"),
           6: ("top_p", "conjunction"), 7: ("top_k", "disjunction"),
           8: ("top_p", "disjunction")}
EPSILON = 1e-10


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or str(value).strip().lower() in ("", "nan", "na"):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite number: {!r}".format(value))
    return result


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def log_score(y, p):
    if not 0 <= y <= 1 or not 0 <= p <= 1:
        raise ValueError("Human rates and model probabilities must be in [0, 1]")
    p = min(1 - EPSILON, max(EPSILON, p))
    return y * math.log(p) + (1 - y) * math.log1p(-p)


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(math.fsum(x*x for x in dx) * math.fsum(y*y for y in dy))
    if not denominator:
        return None
    return max(-1., min(1., math.fsum(x*y for x, y in zip(dx, dy)) / denominator))


def dataset_id(row):
    family = row["dataset_family"]
    if family == "hu_2023_benchmark":
        return "hu_" + row["dataset"]
    if family == "ronai_xiang_2024":
        return "rnx_" + row["condition"].lower()
    if family == "novel_focus":
        return "novel_focus"
    raise ValueError("Unknown dataset family: " + family)


def unit_key(row):
    unit = row["item_id"]
    if row["dataset_family"] == "hu_2023_benchmark":
        unit = row["dataset"] + "::" + row["scale_id"]
    return dataset_id(row), unit


def included(row):
    return row["dataset_family"] != "hu_2023_benchmark" or row["hu_original_analysis_included"].lower() == "true"


def summarize(items):
    records = []
    for model in range(len(MODELS)):
        rows = [row for row in items if row["p"][model] is not None]
        records.append({"n": len(rows), "r": pearson([r["p"][model] for r in rows], [r["y"] for r in rows]),
                        "log": mean(r["scores"][model] for r in rows),
                        "human": mean(r["y"] for r in rows), "prediction": mean(r["p"][model] for r in rows)})
    return records


def build_items(source, oof):
    source_groups, variants = defaultdict(list), defaultdict(dict)
    for index, row in enumerate(source):
        if included(row):
            source_groups[unit_key(row)].append((index, row))
    for row in oof:
        key = row["analysis_dataset_id"], row["analysis_unit_id"]
        if row["variant"] in variants[key]:
            raise ValueError("Duplicate out-of-fold analysis unit: " + str(key))
        variants[key][row["variant"]] = row
    if set(source_groups) != set(variants):
        raise ValueError("Source and out-of-fold analysis units differ")
    items = []
    for key in sorted(variants):
        pair = variants[key]
        if set(pair) != {"top_k", "top_p"}:
            raise ValueError("Incomplete variants: " + str(key))
        k, p = pair["top_k"], pair["top_p"]
        y = number(k["human_rate"])
        rows = source_groups[key]
        if abs(y - mean(number(row["human_rate"]) for _, row in rows)) > 1e-10:
            raise ValueError("Source and OOF human rates disagree")
        if number(p["human_rate"]) != y or k["cv_fold"] != p["cv_fold"]:
            raise ValueError("OOF variants disagree on human rate or fold")
        if abs(number(k["ordering_probability"]) - number(p["ordering_probability"])) > 1e-10:
            raise ValueError("Ordering must be boundary-independent")
        predictions = [None] * len(MODELS)
        for model, column in [(0, "query_logprob_sum"), (1, "x_but_not_y_logprob_sum")]:
            values = [number(row[column]) for _, row in rows]
            available = [math.exp(value) for value in values if value is not None]
            if available and len(available) != len(values):
                raise ValueError("Partial direct-score coverage")
            predictions[model] = mean(available)
        for model, (variant, structure) in SAMPLED.items():
            predictions[model] = number(pair[variant][structure + "_probability"])
        first = rows[0][1]
        items.append({"dataset": key[0], "id": key[1], "context": first["context_id"] if key[0] == "novel_focus" else "",
                      "trigger": first["trigger"], "query": first["query"], "y": y,
                      "fold": int(k["cv_fold"]), "k": number(k["boundary"]), "topP": number(p["boundary"]),
                      "p": predictions, "scores": [log_score(y, value) if value is not None else None for value in predictions],
                      "sources": [index for index, _ in rows]})
    return items


def verify_tables(results, summaries):
    checked = 0
    for metric, filename in [("r", "correlations"), ("log", "log_scores")]:
        path = results / "linking_structure_tables" / (filename + "_by_dataset_and_linking_structure.csv")
        table = {row["Dataset"]: row for row in read_csv(path)}
        for dataset, label in DATASETS:
            for index, model in enumerate(MODELS):
                expected, actual = number(table[label][model]), summaries[dataset][index][metric]
                if (expected is None) != (actual is None) or (expected is not None and abs(expected - actual) > 1e-8):
                    raise ValueError("Viewer disagrees with saved {}: {} / {}: {} vs {}".format(metric, label, model, actual, expected))
                checked += 1
    return checked


def build_prompts(source):
    prompts, compact_source = {}, []
    def add(prompt_id, text, frame, row, word, role, prefix):
        prompt = prompts.setdefault(prompt_id, {"id": prompt_id, "text": text, "frame": frame,
            "datasets": set(), "contexts": set(), "candidates": {}, "distribution": [],
            "distributionCoverage": "unavailable"})
        if prompt["text"] != text:
            raise ValueError("Prompt ID maps to conflicting text")
        prompt["datasets"].add(dataset_id(row))
        if dataset_id(row) == "novel_focus":
            prompt["contexts"].add(row["context_id"])
        value = number(row[prefix + "logprob_sum"])
        if value is not None:
            candidate = prompt["candidates"].setdefault(word, {"word": word, "logp": value,
                "tokens": number(row.get(prefix + "token_count")), "roles": set()})
            if abs(candidate["logp"] - value) > 1e-5:
                raise ValueError("Conflicting scores for same prompt/candidate: " + word)
            candidate["roles"].add(role)
    for row in source:
        pid = row["prompt_id"]
        add(pid, row["generation_prompt"], "Neutral", row, row["trigger"], "trigger", "trigger_")
        add(pid, row["generation_prompt"], "Neutral", row, row["query"], "query", "query_")
        framed_id = None
        if number(row["x_but_not_y_logprob_sum"]) is not None:
            text = row["x_but_not_y_prompt"]
            framed_id = "framed_" + hashlib.sha256(text.encode()).hexdigest()[:20]
            add(framed_id, text, "X but not Y", row, row.get("x_but_not_y_query") or row["query"], "query", "x_but_not_y_")
        compact_source.append({"dataset": dataset_id(row), "context": row["context_id"], "item": row["item_id"],
            "trigger": row["trigger"], "query": row["query"], "prompt": pid, "framedPrompt": framed_id,
            "included": included(row), "human": number(row["human_rate"]), "yes": number(row["human_yes"]),
            "total": number(row["human_total"]), "countStatus": row["human_count_status"],
            "scoreFile": row["source_score_file"], "humanFile": row["source_human_file"]})
    for prompt in prompts.values():
        prompt["datasets"] = sorted(prompt["datasets"])
        prompt["contexts"] = sorted(prompt["contexts"])
        prompt["candidates"] = sorted(prompt["candidates"].values(), key=lambda x: -x["logp"])
        for candidate in prompt["candidates"]:
            candidate["roles"] = sorted(candidate["roles"])
    return prompts, compact_source


def add_diagnostic(prompts, pipeline):
    """Expose the checked-in mask distribution extract, with its distinct source."""
    directory = pipeline / "diagnostics/mask_top_p_2026-09-11"
    top = directory / "tables/top_100_candidates.csv"
    if not top.exists():
        return []
    candidates = {}
    for filename in ["top_100_candidates.csv", "experimental_alternatives.csv"]:
        for row in read_csv(directory / "tables" / filename):
            candidates[row["candidate"]] = {"word": row["candidate"], "logp": number(row["saved_log_score"]),
                "normalized": number(row["normalized_sampling_probability"]), "rank": int(row["rank_one_based"])}
    matches = [p for p in prompts.values() if p["frame"] == "Neutral" and "mask" in p["contexts"]]
    if len(matches) != 1:
        raise ValueError("Mask diagnostic does not map to exactly one neutral prompt")
    # Check provenance against the source manifest before using a diagnostic extract.
    provenance = json.loads((directory / "input_provenance.json").read_text())
    recorded = next(f["sha256"] for f in provenance["files"] if f["role"] == "source_rows")
    current = hashlib.sha256((pipeline / "scoring_manifests/set_variant_qwen/source_rows.csv").read_bytes()).hexdigest()
    if recorded != current:
        return []
    prompt = matches[0]
    prompt["distribution"] = sorted(candidates.values(), key=lambda x: x["rank"])
    prompt["distributionCoverage"] = "Saved top 100 + experimental alternatives (mask diagnostic, 2026-09-11)"
    prompt["supportSize"] = 121301
    return [top, directory / "tables/experimental_alternatives.csv", directory / "input_provenance.json"]


def add_full_distributions(prompts, directory, top_n, vocab_directory=None):
    """Read each array once; retain top N plus experimental targets (0 = all)."""
    import numpy as np
    manifest_path = directory / "vocab_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    words, paths = [], [manifest_path]
    for source in manifest["sources"]:
        path = (vocab_directory / Path(source["path"]).name) if vocab_directory else Path(source["path"])
        if not path.is_absolute():
            path = directory / path
        values = path.read_text(encoding="utf-8").splitlines()
        if len(values) != int(source["count"]):
            raise ValueError("Vocabulary source count mismatch: " + str(path))
        words.extend(values)
        paths.append(path)
    lookup = {word.strip().lower(): i for i, word in enumerate(words)}
    if len(lookup) != len(words) or len(words) != int(manifest["total_count"]):
        raise ValueError("Duplicate vocabulary or total count mismatch")
    for prompt in prompts.values():
        if prompt["frame"] != "Neutral":
            continue
        path = directory / (prompt["id"] + ".log_probs.npy")
        metadata_path = directory / (prompt["id"] + ".meta.json")
        progress_path = directory / (prompt["id"] + ".progress.json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("context") != prompt["id"] or int(metadata.get("target_vocab_size", -1)) != len(words):
            raise ValueError("Array metadata mismatch: " + str(path))
        if metadata.get("prompt", "").rstrip() != prompt["text"].rstrip():
            raise ValueError("Array prompt text mismatch: " + str(path))
        progress = json.loads(progress_path.read_text())
        if not all(progress.get("sources", {}).get(kind, {}).get("done") for kind in ("1gram", "2gram")):
            raise ValueError("Score array is incomplete: " + str(path))
        scores = np.load(path, allow_pickle=False).astype(float)
        if scores.shape != (len(words),) or not np.isfinite(scores).all() or (scores > 0).any():
            raise ValueError("Invalid score array: " + str(path))
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(words), dtype=int)
        ranks[order] = np.arange(1, len(words) + 1)
        targets = {lookup[c["word"].strip().lower()] for c in prompt["candidates"]}
        selected = set(order[:top_n]) | targets if top_n else set(order)
        prompt["distribution"] = [{"word": words[i], "logp": float(scores[i]), "normalized": float(probabilities[i]), "rank": int(ranks[i])}
                                  for i in sorted(selected, key=lambda i: ranks[i])]
        prompt["supportSize"] = len(words)
        prompt["distributionCoverage"] = "Full support exported" if not top_n else "Top {} + experimental alternatives from full support".format(top_n)
        paths.extend([path, metadata_path, progress_path])
    return paths


def build_payload(results, manifest, log_probs_dir=None, top_n=50, vocab_dir=None):
    source_path, oof_path = manifest / "source_rows.csv", results / "cv_results/oof_predictions.csv"
    source, oof = read_csv(source_path), read_csv(oof_path)
    items = build_items(source, oof)
    dataset_summaries = {key: summarize([r for r in items if r["dataset"] == key]) for key, _ in DATASETS}
    verified = verify_tables(results, dataset_summaries)
    contexts = sorted({r["context"] for r in items if r["dataset"] == "novel_focus"})
    if len(contexts) != 16 or {r["dataset"] for r in items} != {key for key, _ in DATASETS}:
        raise ValueError("Expected ten datasets and sixteen novel-focus contexts")
    prompts, sources = build_prompts(source)
    provenance = [source_path, oof_path, results / "cv_results/fold_selections.csv"]
    if manifest.resolve() == (PIPELINE / "scoring_manifests/set_variant_qwen").resolve():
        provenance.extend(add_diagnostic(prompts, PIPELINE))
    if log_probs_dir:
        provenance.extend(add_full_distributions(prompts, log_probs_dir, top_n, vocab_dir))
    baseline_path = results / "advisor_summary/advisor_dataset_base_rate_scores.csv"
    baselines = {row["analysis_dataset_id"]: number(row["mean_oof_log_score"]) for row in read_csv(baseline_path)} if baseline_path.exists() else {}
    if baseline_path.exists():
        provenance.append(baseline_path)
    for metric in ["correlations", "log_scores"]:
        provenance.append(results / "linking_structure_tables" / (metric + "_by_dataset_and_linking_structure.csv"))
    grid = read_csv(results / "prediction_grid.csv")
    bounds = {variant: sorted({number(row["boundary"]) for row in grid if row["variant"] == variant}) for variant in ["top_k", "top_p"]}
    provenance.append(results / "prediction_grid.csv")
    return {"generated": datetime.now(timezone.utc).isoformat(), "datasets": [{"id": key, "label": label} for key, label in DATASETS],
        "models": MODELS, "contexts": contexts, "items": items, "sources": sources, "prompts": list(prompts.values()),
        "datasetSummaries": dataset_summaries, "contextSummaries": {c: summarize([r for r in items if r["context"] == c]) for c in contexts},
        "baselines": baselines, "folds": read_csv(results / "cv_results/fold_selections.csv"), "boundaries": bounds,
        "sourceRows": len(source), "verifiedCells": verified, "modelName": source[0]["model_name"], "revision": source[0]["model_revision"],
        "provenance": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in provenance]}


def preserve_distributions(payload, previous_html):
    """Reuse exported arrays for the same analysis when rebuilding the UI locally.

    Reject changed scientific inputs rather than silently attach stale scores.
    The source HTML need not have access to its original cluster input paths.
    """
    text = previous_html.read_text(encoding="utf-8")
    match = re.search(r'<script id="results-data" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError("Cannot read existing viewer data: " + str(previous_html))
    old = json.loads(match.group(1))
    available = [p for p in old["prompts"] if p.get("distribution")]
    if not available:
        return 0
    for field in ("modelName", "revision"):
        if old.get(field) != payload.get(field):
            raise ValueError("Model changed; rebuild with --log-probs-dir or explicitly --discard-distributions")
    for filename in ("source_rows.csv", "oof_predictions.csv", "prediction_grid.csv"):
        old_hashes = {p["sha256"] for p in old["provenance"] if Path(p["path"]).name == filename}
        new_hashes = {p["sha256"] for p in payload["provenance"] if Path(p["path"]).name == filename}
        if not old_hashes or old_hashes != new_hashes:
            raise ValueError("{} changed; rebuild with --log-probs-dir or explicitly --discard-distributions".format(filename))
    current = {p["id"]: p for p in payload["prompts"]}
    for prompt in available:
        target = current.get(prompt["id"])
        if target is None or target["text"] != prompt["text"] or target["frame"] != prompt["frame"]:
            raise ValueError("Cannot reuse scores for changed prompt: " + prompt["id"])
    for prompt in available:
        for field in ("distribution", "distributionCoverage", "supportSize"):
            if field in prompt:
                current[prompt["id"]][field] = prompt[field]
    # Retain the source-array hashes from the cluster, even when unavailable locally.
    seen = {(p["path"], p["sha256"]) for p in payload["provenance"]}
    for record in old["provenance"]:
        key = record["path"], record["sha256"]
        if key not in seen:
            payload["provenance"].append(record)
            seen.add(key)
    payload["distributionReuse"] = old.get("distributionReuse") or {
        "exportGenerated": old["generated"],
        "sourceHtmlSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "promptCount": len(available),
    }
    return len(available)


def render_fallback(payload):
    """Keep saved results readable when an HTML preview does not execute scripts."""
    output = ['<section class="panel"><h1>Focus Alternatives results</h1>',
              '<p id="viewer-status" role="status">Starting the interactive viewer. If this message remains, open index.html directly in a web browser; this preview may not run JavaScript.</p>',
              '<details><summary>Read saved dataset results without interactive charts</summary>']
    for metric, label in [("r", "Pearson correlation"), ("log", "Mean proper log score")]:
        output.append('<h2>{}</h2><div class="table-wrap"><table><thead><tr><th>Dataset</th>'.format(label))
        output.extend('<th>{}</th>'.format(escape(model)) for model in payload.get("models", []))
        output.append('</tr></thead><tbody>')
        for dataset in payload.get("datasets", []):
            output.append('<tr><th>{}</th>'.format(escape(dataset["label"])))
            for stats in payload["datasetSummaries"][dataset["id"]]:
                value = stats[metric]
                output.append('<td>{}</td>'.format('—' if not stats["n"] else 'N/A' if value is None else '{:.3f}'.format(value)))
            output.append('</tr>')
        output.append('</tbody></table></div>')
    output.append('<p>Higher is better for both measures. — indicates a structurally unavailable model; N/A indicates an undefined correlation.</p></details></section>')
    return ''.join(output)


def render_html(payload, template_dir):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return (template_dir / "viewer.html.in").read_text(encoding="utf-8").replace("/* VIEWER_CSS */", (template_dir / "viewer.css").read_text()).replace("/* VIEWER_JS */", (template_dir / "viewer.js").read_text()).replace("__VIEWER_FALLBACK__", render_fallback(payload)).replace("__VIEWER_DATA__", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PIPELINE / "results/set_variant_qwen")
    parser.add_argument("--manifest-dir", type=Path, default=PIPELINE / "scoring_manifests/set_variant_qwen")
    parser.add_argument("--output", type=Path, default=PIPELINE / "results_viewer/index.html")
    distributions = parser.add_mutually_exclusive_group()
    distributions.add_argument("--log-probs-dir", type=Path, help="Optional full neutral-prompt array directory; requires NumPy")
    distributions.add_argument("--reuse-distributions-from", type=Path, help="Reuse a previous HTML export with matching scientific inputs")
    distributions.add_argument("--discard-distributions", action="store_true", help="Explicitly allow replacing embedded distributions with the local candidate subset")
    parser.add_argument("--vocab-dir", type=Path, help="Local directory for vocabulary files relocated from Oscar")
    parser.add_argument("--top-candidates", type=int, default=50, help="Full-array export: top N plus targets; 0 exports all (large)")
    args = parser.parse_args()
    if args.top_candidates < 0:
        parser.error("--top-candidates must be nonnegative")
    payload = build_payload(args.results_dir, args.manifest_dir, args.log_probs_dir, args.top_candidates, args.vocab_dir)
    if not args.log_probs_dir and not args.discard_distributions:
        previous = args.reuse_distributions_from or args.output
        if args.reuse_distributions_from or previous.exists():
            count = preserve_distributions(payload, previous)
            print("Preserved distributions for {} prompts from {}".format(count, previous))
    html = render_html(payload, PIPELINE / "results_viewer")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    neutral = [p for p in payload["prompts"] if p["frame"] == "Neutral"]
    print("Neutral-prompt distribution coverage: {}/{}".format(sum(bool(p["distribution"]) for p in neutral), len(neutral)))
    print("Built {} ({:,} bytes); {} units, {} datasets, {} contexts; verified {} saved metric cells.".format(
        args.output, len(html.encode()), len(payload["items"]), len(payload["datasets"]), len(payload["contexts"]), payload["verifiedCells"]))


if __name__ == "__main__":
    main()
