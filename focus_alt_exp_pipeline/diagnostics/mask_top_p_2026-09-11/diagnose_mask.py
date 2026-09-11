#!/usr/bin/env python3
"""Read-only score audit and 500-ordering diagnostic replication; never loads Qwen.

Run with the pipeline's Python environment. Outputs are confined to --output-dir.
The original prediction grid remains authoritative. See REPORT.md for interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mask-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROMPT_ID = "prompt_7ebf909e4b947d98da36"
TARGETS = ["bandana", "handkerchief", "napkins", "gloves", "wallet", "candy"]
BOUNDARIES = [0.6, 0.7, 0.8, 0.9, 0.95]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--scores", type=Path, default=Path("/users/ljohnst7/data/ljohnst7/ngrams/qwen_set_variant_log_probs"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    pipe = args.repo / "focus_alt_exp_pipeline"
    if out == args.scores.resolve() or (pipe / "results").resolve() in out.parents:
        raise ValueError("Choose a new diagnostic directory outside original results/scores")
    out.mkdir(parents=True, exist_ok=True)
    tables, plots = out / "tables", out / "plots"
    tables.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    paths = {name: args.scores / filename for name, filename in {
        "array": f"{PROMPT_ID}.log_probs.npy", "metadata": f"{PROMPT_ID}.meta.json",
        "progress": f"{PROMPT_ID}.progress.json", "vocab_manifest": "vocab_manifest.json",
    }.items()}
    paths.update({
        "source_rows": pipe / "scoring_manifests/set_variant_qwen/source_rows.csv",
        "prompts": pipe / "scoring_manifests/set_variant_qwen/prompts.csv",
        "prediction_grid": pipe / "results/set_variant_qwen/prediction_grid.csv",
        "fold_selections": pipe / "results/set_variant_qwen/cv_results/fold_selections.csv",
        "prediction_code": pipe / "code/build_set_variant_prediction_grid.py",
        "evaluation_code": pipe / "code/evaluate_set_variant_grid.py",
        "precompute_code": pipe / "code/precompute_qwen_vocab_log_probs.py",
        "postprocessing_wrapper": pipe / "cluster/run_set_variant_postprocessing.sh",
    })
    manifest = json.loads(paths["vocab_manifest"].read_text())
    meta = json.loads(paths["metadata"].read_text())
    progress = json.loads(paths["progress"].read_text())
    paths.update({f"vocab_{s['name']}": Path(s["path"]) for s in manifest["sources"]})
    tokenizer_path = Path(meta["model_path"]) / "tokenizer.json"
    if tokenizer_path.exists():
        paths["tokenizer"] = tokenizer_path
    before = {k: digest(p) for k, p in paths.items()}
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    tokens, sources, source_paths, source_lines = [], [], [], []
    source_validation = []
    for s in manifest["sources"]:
        with Path(s["path"]).open(encoding="utf-8") as stream:
            values = [line.rstrip("\n") for line in stream]
        check(s["offset"] == len(tokens), f"Noncontiguous/incorrect offset: {s['name']}")
        check(s["count"] == len(values), f"Vocabulary count mismatch: {s['name']}")
        done = progress["sources"].get(s["name"], {})
        check(done.get("done") is True, f"Incomplete source: {s['name']}")
        check(done.get("last_line") == len(values), f"Progress count mismatch: {s['name']}")
        source_validation.append({**s, "actual_count": len(values), "progress": done})
        tokens.extend(values)
        sources.extend([s["name"]] * len(values))
        source_paths.extend([s["path"]] * len(values))
        source_lines.extend(range(1, len(values) + 1))
    lookup = {t.strip().lower(): i for i, t in enumerate(tokens) if t.strip()}
    check(len(lookup) == len(tokens), "Duplicate/empty normalized vocabulary entries")
    saved = np.load(paths["array"], mmap_mode="r")
    check(saved.shape == (121301,), f"Expected 121301 scores, found {saved.shape}")
    check(str(saved.dtype) == manifest["dtype"] == "float32", "Unexpected dtype")
    check(len(tokens) == manifest["total_count"] == len(saved), "Vocabulary/array length mismatch")
    check(bool(np.isfinite(saved).all()), "Nonfinite saved scores")
    check(bool((saved <= 0).all()), "Positive saved log scores")
    check(meta["context"] == PROMPT_ID, "Metadata prompt ID mismatch")
    check(meta["target_vocab_size"] == meta["requested_target_vocab_size"] == len(saved), "Metadata length mismatch")
    check(Path(meta["manifest_path"]).resolve() == paths["vocab_manifest"].resolve(), "Metadata manifest path mismatch")
    check(Path(meta["output_path"]).resolve() == paths["array"].resolve(), "Metadata array path mismatch")
    check("prompt_" + hashlib.sha256(meta["prompt"].encode()).hexdigest()[:20] == PROMPT_ID, "Prompt content hash mismatch")
    all_rows = pd.read_csv(paths["source_rows"])
    rows = all_rows.loc[all_rows.prompt_id.eq(PROMPT_ID)].copy()
    check(len(rows) == 30, "Expected 30 mask source rows")
    check(bool(rows.generation_prompt.eq(meta["prompt"]).all()), "Source prompt text mismatch")
    prompts = pd.read_csv(paths["prompts"])
    prompt_rows = prompts.loc[prompts.prompt_id.eq(PROMPT_ID)]
    check(len(prompt_rows) == 1 and bool(prompt_rows.generation_prompt.eq(meta["prompt"]).all()), "Prompt manifest mismatch")
    required = set(pd.concat([rows.trigger, rows["query"]]).str.strip().str.lower())
    check(required == set(TARGETS), "Unexpected mask target set")
    check(required.issubset(lookup), "Missing experimental alternatives")
    spec = importlib.util.spec_from_file_location("saved_grid_implementation", paths["prediction_code"])
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    code_tokens, code_lookup = implementation._load_vocab(paths["vocab_manifest"])
    check(code_tokens == tokens and code_lookup == lookup, "Mapping differs from prediction implementation")
    validation = {"ready": not errors, "errors": errors, "prompt_id": PROMPT_ID,
                  "prompt": meta["prompt"], "array_shape": list(saved.shape), "dtype": str(saved.dtype),
                  "finite_count": int(np.isfinite(saved).sum()), "unique_normalized_candidates": len(lookup),
                  "source_rows": len(rows), "candidate_sources": source_validation,
                  "target_indices_zero_based": {t: lookup.get(t) for t in TARGETS}}
    dump(out / "validation.json", validation)
    if errors:
        raise RuntimeError("Validation failed before interpreting distribution: " + "; ".join(errors))
    print("Validated 121301 candidates; all finite, aligned, and covered.", flush=True)

    # Float64 conversion and operations exactly match build_prediction_grid.
    log_probs = np.asarray(saved, dtype=np.float64)
    weights = np.exp(log_probs - log_probs.max())
    probabilities = weights / weights.sum()
    raw_probabilities = np.exp(log_probs)
    assert np.all(probabilities > 0) and np.isclose(probabilities.sum(), 1)
    order = np.argsort(-probabilities, kind="stable")  # ties: ascending original index
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    cumulative = probabilities[order].cumsum()
    cumulative_by_index = np.empty(len(order))
    cumulative_by_index[order] = cumulative
    candidates = pd.DataFrame({"vocab_index_zero_based": np.arange(len(tokens)), "candidate": tokens,
        "vocabulary_source": sources, "vocabulary_path": source_paths, "source_line_one_based": source_lines,
        "saved_log_score": log_probs, "unnormalized_continuation_probability": raw_probabilities,
        "normalized_sampling_probability": probabilities, "rank_one_based": ranks,
        "descending_cumulative_probability": cumulative_by_index})
    candidates.iloc[order[:100]].to_csv(tables / "top_100_candidates.csv", index=False)
    ti = np.array([lookup[t] for t in TARGETS])
    mask_index = lookup["mask"]
    target_table = candidates.iloc[ti].copy()
    target_table["theoretical_inclusion_p06"] = probabilities[ti] / (probabilities[ti] + probabilities[mask_index])
    target_table["expected_inclusions_in_500_p06"] = 500 * target_table.theoretical_inclusion_p06
    target_table.to_csv(tables / "experimental_alternatives.csv", index=False)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    mass_table = pd.DataFrame([{"mass_boundary": b, "highest_ranked_candidate_count": int(np.searchsorted(cumulative, b) + 1),
        "attained_mass": float(cumulative[np.searchsorted(cumulative, b)])} for b in thresholds])
    mass_table.to_csv(tables / "descending_mass_thresholds.csv", index=False)
    source_mass = candidates.groupby("vocabulary_source").agg(candidate_count=("candidate", "size"),
        normalized_probability_mass=("normalized_sampling_probability", "sum"))
    source_mass.to_csv(tables / "vocabulary_source_mass.csv")
    # Representation inspection: exact strings are unique; phrases are separate candidate events.
    related = candidates.candidate.str.contains(r"\b(?:mask|masks|bandana|handkerchief|napkins|gloves|wallet|candy)\b", regex=True)
    candidates.loc[related].sort_values("rank_one_based").to_csv(tables / "related_candidate_strings.csv", index=False)

    grid = pd.read_csv(paths["prediction_grid"])
    original = grid.loc[grid.analysis_dataset_id.eq("novel_focus") & grid.analysis_unit_id.isin(rows.item_id)].copy()
    human_columns = ["item_id", "trigger", "query", "human_yes", "human_total"]
    original = original.merge(rows[human_columns], left_on="analysis_unit_id", right_on="item_id", validate="many_to_one")
    assert len(original) == 30 * 17
    selected = original.loc[original.variant.eq("top_p") & original.boundary.eq(0.6)]
    assert len(selected) == 30 and selected.set_probability.eq(0).all()
    assert selected.conjunction_probability.eq(0).all()
    assert selected.disjunction_probability.eq(selected.ordering_probability).all()
    original.to_csv(tables / "saved_mask_grid_authoritative.csv", index=False)
    folds = pd.read_csv(paths["fold_selections"])
    assert folds.loc[folds.variant.eq("top_p"), "selected_boundary"].eq(0.6).all()

    print("Sampling 500 weighted orderings, retained length 32768, seed", args.seed, flush=True)
    rng = np.random.default_rng(args.seed)
    sampled = implementation._sample_prefixes(probabilities, num_reps=500, prefix_size=32768, rng=rng)
    ordered_targets = sorted(ti.tolist())
    positions = implementation._target_positions(sampled, probabilities, target_indices=ordered_targets, rng=rng)
    columns = {t: ordered_targets.index(lookup[t]) for t in TARGETS}
    prefix_mass = probabilities[sampled].cumsum(axis=1)
    assert np.all(prefix_mass[:, -1] >= max(BOUNDARIES))
    lengths = {b: np.argmax(prefix_mass >= b, axis=1) + 1 for b in BOUNDARIES}
    assert all(len(np.unique(rep)) == sampled.shape[1] for rep in sampled)
    # At 0.6, total non-mask mass is <0.6, and mask mass alone exceeds 0.6.
    assert probabilities[mask_index] > 0.6 and 1 - probabilities[mask_index] < 0.6
    assert np.all(sampled[np.arange(500), lengths[0.6] - 1] == mask_index)
    summaries, length_rows, inclusion_rows = [], [], []
    for b in BOUNDARIES:
        lens = lengths[b]
        assert np.all(prefix_mass[np.arange(500), lens - 1] >= b)
        assert np.all((lens == 1) | (prefix_mass[np.arange(500), np.maximum(lens - 2, 0)] < b))
        summaries.append({"boundary": b, "n": 500, "mean": lens.mean(), "std": lens.std(ddof=1),
            "min": int(lens.min()), **{f"q{q:02}": float(np.quantile(lens, q / 100)) for q in [5, 25, 50, 75, 95]},
            "max": int(lens.max())})
        length_rows.extend({"ordering_zero_based": rep, "boundary": b, "prefix_length": int(lens[rep]),
            "attained_mass": float(prefix_mass[rep, lens[rep] - 1])} for rep in range(500))
        for t in TARGETS:
            count = int((positions[:, columns[t]] < lens).sum())
            saved_rates = original.loc[original.variant.eq("top_p") & original.boundary.eq(b) & original["query"].eq(t), "set_probability"].unique()
            assert len(saved_rates) == 1
            inclusion_rows.append({"candidate": t, "boundary": b, "diagnostic_count": count,
                "diagnostic_n": 500, "diagnostic_inclusion_frequency": count / 500,
                "saved_authoritative_inclusion_probability": saved_rates[0]})
    pd.DataFrame(summaries).to_csv(tables / "prefix_length_summary.csv", index=False)
    length_df = pd.DataFrame(length_rows)
    length_df.to_csv(tables / "prefix_lengths_all_500.csv", index=False)
    length_df.groupby(["boundary", "prefix_length"]).size().rename("count").reset_index().to_csv(tables / "prefix_length_distribution.csv", index=False)
    inclusion = pd.DataFrame(inclusion_rows)
    inclusion.to_csv(tables / "target_inclusion_by_boundary.csv", index=False)
    counts = np.bincount(np.concatenate([sampled[r, :lengths[0.6][r]] for r in range(500)]), minlength=len(tokens))
    frequent = candidates.loc[counts > 0].copy()
    frequent["prefix_count"] = counts[counts > 0]
    frequent["prefix_inclusion_frequency"] = frequent.prefix_count / 500
    frequent["theoretical_inclusion_p06"] = probabilities[counts > 0] / (probabilities[counts > 0] + probabilities[mask_index])
    frequent.loc[frequent.candidate.eq("mask"), "theoretical_inclusion_p06"] = 1.0
    frequent.sort_values(["prefix_count", "rank_one_based"], ascending=[False, True]).to_csv(tables / "p06_prefix_occupants.csv", index=False)
    # First examples of lengths 1..max; first five if unusually many lengths.
    reps = [int(np.flatnonzero(lengths[0.6] == n)[0]) for n in np.unique(lengths[0.6])[:5]]
    representative_rows = []
    for rep in reps:
        for pos, idx in enumerate(sampled[rep, :lengths[0.6][rep]]):
            representative_rows.append({"ordering_zero_based": rep, "boundary": 0.6, "position_one_based": pos + 1,
                "vocab_index_zero_based": int(idx), "candidate": tokens[idx],
                "normalized_sampling_probability": float(probabilities[idx]), "cumulative_mass": float(prefix_mass[rep, pos])})
    pd.DataFrame(representative_rows).to_csv(tables / "representative_p06_prefixes.csv", index=False)
    comparisons = []
    for row in rows.itertuples():
        x, y = row.trigger, row.query
        diagnostic = implementation._probability_records(sampled, probabilities,
            query_position=positions[:, columns[y]], trigger_position=positions[:, columns[x]],
            k_values=[], p_values=BOUNDARIES)
        for d in diagnostic:
            # Independently check the implementation's Set event against prefix membership.
            assert d["set_probability"] == float((positions[:, columns[y]] < lengths[d["boundary"]]).mean())
        saved_row = selected.loc[selected.item_id.eq(row.item_id)].iloc[0]
        comparisons.append({"trigger": x, "query": y, "human_yes": row.human_yes, "human_total": row.human_total,
            "human_rate": row.human_rate, "trigger_array_weight": float(weights[lookup[x]]),
            "query_array_weight": float(weights[lookup[y]]),
            "theoretical_ordering_probability": float(weights[lookup[y]] / (weights[lookup[x]] + weights[lookup[y]])),
            "saved_authoritative_ordering_probability": saved_row.ordering_probability,
            "saved_authoritative_set_p06": saved_row.set_probability,
            "diagnostic_ordering_probability": diagnostic[0]["ordering_probability"],
            "diagnostic_set_p06": diagnostic[0]["set_probability"]})
    pairs = pd.DataFrame(comparisons)
    pairs.to_csv(tables / "pairwise_ordering_comparison.csv", index=False)

    tokenizer_audit = {"status": "unavailable", "model_inference_performed": False}
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        prepared_prompt = meta["prompt"].rstrip() + " "
        prompt_ids = tokenizer.encode(prepared_prompt, add_special_tokens=False).ids
        token_rows = []
        for t in ["mask", *TARGETS]:
            encoded = tokenizer.encode(" " + t, add_special_tokens=False)
            assembled = prompt_ids + encoded.ids
            joined = tokenizer.encode(prepared_prompt + t, add_special_tokens=False).ids
            token_rows.append({"candidate": t, "continuation_token_count": len(encoded.ids),
                "continuation_ids": json.dumps(encoded.ids), "continuation_tokens": json.dumps(encoded.tokens),
                "separate_encoding_decoded_tail": tokenizer.decode(assembled)[-45:],
                "ordinary_single_space_decoded_tail": tokenizer.decode(joined)[-45:],
                "same_as_single_space_encoding": assembled == joined})
        pd.DataFrame(token_rows).to_csv(tables / "tokenizer_boundary_audit.csv", index=False)
        tokenizer_audit = {"status": "inspected_local_tokenizer_only", "model_inference_performed": False,
            "prepared_prompt_tail_repr": repr(prepared_prompt[-35:]),
            "prepared_prompt_last_token_ids": prompt_ids[-5:],
            "prepared_prompt_last_tokens": [tokenizer.id_to_token(i) for i in prompt_ids[-5:]],
            "qualification": "Verifies current scorer's token construction, not historical execution or score changes under another construction."}
    dump(out / "tokenizer_audit.json", tokenizer_audit)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    top = candidates.iloc[order[:15]].iloc[::-1]
    axes[0].barh(top.candidate, top.normalized_sampling_probability * 100, color="#377eb8")
    axes[0].set(xlabel="Probability normalized over scored support (%)", title="Mask prompt: leading candidates")
    for j, p in enumerate(top.normalized_sampling_probability):
        axes[0].text(p * 100 + 0.5, j, f"{p * 100:.2f}%", va="center", fontsize=8)
    axes[0].set_xlim(0, 76)
    axes[1].plot(np.arange(1, len(order) + 1), cumulative)
    for b in thresholds:
        n = int(np.searchsorted(cumulative, b) + 1)
        axes[1].scatter(n, cumulative[n - 1], s=20)
        axes[1].axhline(b, color="gray", alpha=0.25, linewidth=0.8)
    axes[1].set(xscale="log", xlabel="Candidates in descending probability order", ylabel="Cumulative normalized probability",
        title="Mass concentration (sorted support, not sampled prefixes)", ylim=(0.45, 1.01))
    fig.savefig(plots / "probability_mass.png", dpi=180)
    fig.savefig(plots / "probability_mass.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].boxplot([lengths[b] for b in BOUNDARIES], tick_labels=[str(b) for b in BOUNDARIES], showfliers=True)
    axes[0].set(yscale="log", xlabel="Top-p boundary", ylabel="Sampled prefix length (log scale)", title="Diagnostic replication: 500 orderings, seed 7")
    for t in TARGETS:
        sub = inclusion.loc[inclusion.candidate.eq(t)]
        line, = axes[1].plot(sub.boundary, sub.diagnostic_inclusion_frequency, marker="o", label=t)
        axes[1].scatter(sub.boundary, sub.saved_authoritative_inclusion_probability, marker="x", color=line.get_color(), s=50)
    axes[1].set(xlabel="Top-p boundary", ylabel="Inclusion frequency", ylim=(-0.035, 1.04),
        title="Circles: diagnostic; crosses: authoritative saved grid")
    axes[1].legend(fontsize=8)
    fig.savefig(plots / "sampled_prefixes.png", dpi=180)
    fig.savefig(plots / "sampled_prefixes.pdf")
    plt.close(fig)
    example = pairs.loc[pairs.trigger.eq("napkins") & pairs["query"].isin(["bandana", "wallet"])].set_index("query").loc[["bandana", "wallet"]]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    locations = np.arange(2)
    for j, (column, label, color) in enumerate([
        ("human_rate", "Human exclusion", "#777777"),
        ("saved_authoritative_set_p06", "Saved Set (p=0.6)", "#e69f00"),
        ("saved_authoritative_ordering_probability", "Saved Ordering", "#377eb8"),
        ("theoretical_ordering_probability", "Theoretical Ordering", "#009e73")]):
        ax.bar(locations + (j - 1.5) * 0.19, example[column], width=0.18, label=label, color=color)
    ax.set(xticks=locations, xticklabels=["napkins → bandana\nhuman 5/5", "napkins → wallet\nhuman 0/4"],
        ylabel="Exclusion probability / observed rate", ylim=(0, 1.2), title="High relative rank can coexist with negligible prefix inclusion")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    fig.savefig(plots / "worked_examples.png", dpi=180)
    fig.savefig(plots / "worked_examples.pdf")
    plt.close(fig)

    total_target_mass = float(probabilities[ti].sum())
    any_target_inclusion = total_target_mass / (probabilities[mask_index] + total_target_mass)
    summary = {"replication": "diagnostic, not exact reproduction of shared-RNG saved grid", "seed": args.seed,
        "num_orderings": 500, "retained_prefix_size": 32768, "bit_generator": type(rng.bit_generator).__name__,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "raw_exp_log_score_sum": float(raw_probabilities.sum()), "shifted_weight_sum": float(weights.sum()),
        "max_log_score": float(log_probs.max()), "minimum_normalized_probability": float(probabilities.min()),
        "normalization_sum": float(probabilities.sum()), "target_mass": total_target_mass,
        "mask_normalized_probability": float(probabilities[mask_index]), "mask_raw_continuation_probability": float(raw_probabilities[mask_index]),
        "theoretical_any_target_in_p06": float(any_target_inclusion),
        "probability_no_target_in_any_of_500_p06": float((1 - any_target_inclusion) ** 500),
        "diagnostic_p06_orderings_with_any_target": int(np.any(positions < lengths[0.6][:, None], axis=1).sum()),
        "minimum_retained_prefix_mass": float(prefix_mass[:, -1].min()),
        "representative_ordering_indices": reps, "mask_cv_fold": sorted(selected.cv_fold.unique().tolist()),
        "all_ten_saved_top_p_fold_selections": 0.6,
        "checks": ["offset/count/unique vocabulary", "metadata/prompt hash/source prompt identity", "all scores finite and positive after normalization",
                   "prefix without replacement", "all 500 reach every boundary", "shortest-prefix boundary checks",
                   "every p06 prefix ends at mask", "direct imported implementation agrees with diagnostic Set events",
                   "saved grid has 30 p06 mask zeros and conjunction/disjunction identities"]}
    dump(out / "summary.json", summary)
    after = {k: digest(p) for k, p in paths.items()}
    assert before == after, "An input changed during diagnosis"
    dump(out / "input_provenance.json", {"inputs_unchanged_after_run": True, "files": [
        {"role": k, "path": str(p), "sha256": before[k], "size_bytes": p.stat().st_size} for k, p in paths.items()],
        "diagnostic_script_sha256": digest(__file__)})
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(inclusion.to_string(index=False))


if __name__ == "__main__":
    main()
