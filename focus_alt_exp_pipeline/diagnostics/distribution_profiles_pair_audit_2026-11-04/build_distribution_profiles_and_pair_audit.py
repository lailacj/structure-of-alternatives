#!/usr/bin/env python3
"""Plot complete corrected candidate distributions and audit frozen pair scores.

Descriptive analysis only: no LM scoring, new samples, fitting, or hypothesis
tests. Imports the saved entropy analysis solely for its strict input checks.
Requires NumPy and Matplotlib. Choose a fresh --output-dir for each rerun.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shlex
import socket
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODELS = {"set": "Set Top-K", "ordering": "Ordering",
          "conjunction": "Conjunction Top-K", "disjunction": "Disjunction Top-K"}
DISPLAY = {"set": "Set", "ordering": "Ordering", "conjunction": "Conjunction", "disjunction": "Disjunction"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6B46A5"]
EPSILON = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def proper_score(y, p):
    require(math.isfinite(y) and math.isfinite(p) and 0 <= y <= 1 and 0 <= p <= 1,
            "Invalid human/model probability")
    clipped = min(1 - EPSILON, max(EPSILON, p))
    return y * math.log(clipped) + (1 - y) * math.log1p(-clipped)


def rank_text(alternative):
    lo, hi = alternative["rank_first"], alternative["rank_last"]
    return f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,} (tie)"


def load_inputs(args):
    validator_path = args.entropy_analysis_dir / "analyze_entropy_ordering_advantage.py"
    spec = importlib.util.spec_from_file_location("saved_entropy_validation", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    entropy_rows, validation = validator.analyze(argparse.Namespace(
        repo_root=args.repo_root, results_dir=args.results_dir, log_probs_dir=args.log_probs_dir))
    previous = json.loads((args.entropy_analysis_dir / "provenance_validation.json").read_text())
    for name, source_record in validation["inputs"].items():
        require(source_record == previous["inputs"][name], f"Previously frozen input changed: {name}")
    saved_entropy_path = args.entropy_analysis_dir / "entropy_ordering_advantage.csv"
    require(record(saved_entropy_path)["sha256"] == previous["output_csv"]["sha256"], "Saved entropy table changed")
    saved_entropy = {row["context"]: row for row in read_csv(saved_entropy_path)}
    for row in entropy_rows:
        require(row["entropy_bits"] == float(saved_entropy[row["context"]]["entropy_bits"]), "Entropy reproduction mismatch")
    source = [row for row in read_csv(args.results_dir / "inputs/source_rows.csv") if row["dataset_family"] == "novel_focus"]
    oof = [row for row in read_csv(args.results_dir / "cv_results/oof_predictions.csv")
           if row["dataset_family"] == "novel_focus" and row["variant"] == "top_k"]
    frozen = {row["analysis_unit_id"]: row for row in oof}
    score_table = {row["Context"]: row for row in read_csv(args.results_dir / "linking_structure_tables/log_scores_by_focus_context_and_linking_structure.csv")}
    parent = json.loads((args.results_dir / "inputs/parent_score_artifact_provenance.json").read_text())
    manifest, lookup, _ = validator.load_vocabulary(args.log_probs_dir, parent)
    tokens = [None] * len(lookup)
    for token, index in lookup.items():
        tokens[index] = token
    return source, frozen, score_table, entropy_rows, validation, manifest, lookup, tokens, validator_path


def build_data(args):
    source, frozen, score_table, entropy_rows, validation, manifest, lookup, tokens, validator_path = load_inputs(args)
    entropy = {row["context"]: row["entropy_bits"] for row in entropy_rows}
    profiles, alternative_rows, pair_rows, summaries, query_rows, head_rows = {}, [], [], [], [], []
    max_sum_error = 0.0
    for context in sorted(entropy):
        rows = [row for row in source if row["context_id"] == context]
        array_path = Path(rows[0]["source_score_file"])
        raw = array_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == rows[0]["neutral_score_array_sha256"], f"{context}: array changed")
        scores = np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float64)
        weights = np.exp(scores - scores.max())
        p = weights / weights.sum(dtype=np.float64)
        require(len(p) == 121301 and bool(np.isfinite(p).all()) and bool((p > 0).all()), f"{context}: invalid full support")
        order = np.argsort(-scores, kind="stable")
        sorted_scores, sorted_p = scores[order], p[order]
        cumulative = np.cumsum(sorted_p, dtype=np.float64)
        alternatives = []
        for word in sorted({row["query"] for row in rows}):
            index = lookup[word.strip().lower()]
            first = int(np.searchsorted(-sorted_scores, -scores[index], side="left")) + 1
            last = int(np.searchsorted(-sorted_scores, -scores[index], side="right"))
            q_pairs = [row for row in rows if row["query"] == word]
            set_predictions = {float(frozen[row["item_id"]]["set_probability"]) for row in q_pairs}
            require(len(set_predictions) == 1 and len(q_pairs) == 5, f"{context}/{word}: Set varies across triggers")
            alternatives.append({"context": context, "prompt_id": rows[0]["prompt_id"], "alternative": word,
                                 "candidate_index": index, "rank_first": first, "rank_last": last,
                                 "rank_mid": (first + last) / 2, "probability_full_support": float(p[index]),
                                 "saved_continuation_log_score": float(scores[index]),
                                 "cumulative_mass_at_rank_end": float(cumulative[last - 1]),
                                 "frozen_set_inclusion_probability": next(iter(set_predictions))})
        alternatives.sort(key=lambda row: (row["rank_mid"], row["alternative"]))
        alt_map = {row["alternative"]: row for row in alternatives}
        alternative_rows.extend(alternatives)
        for rank, index in enumerate(order[:20], 1):
            head_rows.append({"context": context, "rank_position": rank, "candidate": tokens[index],
                              "probability_full_support": float(p[index]), "is_experimental_alternative": tokens[index] in alt_map})
        context_pairs = []
        for row in sorted(rows, key=lambda row: (alt_map[row["trigger"]]["rank_mid"], alt_map[row["query"]]["rank_mid"])):
            saved = frozen[row["item_id"]]
            y = float(row["human_rate"])
            pair = {"context": context, "item_id": row["item_id"], "trigger": row["trigger"], "query": row["query"],
                    "cv_fold": int(saved["cv_fold"]), "selected_k": float(saved["boundary"]),
                    "human_exclusion_rate": y, "human_yes": float(row["human_yes"]), "human_total": float(row["human_total"]),
                    "trigger_probability_full_support": alt_map[row["trigger"]]["probability_full_support"],
                    "query_probability_full_support": alt_map[row["query"]]["probability_full_support"],
                    "trigger_rank_mid": alt_map[row["trigger"]]["rank_mid"], "query_rank_mid": alt_map[row["query"]]["rank_mid"]}
            for model in MODELS:
                probability = float(saved[model + "_probability"])
                pair[model + "_probability"] = probability
                pair[model + "_log_score"] = proper_score(y, probability)
                pair[model + "_clipping_applied"] = probability < EPSILON or probability > 1 - EPSILON
                pair[model + "_endpoint_prediction"] = probability in (0.0, 1.0)
            require(pair["conjunction_probability"] <= min(pair["set_probability"], pair["ordering_probability"]) + 1e-12
                    and pair["disjunction_probability"] + 1e-12 >= max(pair["set_probability"], pair["ordering_probability"]),
                    f"{row['item_id']}: invalid conjunction/disjunction ordering")
            for i, a in enumerate(MODELS):
                for b in list(MODELS)[i + 1:]:
                    pair[a + "_minus_" + b + "_log_score"] = pair[a + "_log_score"] - pair[b + "_log_score"]
            pair["ordering_advantage"] = -pair["set_minus_ordering_log_score"]
            best_pair_score = max(pair[model + "_log_score"] for model in MODELS)
            pair["highest_scoring_structure_on_pair"] = " | ".join(DISPLAY[model] for model in MODELS
                if math.isclose(pair[model + "_log_score"], best_pair_score, rel_tol=0, abs_tol=1e-12))
            context_pairs.append(pair)
        by_magnitude = sorted(context_pairs, key=lambda pair: abs(pair["set_minus_ordering_log_score"]), reverse=True)
        for position, pair in enumerate(by_magnitude, 1):
            pair["absolute_set_ordering_difference_rank_within_context"] = position
        pair_rows.extend(context_pairs)
        summary = {"context": context, "prompt_id": rows[0]["prompt_id"], "n_candidates": len(p),
                   "n_trigger_query_pairs": len(rows), "entropy_bits": entropy[context],
                   "mass_top_20_candidates": float(sorted_p[:20].sum()),
                   "mass_six_experimental_alternatives": math.fsum(row["probability_full_support"] for row in alternatives),
                   "best_alternative_rank_first": min(row["rank_first"] for row in alternatives),
                   "worst_alternative_rank_last": max(row["rank_last"] for row in alternatives),
                   "highest_probability_candidate": tokens[order[0]], "highest_candidate_probability": float(sorted_p[0])}
        totals = {}
        for model, column in MODELS.items():
            total = math.fsum(pair[model + "_log_score"] for pair in context_pairs)
            error = abs(total - float(score_table[context][column]))
            require(error < 1e-10, f"{context}: frozen summed score mismatch for {model}: {error}")
            max_sum_error = max(max_sum_error, error)
            totals[model] = total
            summary[model + "_log_score"] = total
            endpoints = [pair for pair in context_pairs if pair[model + "_endpoint_prediction"]]
            summary[model + "_endpoint_pair_count"] = len(endpoints)
            summary[model + "_endpoint_pairs_log_score_sum"] = math.fsum(pair[model + "_log_score"] for pair in endpoints)
        best = max(totals.values())
        summary["highest_scoring_of_four"] = " | ".join(DISPLAY[model] for model, value in totals.items()
            if math.isclose(value, best, rel_tol=0, abs_tol=1e-12))
        ordered_totals = sorted(totals.values(), reverse=True)
        summary["best_minus_second_best_log_score"] = ordered_totals[0] - ordered_totals[1]
        summary["set_advantage"] = totals["set"] - totals["ordering"]
        summary["ordering_advantage"] = -summary["set_advantage"]
        absolute_total = math.fsum(abs(pair["set_minus_ordering_log_score"]) for pair in context_pairs)
        summary["top_3_pairs_share_of_absolute_set_ordering_difference"] = (
            math.fsum(abs(pair["set_minus_ordering_log_score"]) for pair in by_magnitude[:3]) / absolute_total if absolute_total else 0)
        for word in [row["alternative"] for row in alternatives]:
            pairs = [pair for pair in context_pairs if pair["query"] == word]
            query_rows.append({"context": context, "query": word, "n_triggers": len(pairs),
                               "query_rank_mid": alt_map[word]["rank_mid"],
                               "human_exclusion_rate_min": min(pair["human_exclusion_rate"] for pair in pairs),
                               "human_exclusion_rate_max": max(pair["human_exclusion_rate"] for pair in pairs),
                               "human_exclusion_rate_range": max(pair["human_exclusion_rate"] for pair in pairs) - min(pair["human_exclusion_rate"] for pair in pairs),
                               "frozen_set_probability": pairs[0]["set_probability"],
                               "ordering_probability_min": min(pair["ordering_probability"] for pair in pairs),
                               "ordering_probability_max": max(pair["ordering_probability"] for pair in pairs),
                               "set_minus_ordering_log_score_sum": math.fsum(pair["set_minus_ordering_log_score"] for pair in pairs)})
        summary["max_absolute_pair_contribution"] = abs(by_magnitude[0]["set_minus_ordering_log_score"])
        summaries.append(summary)
        profiles[context] = {"ranks": np.arange(1, len(p) + 1), "p": sorted_p, "cumulative": cumulative,
                             "alternatives": alternatives, "pairs": context_pairs, "summary": summary,
                             "prompt": rows[0]["generation_prompt"]}
    require(len(alternative_rows) == 96 and len(pair_rows) == 480 and len(query_rows) == 96, "Incomplete audit coverage")
    validation["profile_and_pair_audit"] = {"n_contexts": len(profiles), "n_alternative_rows": len(alternative_rows),
                                           "n_pair_rows": len(pair_rows), "n_query_audit_rows": len(query_rows),
                                           "all_64_frozen_context_model_sums_match": True,
                                           "maximum_absolute_frozen_sum_difference": max_sum_error,
                                           "set_prediction_constant_across_five_triggers_for_each_query": True,
                                           "rank_ties": [row for row in alternative_rows if row["rank_first"] != row["rank_last"]]}
    return profiles, alternative_rows, pair_rows, summaries, query_rows, head_rows, validation, validator_path


# Plotting helpers are below; every curve uses full-support probabilities.


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#86939B", "axes.labelcolor": "#243640",
                         "text.color": "#243640", "pdf.fonttype": 42,
                         "pdf.compression": 9, "path.simplify_threshold": 0.3})


def label_checks(fig, labels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer=renderer) for label in labels]
    for box in boxes:
        require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1,
                "Alternative label is clipped by the figure")
    require(not any(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))),
            "Alternative text labels overlap")


def profile_panel(fig, ax, gutter, profile, limits, compact=False):
    ax.loglog(profile["ranks"], profile["p"], color="#486775", lw=1.15 if compact else 1.4)
    ax.set_xlim(1, limits[0])
    ax.set_ylim(*limits[1])
    ax.axvline(20, color="#A6ACB0", linestyle=(0, (3, 3)), lw=0.7, zorder=1)
    ax.grid(alpha=0.18, lw=0.6)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set_yticks([10.0 ** exponent for exponent in (range(0, -26, -5) if limits[0] == 121301 else range(0, -9, -2))
                  if limits[1][0] <= 10.0 ** exponent <= limits[1][1]])
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(labelsize=8 if compact else 9)
    gutter.set_axis_off()
    labels = []
    for i, alternative in enumerate(profile["alternatives"]):
        slot = 0.91 - i * 0.16
        xy = (alternative["rank_mid"], alternative["probability_full_support"])
        ax.scatter(*xy, s=28 if compact else 40, color=COLORS[i], edgecolors="white", linewidth=0.45, zorder=5)
        connector = ConnectionPatch(xyA=xy, coordsA=ax.transData,
                                    xyB=(0.0, slot), coordsB=gutter.transAxes,
                                    color=COLORS[i], lw=0.55, alpha=0.8, clip_on=False, zorder=2)
        fig.add_artist(connector)
        labels.append(gutter.text(0.035, slot, alternative["alternative"] + "\nrank " + rank_text(alternative),
                                  color=COLORS[i], fontsize=8 if compact else 10, va="center", linespacing=1.15))
    require(len(labels) == 6, "A profile is missing alternatives")
    return labels


def profile_page(context, profile, full_limits, zoom_limits):
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 0.63, 1.0, 0.63], left=0.065, right=0.975,
                          top=0.79, bottom=0.13, wspace=0.20, hspace=0.55)
    summary = profile["summary"]
    fig.text(0.065, 0.975, f"{context.upper()} · candidate distribution and experimental alternatives", fontsize=19, weight="bold", va="top")
    fig.text(0.065, 0.933, f"Entropy {summary['entropy_bits']:.3f} bits  |  Highest frozen score of four: {summary['highest_scoring_of_four']}  |  Set − Ordering: {summary['set_advantage']:+.3f}", fontsize=11)
    fig.text(0.065, 0.902, textwrap.fill(profile["prompt"], width=170), fontsize=9, color="#566671", va="top", linespacing=1.2)
    a, ag, b, bg = [fig.add_subplot(gs[0, i]) for i in range(4)]
    labels = profile_panel(fig, a, ag, profile, full_limits) + profile_panel(fig, b, bg, profile, zoom_limits)
    a.set_title("Complete distribution: all 121,301 candidates", fontsize=11, loc="left", pad=11)
    b.set_title("Closer view containing all six alternatives", fontsize=11, loc="left", pad=11)
    for ax in (a, b):
        ax.set_xlabel("Candidate rank (log scale)")
        ax.set_ylabel("Full-support probability (log scale)")
    c = fig.add_subplot(gs[1, :2])
    c.semilogx(profile["ranks"], profile["cumulative"], color="#486775", lw=1.5)
    c.axvline(20, color="#A6ACB0", linestyle=(0, (3, 3)), lw=0.8)
    for i, alternative in enumerate(profile["alternatives"]):
        c.scatter(alternative["rank_last"], alternative["cumulative_mass_at_rank_end"], s=35, color=COLORS[i], zorder=4)
    c.set(xlim=(1, 121301), ylim=(0, 1.025), xlabel="Candidate rank (log scale)", ylabel="Cumulative probability")
    c.set_title("How quickly probability mass accumulates", loc="left", fontsize=11, pad=10)
    c.grid(alpha=0.18)
    c.text(0.03, 0.09, f"Top 20 mass: {summary['mass_top_20_candidates']:.2%}\nSix alternatives combined: {summary['mass_six_experimental_alternatives']:.4%}",
           transform=c.transAxes, fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    table_ax = fig.add_subplot(gs[1, 2:])
    table_ax.set_axis_off()
    cells = [[row["alternative"], rank_text(row), f"{row['probability_full_support']:.3g}",
              f"{row['cumulative_mass_at_rank_end']:.4f}", f"{row['frozen_set_inclusion_probability']:.3f}"] for row in profile["alternatives"]]
    table = table_ax.table(cellText=cells, colLabels=["Alternative", "Rank", "Probability", "Cum. mass", "Set inclusion"],
                           cellLoc="left", colLoc="left", colWidths=[0.29, 0.18, 0.18, 0.17, 0.18], bbox=[0, 0.20, 1, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DEE4E7")
        if row == 0:
            cell.set_facecolor("#EDF2F5")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(color=COLORS[row - 1])
    table_ax.text(0, 0.075, f"Most probable candidate: {summary['highest_probability_candidate']!r} ({summary['highest_candidate_probability']:.2%}).", fontsize=10)
    fig.text(0.065, 0.065, "Probabilities are normalized once over every saved one- and two-word candidate. Both views use the same probabilities; all axes have common limits across contexts.", fontsize=9)
    fig.text(0.065, 0.040, "Dashed rank-20 lines are visual references. The frozen Set model uses weighted samples of 20 candidates. Ties share a rank range and a midpoint marker.", fontsize=9)
    label_checks(fig, labels)
    return fig


def overview(profiles, limits, zoom):
    fig = plt.figure(figsize=(24, 20))
    grid = fig.add_gridspec(4, 4, left=0.045, right=0.982, top=0.905, bottom=0.065, hspace=0.40, wspace=0.22)
    title = "The six alternatives within each candidate distribution" if zoom else "Complete candidate distributions for all 16 contexts"
    fig.text(0.045, 0.978, title, fontsize=24, weight="bold", va="top")
    subtitle = f"Closer view through rank {limits[0]:,}; all six alternatives remain visible" if zoom else "All 121,301 candidates shown in every panel"
    fig.text(0.045, 0.948, subtitle + "  ·  common axis limits  ·  all probabilities use full-support normalization", fontsize=13)
    labels = []
    for cell, (context, profile) in zip(grid, profiles.items()):
        inner = cell.subgridspec(1, 2, width_ratios=[1, 0.68], wspace=0.08)
        ax, gutter = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        labels.extend(profile_panel(fig, ax, gutter, profile, limits, compact=True))
        ax.set_title(f"{context.upper()} · H={profile['summary']['entropy_bits']:.2f}", fontsize=12, weight="bold", loc="left", pad=13)
        ax.set_xlabel("Rank (log)", fontsize=9)
        ax.set_ylabel("Probability (log)", fontsize=9)
    fig.text(0.045, 0.027, "Colored markers and callouts locate the six experimental alternatives. Dashed rank-20 lines are reference lines, not deterministic Set boundaries. Tied ranks are marked explicitly.", fontsize=12)
    label_checks(fig, labels)
    require(len(labels) == 96, "Overview must label all 96 context/alternative combinations")
    return fig


def pair_matrix(profile, field):
    words = [row["alternative"] for row in profile["alternatives"]]
    index = {word: i for i, word in enumerate(words)}
    matrix = np.full((6, 6), np.nan)
    for pair in profile["pairs"]:
        matrix[index[pair["trigger"]], index[pair["query"]]] = pair[field]
    require(int(np.isfinite(matrix).sum()) == 30, "Heatmap must show exactly 30 pairs")
    return words, matrix


def draw_matrix(ax, profile, field, title, difference_limit=None, annotate=True):
    words, matrix = pair_matrix(profile, field)
    cmap = plt.get_cmap("RdBu_r" if difference_limit is not None else "viridis").copy()
    cmap.set_bad("#EDF0F2")
    vmin, vmax = (-difference_limit, difference_limit) if difference_limit is not None else (0, 1)
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(6), words, rotation=45, ha="right", fontsize=8 if annotate else 7)
    ax.set_yticks(range(6), words, fontsize=8 if annotate else 7)
    ax.set_title(title, fontsize=11 if annotate else 10, loc="left", pad=10)
    ax.set_xlabel("Query", fontsize=9)
    ax.set_ylabel("Trigger", fontsize=9)
    if annotate:
        for row in range(6):
            for col in range(6):
                value = matrix[row, col]
                if np.isnan(value):
                    continue
                text = f"{value:+.1f}" if difference_limit is not None else f"{value:.2f}"
                if difference_limit is None and field != "human_exclusion_rate" and value in (0, 1):
                    text += "*"
                rgba = im.cmap(im.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col, row, text, ha="center", va="center", fontsize=8, color="black" if luminance > 0.57 else "white")
    return im


def audit_page(context, profile, difference_limit):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.87, bottom=0.22, wspace=0.55, hspace=0.95)
    summary = profile["summary"]
    fig.suptitle(f"{context.upper()} · frozen predictions and pair-level score differences", x=0.055, y=0.98, ha="left", fontsize=19, weight="bold")
    fig.text(0.055, 0.925, "Highest summed score of four: " + summary["highest_scoring_of_four"] + "  |  " +
             "   ".join(f"{DISPLAY[m]} {summary[m + '_log_score']:.2f}" for m in MODELS), fontsize=11)
    fields = [("human_exclusion_rate", "Human exclusion rate"), ("set_probability", "Set probability"),
              ("ordering_probability", "Ordering probability"), ("conjunction_probability", "Conjunction probability"),
              ("disjunction_probability", "Disjunction probability"), ("set_minus_ordering_log_score", "Set − Ordering: pair log-score contribution")]
    for ax, (field, title) in zip(axes.flat, fields):
        im = draw_matrix(ax, profile, field, title, difference_limit if field == "set_minus_ordering_log_score" else None)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    fig.text(0.055, 0.065, "Rows are triggers; columns are queries. Alternatives are ordered by neutral continuation probability. Gray diagonal cells are untested self-pairs.", fontsize=10)
    fig.text(0.055, 0.035, "* Frozen model probability equals 0 or 1. Scores retain original clipping at 1e−10. Positive score contributions favor Set; negative contributions favor Ordering.", fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footers = [text.get_window_extent(renderer) for text in fig.texts[-2:]]
    for ax in axes.flat:
        for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label, ax.title]:
            box = text.get_window_extent(renderer)
            require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1,
                    f"{context}: heatmap label clipped")
            require(not any(box.overlaps(footer) for footer in footers), f"{context}: heatmap labels overlap footer")
    return fig


def difference_overview(profiles, difference_limit):
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    fig.subplots_adjust(left=0.085, right=0.91, top=0.93, bottom=0.095, wspace=0.75, hspace=0.85)
    for ax, (context, profile) in zip(axes.flat, profiles.items()):
        im = draw_matrix(ax, profile, "set_minus_ordering_log_score", f"{context.upper()} · total {profile['summary']['set_advantage']:+.2f}", difference_limit, annotate=False)
    cbar_ax = fig.add_axes([0.945, 0.22, 0.014, 0.52])
    fig.colorbar(im, cax=cbar_ax, label="Pair log score: Set − Ordering")
    fig.text(0.06, 0.985, "Which trigger–query pairs drive Set versus Ordering?", fontsize=23, weight="bold", va="top")
    fig.text(0.06, 0.952, "Red favors Set; blue favors Ordering. Common color scale across all 480 pairs.", fontsize=13)
    fig.text(0.06, 0.025, "Rows = triggers; columns = queries. Within each context, words are ordered by their full-support neutral probability. All original model predictions and score clipping are retained.", fontsize=11)
    return fig


def make_figures(out, profiles):
    style()
    full_limits = (121301, (min(float(profile["p"].min()) for profile in profiles.values()) / 3, 1.0))
    zoom_limits = (min(121301, math.ceil(max(row["rank_last"] for profile in profiles.values() for row in profile["alternatives"]) * 1.35)),
                   (min(row["probability_full_support"] for profile in profiles.values() for row in profile["alternatives"]) / 3, 1.0))
    difference_limit = max(abs(pair["set_minus_ordering_log_score"]) for profile in profiles.values() for pair in profile["pairs"])
    with PdfPages(out / "distribution_profiles.pdf") as distribution_pdf, PdfPages(out / "pair_level_audit.pdf") as audit_pdf:
        for context, profile in profiles.items():
            fig = profile_page(context, profile, full_limits, zoom_limits)
            fig.savefig(out / "profiles" / f"{context}_distribution.png", dpi=180, facecolor="white")
            distribution_pdf.savefig(fig)
            plt.close(fig)
            fig = audit_page(context, profile, difference_limit)
            fig.savefig(out / "pair_heatmaps" / f"{context}_pair_audit.png", dpi=160, facecolor="white")
            audit_pdf.savefig(fig)
            plt.close(fig)
            print(f"Rendered distribution and pair audit: {context}", flush=True)
    for zoom in (False, True):
        fig = overview(profiles, zoom_limits if zoom else full_limits, zoom)
        name = "distribution_overview_zoom" if zoom else "distribution_overview_full"
        fig.savefig(out / (name + ".png"), dpi=160, facecolor="white")
        fig.savefig(out / (name + ".pdf"))
        plt.close(fig)
    fig = difference_overview(profiles, difference_limit)
    fig.savefig(out / "set_ordering_pair_differences_overview.png", dpi=150, facecolor="white")
    fig.savefig(out / "set_ordering_pair_differences_overview.pdf")
    plt.close(fig)
    return {"distribution_pdf_pages": 16, "audit_pdf_pages": 16,
            "individual_distribution_pngs": 16, "individual_pair_audit_pngs": 16,
            "all_96_alternatives_labeled_on_both_overviews": True,
            "alternative_callouts_checked_for_overlap_and_clipping": True,
            "pair_heatmap_axis_labels_checked_for_clipping_and_footer_overlap": True,
            "full_profile_axis_limits": full_limits, "zoom_axis_limits": zoom_limits,
            "full_distribution_curve_points_per_context": 121301,
            "probability_heatmap_limits": [0, 1], "pair_difference_heatmap_limits": [-difference_limit, difference_limit]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--log-probs-dir", type=Path)
    parser.add_argument("--entropy-analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.results_dir = (args.results_dir or args.repo_root / "focus_alt_exp_pipeline/results/set_variant_qwen_corrected_rnx_contrastive_2026-10-23").resolve()
    args.log_probs_dir = (args.log_probs_dir or args.repo_root.parent / "ngrams/qwen_set_variant_log_probs_single_space").resolve()
    args.entropy_analysis_dir = (args.entropy_analysis_dir or args.repo_root / "focus_alt_exp_pipeline/diagnostics/entropy_ordering_advantage_2026-11-04").resolve()
    out = args.output_dir.resolve()
    protected = ["alternative_positions.csv", "pair_level_audit.csv", "context_audit_summary.csv", "query_trigger_audit.csv",
                 "top_20_candidates.csv", "provenance_validation.json", "distribution_profiles.pdf", "pair_level_audit.pdf",
                 "distribution_overview_full.png", "distribution_overview_full.pdf", "distribution_overview_zoom.png",
                 "distribution_overview_zoom.pdf", "set_ordering_pair_differences_overview.png", "set_ordering_pair_differences_overview.pdf",
                 "profiles", "pair_heatmaps"]
    for name in protected:
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing analysis output: {out / name}")
    profiles, alternatives, pairs, summaries, queries, head, validation, validator_path = build_data(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profiles").mkdir()
    (out / "pair_heatmaps").mkdir()
    print("Validated all 16 corrected distributions, 96 alternative positions, and 64 frozen model-score sums.", flush=True)
    plot_checks = make_figures(out, profiles)
    for name, rows in [("alternative_positions.csv", alternatives), ("pair_level_audit.csv", pairs),
                       ("context_audit_summary.csv", summaries), ("query_trigger_audit.csv", queries), ("top_20_candidates.csv", head)]:
        write_csv(out / name, rows)
    report = {"validation_passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
              "working_directory": str(Path.cwd().resolve()), "hostname": socket.gethostname(),
              "script": record(__file__), "validation_script": record(validator_path),
              "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
              "inputs": validation["inputs"], "vocabulary": validation["vocabulary"], "arrays": validation["arrays"],
              "validation": validation["profile_and_pair_audit"], "plot_validation": plot_checks,
              "definitions": {"probabilities": "float64 exp(log_score - max) normalized over all 121301 one-/two-word candidates",
                              "ranks": "1-based descending saved score; ties record first/last ranks and markers use their midpoint",
                              "cumulative_mass": "Sum of probabilities through the last rank in a tied group",
                              "zoom": "Display limits only; probabilities always normalized over the complete saved distribution",
                              "score": "y*ln(clip(p))+(1-y)*ln(1-clip(p)); original epsilon=1e-10; summed without participant weights",
                              "structures": MODELS, "set_advantage": "Set log score minus Ordering log score",
                              "pair_score_audit": "Recomputed only from frozen probabilities to attribute the existing scores; no changed predictions or clipping",
                              "top_three_share": "Sum of the three largest absolute pair differences divided by sum of all 30 absolute pair differences",
                              "scope": "Descriptive plots and score decomposition only; no correlations, tests, model fitting, LM inference, or new samples"},
              "outputs": [record(path) for path in sorted(out.rglob("*")) if path.is_file()
                          and path.name != "provenance_validation.json" and "__pycache__" not in path.parts]}
    with (out / "provenance_validation.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Complete: {out}", flush=True)
    for row in summaries:
        print(f"{row['context']:10} winner={row['highest_scoring_of_four']:12} Set-Ordering={row['set_advantage']:+.4f} "
              f"alternative mass={row['mass_six_experimental_alternatives']:.6g}", flush=True)


if __name__ == "__main__":
    main()
