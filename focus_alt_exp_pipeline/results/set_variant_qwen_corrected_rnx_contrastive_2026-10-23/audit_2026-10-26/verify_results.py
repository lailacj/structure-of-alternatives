"""Audit the completed extension against its parent and raw returned scores.

Run with Python, NumPy and SciPy. No pipeline modules are imported, and no
inference, sampling or fitting is performed. Only verification.json is written.
"""

import csv
from datetime import datetime
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scipy.stats import pearsonr, spearmanr


AUDIT = Path(__file__).resolve().parent
BUNDLE = AUDIT.parent
PIPELINE = BUNDLE.parents[1]
REPOSITORY = PIPELINE.parent
PARENT = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def payload(path):
    html = path.read_text()
    match = re.search(r'<script id="results-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, path
    assert "__VIEWER_DATA__" not in html and "/* VIEWER_JS */" not in html
    return json.loads(match.group(1))


def main():
    before = json.loads((AUDIT / "original_file_hashes.json").read_text())
    changed = [name for name, checksum in before.items()
               if digest(REPOSITORY / name) != checksum]
    assert not changed, changed
    old_source = rows(PARENT / "inputs/source_rows.csv")
    new_source = rows(BUNDLE / "inputs/source_rows.csv")
    extension = json.loads((BUNDLE / "inputs/rnx_contrastive_provenance.json").read_text())
    allowed = set(extension["changed_source_fields"])
    assert len(old_source) == len(new_source) == 1089
    source_changes = 0
    for old, new in zip(old_source, new_source):
        is_rnx = old["dataset_family"] == "ronai_xiang_2024"
        differences = {key for key in old if old[key] != new[key]}
        assert differences <= allowed if is_rnx else not differences
        assert all(not new[key] or (is_rnx and key in allowed)
                   for key in new.keys() - old.keys())
        if is_rnx:
            source_changes += 1
            assert new["x_but_not_y_applicable"] == "True"
            assert math.isfinite(float(new["x_but_not_y_logprob_sum"]))
    assert source_changes == 300
    assert extension["source_rows_sha256"] == digest(BUNDLE / "inputs/source_rows.csv")
    assert extension["parent_source_rows_sha256"] == digest(PARENT / "inputs/source_rows.csv")

    prediction_counts = {}
    for relative in ["prediction_grid.csv", "cv_results/oof_predictions.csv"]:
        with (PARENT / relative).open(newline="") as a, (BUNDLE / relative).open(newline="") as b:
            count = 0
            for old, new in zip_longest(csv.DictReader(a), csv.DictReader(b)):
                assert old is not None and new is not None
                assert old.pop("source_rows_sha256") == extension["parent_source_rows_sha256"]
                assert new.pop("source_rows_sha256") == extension["source_rows_sha256"]
                assert old == new
                count += 1
        prediction_counts[relative] = count
        assert count == extension["copied_predictions"][relative]["rows"]
    cv_files = [p for p in (PARENT / "cv_results").iterdir()
                if p.is_file() and p.name != "oof_predictions.csv"]
    for path in cv_files:
        assert digest(path) == digest(BUNDLE / "cv_results" / path.name)

    old_view = payload(PIPELINE / "results_viewer/index.html")
    new_view = payload(BUNDLE / "results_viewer.html")
    assert len(new_view["items"]) == len(old_view["items"]) == 993
    assert len(new_view["datasets"]) == 10 and len(new_view["contexts"]) == 16
    assert new_view["verifiedCells"] == 261
    max_score_roundoff = 0.0
    for old, new in zip(old_view["items"], new_view["items"]):
        current = json.loads(json.dumps(new))
        if old["dataset"].startswith("rnx_"):
            assert old["p"][1] is None and old["scores"][1] is None
            assert current["p"][1] is not None and current["scores"][1] is not None
            current["p"][1] = old["p"][1]
            current["scores"][1] = old["scores"][1]
        for a, b in zip(old["scores"], current["scores"]):
            if a is None or b is None:
                assert a is b
            else:
                max_score_roundoff = max(max_score_roundoff, abs(a - b))
                assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
        # Predictions and all non-score fields must match exactly. Recomputed
        # logarithms allow only floating-point roundoff across Python runtimes.
        current["scores"] = old["scores"]
        assert current == old
    old_distributions = {p["id"]: p for p in old_view["prompts"] if p.get("distribution")}
    new_prompts = {p["id"]: p for p in new_view["prompts"]}
    assert len(old_distributions) == 360
    for key, old in old_distributions.items():
        new = new_prompts[key]
        assert new["text"] == old["text"] and new["frame"] == old["frame"]
        for field in ["distribution", "distributionCoverage", "supportSize", "distributionScoringBoundary"]:
            assert old.get(field) == new.get(field)

    metrics = [("correlations", "r"), ("spearman", "rho"), ("log_scores", "log")]
    tables = {stem: {r["Dataset"]: r for r in rows(BUNDLE / "linking_structure_tables" /
              f"{stem}_by_dataset_and_linking_structure.csv")} for stem, _ in metrics}
    unchanged_cells = 0
    for stem, _ in metrics:
        for old in rows(PARENT / "linking_structure_tables" / f"{stem}_by_dataset_and_linking_structure.csv"):
            new = tables[stem][old["Dataset"]]
            for key in old.keys() - {"Dataset"}:
                if old["Dataset"].startswith("R&X ") and key == "X but not Y":
                    continue
                try:
                    a, b = float(old[key]), float(new[key])
                except ValueError:
                    assert old[key] == new[key]
                else:
                    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                unchanged_cells += 1

    scores = rows(PIPELINE / "model_scores/rnx_x_but_not_y_qwen_scores.csv")
    lookup = {(r["condition"], r["context_id"]): r for r in scores}
    assert len(lookup) == 300
    independent = {}
    for condition in ["ESI", "Eweak", "Estrong", "Eonly", "Eonlystrong"]:
        source = [r for r in old_source if r["dataset_family"] == "ronai_xiang_2024"
                  and r["condition"] == condition]
        assert len(source) == 60
        y = [float(r["human_rate"]) for r in source]
        p = [math.exp(float(lookup[condition, r["context_id"]]["query_logprob_sum"])) for r in source]
        clipped = [max(1e-10, min(1 - 1e-10, v)) for v in p]
        values = {"r": float(pearsonr(y, p).statistic), "rho": float(spearmanr(y, p).statistic),
                  "log": math.fsum(a * math.log(b) + (1 - a) * math.log1p(-b)
                                   for a, b in zip(y, clipped))}
        summary = new_view["datasetSummaries"]["rnx_" + condition.lower()][1]
        assert summary["n"] == 60
        for stem, metric in metrics:
            expected = values[metric]
            assert math.isclose(float(tables[stem]["R&X " + condition]["X but not Y"]),
                                expected, rel_tol=0, abs_tol=1e-10)
            assert math.isclose(summary[metric], expected, rel_tol=0, abs_tol=1e-10)
        independent[condition] = {"n": 60, **values}

    paired = rows(BUNDLE / "rnx_contrastive/paired_condition_changes.csv")
    assert len(paired) == 240
    only_rows = [r for r in paired if "Eonly" in r["contrast"]]
    assert len(only_rows) == 120
    assert all(float(r["model_delta"]) == 0 and float(r["model_logprob_delta"]) == 0 for r in only_rows)
    figures = BUNDLE / "advisor_summary/four_model_comparison"
    fits = rows(figures / "plotted_fit_values.csv")
    assert len(fits) == 120 and all(math.isfinite(float(r["value"])) for r in fits)
    for stem in ["log_score", "pearson_r", "spearman_rho"]:
        for suffix in ["png", "pdf", "svg"]:
            assert (figures / f"four_models_{stem}.{suffix}").stat().st_size > 1000

    report = {"verified_at": datetime.now(ZoneInfo("America/New_York")).isoformat(), "status": "passed",
              "original_files_hash_checked": len(before), "original_files_changed": changed,
              "source_rows": 1089, "rnx_rows_extended": 300,
              "prediction_rows_preserved": prediction_counts, "unchanged_cv_files": len(cv_files),
              "analysis_units": 993, "datasets": 10, "focus_contexts": 16,
              "neutral_distributions_preserved": 360, "viewer_metric_cells_verified_by_builder": 261,
              "existing_table_cells_within_tolerance": unchanged_cells,
              "existing_viewer_item_log_scores_max_roundoff": max_score_roundoff,
              "independently_recomputed_rnx_metrics": independent,
              "metric_formula": "scipy.stats Pearson/Spearman; sum y*ln(p)+(1-y)*ln(1-p); p=exp(raw query_logprob_sum); epsilon=1e-10",
              "independent_metric_tolerance": 1e-10, "paired_condition_rows": 240,
              "only_pairs_with_zero_model_change": 120, "four_model_fit_values": 120, "figure_exports": 9,
              "source_rows_sha256": digest(BUNDLE / "inputs/source_rows.csv"),
              "viewer_sha256": digest(BUNDLE / "results_viewer.html")}
    (AUDIT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
