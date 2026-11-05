#!/usr/bin/env python3
"""One-sided lower-tail Spearman permutation test for Set advantage.

Set advantage = Set log score minus Ordering log score. The predicted
association is negative, equivalent to a positive Ordering-advantage association.
The 16 saved entropy/advantage pairs are the observations. Hold entropy fixed
and shuffle advantage ranks across contexts, using one million permutations
and seed 20261104 by default. No model scoring, sampling, or fitting occurs.
Only the primary Spearman test is run; there are no secondary hypothesis tests.

Requires NumPy and SciPy. Existing output files are never overwritten.
Method: https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.permutation_test.html
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import sys
import time

import numpy as np
import scipy
from scipy import stats


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=ANALYSIS_DIR / "entropy_ordering_advantage.csv")
    parser.add_argument("--analysis-provenance", type=Path, default=ANALYSIS_DIR / "provenance_validation.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--n-permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    require(0 < args.n_permutations < math.factorial(16), "Specify a positive Monte Carlo permutation count below 16!")
    require(args.batch_size > 0 and args.seed >= 0, "Batch size must be positive and seed nonnegative")
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "spearman_permutation_result.csv"
    provenance_path = output_dir / "provenance_validation.json"
    for path in (csv_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    input_record = file_record(args.input_csv)
    analysis_provenance = json.loads(args.analysis_provenance.read_text())
    require(analysis_provenance["validation_passed"] is True
            and analysis_provenance["output_csv"]["sha256"] == input_record["sha256"],
            "Input CSV does not match the validated entropy/advantage table")
    with args.input_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # A declared order makes the finite Monte Carlo realization reproducible.
    rows.sort(key=lambda row: row["context"])
    require(len(rows) == len({row["context"] for row in rows}) == 16,
            "Expected exactly 16 unique focus contexts")
    require(all(int(row["n_candidates"]) == 121301 and int(row["n_trigger_query_pairs"]) == 30 for row in rows),
            "Unexpected source candidate or pair counts")
    require(all(float(row["ordering_advantage"]) == float(row["ordering_log_score"]) - float(row["set_log_score"])
                for row in rows), "Ordering-minus-Set arithmetic mismatch")
    x = np.array([float(row["entropy_bits"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["set_log_score"]) - float(row["ordering_log_score"]) for row in rows], dtype=np.float64)
    require(bool(np.array_equal(y, -np.array([float(row["ordering_advantage"]) for row in rows], dtype=np.float64))),
            "Set advantage is not exactly the negative of Ordering advantage")
    n = len(rows)
    require(bool(np.isfinite(x).all() and np.isfinite(y).all()), "Nonfinite input")
    require(np.unique(x).size == np.unique(y).size == n, "This frozen table is expected to be tie-free")
    rank_x = stats.rankdata(x, method="average")
    rank_y = stats.rankdata(y, method="average")
    # Twice-centered ranks are exact integers. The numerator therefore has no
    # floating-point ambiguity when including tied permutation statistics.
    centered_x = (2 * rank_x - (n + 1)).astype(np.int64)
    centered_y = (2 * rank_y - (n + 1)).astype(np.int64)
    denominator = math.sqrt(int(centered_x @ centered_x) * int(centered_y @ centered_y))
    observed_numerator = int(centered_x @ centered_y)

    def spearman_statistic(permuted_y: np.ndarray, axis: int = -1) -> np.ndarray:
        # Permuting saved ranks is equivalent to reranking permuted raw values.
        return np.sum(permuted_y * centered_x, axis=axis, dtype=np.int64) / denominator

    # Validate the statistic without computing another hypothesis test.
    observed_rho = float(spearman_statistic(centered_y))
    rho_from_rank_distances = 1.0 - 6.0 * float(np.sum((rank_x - rank_y) ** 2)) / (n * (n ** 2 - 1))
    require(math.isclose(observed_rho, rho_from_rank_distances, rel_tol=0, abs_tol=1e-15),
            "Spearman rank-distance and centered-rank calculations disagree")
    require(float(spearman_statistic(centered_x)) == 1.0
            and float(spearman_statistic(-centered_x)) == -1.0,
            "Statistic fails perfect/reversed rank validation")

    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    permutation = stats.permutation_test(
        (centered_y,), spearman_statistic, permutation_type="pairings",
        vectorized=True, n_resamples=args.n_permutations, batch=args.batch_size,
        alternative="less", random_state=rng,
    )
    elapsed = time.monotonic() - started
    null_rho = np.asarray(permutation.null_distribution, dtype=np.float64)
    require(null_rho.shape == (args.n_permutations,) and bool(np.isfinite(null_rho).all()),
            "Unexpected permutation null distribution")
    null_numerators = np.rint(null_rho * denominator).astype(np.int64)
    require(bool(np.allclose(null_numerators / denominator, null_rho, rtol=0, atol=1e-15)),
            "Permutation statistics do not match exact rank numerators")
    extreme_count = int(np.count_nonzero(null_numerators <= observed_numerator))
    equal_count = int(np.count_nonzero(null_numerators == observed_numerator))
    pvalue = (extreme_count + 1) / (args.n_permutations + 1)
    require(observed_rho == float(permutation.statistic)
            and math.isclose(pvalue, float(permutation.pvalue), rel_tol=0, abs_tol=1e-15),
            "Independent lower-tail count disagrees with SciPy")
    monte_carlo_se = math.sqrt(pvalue * (1 - pvalue) / args.n_permutations)
    summary = {
        "test": "Spearman context-pairing permutation test", "alternative": "less",
        "advantage_definition": "set_log_score - ordering_log_score",
        "n_contexts": n, "spearman_rho": observed_rho,
        "one_sided_pvalue": pvalue, "n_permutations": args.n_permutations,
        "n_null_statistics_le_observed": extreme_count, "n_null_statistics_equal_observed": equal_count,
        "seed": args.seed, "monte_carlo_standard_error_of_pvalue": monte_carlo_se,
    }
    report = {
        "validation_passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": summary, "input_csv": input_record,
        "analysis_provenance": file_record(args.analysis_provenance), "script": file_record(Path(__file__)),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "working_directory": str(Path.cwd().resolve()), "hostname": socket.gethostname(),
        "python_version": sys.version, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "method": {
            "unit_of_analysis": "context; n=16, with no participant or pair-count weighting",
            "null": "Entropy and advantage pairings are unrelated and exchangeable across contexts",
            "alternative": "Negative monotonic association between entropy and Set-minus-Ordering advantage",
            "equivalence": "Set advantage = -Ordering advantage; the lower-tail Set test is the original upper-tail Ordering test with signs reversed",
            "permutation": "Hold entropy ranks fixed; uniformly shuffle advantage ranks across the 16 contexts",
            "input_order": "context ascending (alphabetical)", "test_type": "Monte Carlo; not exhaustive",
            "rng": type(rng.bit_generator).__name__, "seed": args.seed, "batch_size": args.batch_size,
            "pvalue_formula": "(1 + count(rho_permuted <= rho_observed)) / (1 + n_permutations)",
            "tail_equality": "Included; comparisons validated using exact integer centered-rank numerators",
            "monte_carlo_se_formula": "sqrt(pvalue*(1-pvalue)/n_permutations); simulation error, not an effect-size SE",
            "conditioning": "Frozen model predictions, selected K, measured entropies, and these 16 contexts",
            "scope": "Only the specified primary test; no secondary tests, model fitting, or LM scoring/sampling",
        },
        "validation": {"source_csv_hash_matches": True, "all_16_contexts_included": True,
                       "advantage_arithmetic_matches": True, "set_advantage_is_exact_negative_of_ordering_advantage": True, "no_rank_ties": True,
                       "two_spearman_formulas_agree": True, "scipy_and_integer_tail_counts_agree": True,
                       "observed_rank_numerator": observed_numerator, "rank_denominator": denominator},
        "context_values_and_ranks": [
            {"context": row["context"], "entropy_bits": float(x[i]), "set_advantage": float(y[i]),
             "entropy_rank": float(rank_x[i]), "advantage_rank": float(rank_y[i])}
            for i, row in enumerate(rows)],
        "permutation_elapsed_seconds": elapsed,
    }
    require(file_record(args.input_csv) == input_record, "Input CSV changed during analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    report["output_csv"] = file_record(csv_path)
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Results: {csv_path}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
