"""All 16 contexts for Ordering minus No linking, in the Option 2 layout.

Uses corrected held-out results, without refitting or resampling. Requires
numpy, pandas, and matplotlib. Run with --output-dir to override repo/output/pdf.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plot_all_contexts import CAPTION, plot
from plot_options import OUT, PIPELINE, SOURCE

NAME = "option_2_ordering_context_improvements_all_16"
XLIMITS = (-.65, .38)
XTICKS = (-.6, -.4, -.2, 0, .2)


def collect():
    paths = [SOURCE / "word_spearman_by_context.csv",
             SOURCE / "negation_paired_ranks.csv",
             SOURCE / "negation_spearman_by_context.csv",
             OUT / "ordering_figure_data.csv"]
    words = pd.read_csv(paths[0]).set_index("context")
    predictions = pd.read_csv(paths[1])
    # Both saved Ordering variants have identical predictions; use each pair once.
    variants = predictions[predictions.structure.eq("ordering")].pivot(
        index="item_id", columns="variant", values="model_value")
    assert len(variants) == 480
    assert np.array_equal(variants.top_k, variants.top_p)
    saved_ranks = pd.read_csv(paths[2]).set_index(["context", "structure", "variant"])
    records = []
    for context, group in predictions.groupby("context", sort=True):
        baseline = group[group.structure.eq("No linking structure")].sort_values("item_id")
        ordering = group[group.structure.eq("ordering") & group.variant.eq("top_k")].sort_values("item_id")
        assert baseline.item_id.nunique() == ordering.item_id.nunique() == 30
        assert len(baseline) == len(ordering) == 30
        assert baseline.item_id.tolist() == ordering.item_id.tolist()
        assert np.array_equal(baseline.human_rate.to_numpy(), ordering.human_rate.to_numpy())
        row = {"context": context, "word_rho": words.loc[context, "spearman_rho"], "n_pairs": 30}
        for prefix, structure, variant, frame in [
            ("no_linking", "No linking structure", "direct", baseline),
            ("ordering", "ordering", "top_k", ordering),
        ]:
            row[prefix + "_pearson"] = frame.model_value.corr(frame.human_rate)
            row[prefix + "_spearman"] = frame.model_value.rank().corr(frame.human_rate.rank())
            assert abs(row[prefix + "_spearman"] - saved_ranks.loc[
                (context, structure, variant), "spearman_rho"]) < 1e-12
        for metric in ["pearson", "spearman"]:
            row["delta_" + metric] = row["ordering_" + metric] - row["no_linking_" + metric]
        records.append(row)
    data = pd.DataFrame(records).set_index("context").sort_index()
    assert len(data) == 16 and set(data.index) == set(words.index)
    assert np.isfinite(data.to_numpy()).all()
    assert data.n_pairs.sum() == 480
    assert data.word_rho.gt(.9).sum() == 7
    assert np.isclose(data.word_rho, 1, rtol=0, atol=1e-12).sum() == 2
    saved = pd.read_csv(paths[3]).set_index("context").sort_index()
    pd.testing.assert_frame_equal(data.loc[saved.index, saved.columns], saved, rtol=0, atol=1e-12)
    return data, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    args = parser.parse_args()
    data, sources = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot(data, args.output_dir, model_label="Ordering", output_name=NAME,
         xlimits=XLIMITS, xticks=XTICKS)
    data.to_csv(args.output_dir / f"{NAME}.csv")
    (args.output_dir / f"{NAME}_caption.txt").write_text(CAPTION.replace("Set Top-K", "Ordering") + "\n")
    (args.output_dir / f"{NAME}_provenance.json").write_text(json.dumps({
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "script": str(Path(__file__).resolve()),
        "selection": "All 16 contexts, alphabetical order; 30 pairs per context, 480 total.",
        "evaluation": "Ordering minus No linking within-context correlations. Spearman uses average ranks for ties. Original corrected held-out predictions; no fitting or resampling. Identical Ordering top_k and top_p variants deduplicated.",
        "validation": "All 32 model/context Spearman coefficients match the saved full-data results within 1e-12. Every coefficient and difference for the seven original contexts matches the original Ordering figure data within 1e-12.",
        "horizontal_limits": XLIMITS,
    }, indent=2) + "\n")
    print("Verified all 16 contexts and the original seven-context Ordering figure; exported PDF, PNG, SVG, data, caption, and provenance.")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
