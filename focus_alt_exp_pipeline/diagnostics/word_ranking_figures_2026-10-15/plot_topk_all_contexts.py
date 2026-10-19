"""All-context Option 2 plots for Disjunction Top-K and Conjunction Top-K.

Uses corrected held-out predictions without fitting or resampling. Requires
numpy, pandas, and matplotlib. Default output: <repository>/output/pdf.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plot_all_contexts import CAPTION, plot
from plot_options import PIPELINE, SOURCE

MODELS = {
    "disjunction": ("Disjunction Top-K", (-.65, .38), (-.6, -.4, -.2, 0, .2)),
    "conjunction": ("Conjunction Top-K", (-.30, .38), (-.2, -.1, 0, .1, .2, .3)),
}


def collect(structure):
    model_label = MODELS[structure][0]
    paths = [SOURCE / "word_spearman_by_context.csv",
             SOURCE / "negation_paired_ranks.csv",
             SOURCE / "negation_spearman_by_context.csv",
             SOURCE.parent / "advisor_summary/item_level_scatterplots/novel_focus_by_context/novel_focus_item_level_predictions.csv"]
    words = pd.read_csv(paths[0]).set_index("context")
    predictions = pd.read_csv(paths[1])
    saved_ranks = pd.read_csv(paths[2]).set_index(["context", "structure", "variant"])
    exported = pd.read_csv(paths[3])
    records = []
    for context, group in predictions.groupby("context", sort=True):
        baseline = group[group.structure.eq("No linking structure")].sort_values("item_id")
        linking = group[group.structure.eq(structure) & group.variant.eq("top_k")].sort_values("item_id")
        assert len(baseline) == len(linking) == 30
        assert baseline.item_id.nunique() == linking.item_id.nunique() == 30
        assert baseline.item_id.tolist() == linking.item_id.tolist()
        assert np.array_equal(baseline.human_rate.to_numpy(), linking.human_rate.to_numpy())
        row = {"context": context, "word_rho": words.loc[context, "spearman_rho"], "n_pairs": 30}
        for prefix, label, saved_structure, variant, frame in [
            ("no_linking", "No linking structure", "No linking structure", "direct", baseline),
            (structure + "_top_k", model_label, structure, "top_k", linking),
        ]:
            row[prefix + "_pearson"] = frame.model_value.corr(frame.human_rate)
            row[prefix + "_spearman"] = frame.model_value.rank().corr(frame.human_rate.rank())
            assert abs(row[prefix + "_spearman"] - saved_ranks.loc[
                (context, saved_structure, variant), "spearman_rho"]) < 1e-12
            # Cross-check against the separate saved item-level reporting export.
            check = exported[exported.context_id.eq(context) & exported.model.eq(label)].sort_values("item_id")
            assert len(check) == 30 and check.item_id.tolist() == frame.item_id.tolist()
            np.testing.assert_allclose(check.prediction, frame.model_value, rtol=0, atol=1e-12)
            np.testing.assert_allclose(check.human_rate, frame.human_rate, rtol=0, atol=1e-12)
            assert abs(row[prefix + "_pearson"] - check.prediction.corr(check.human_rate)) < 1e-12
        for metric in ["pearson", "spearman"]:
            row["delta_" + metric] = row[structure + "_top_k_" + metric] - row["no_linking_" + metric]
        records.append(row)
    data = pd.DataFrame(records).set_index("context").sort_index()
    assert len(data) == 16 and set(data.index) == set(words.index)
    assert np.isfinite(data.to_numpy()).all()
    assert data.n_pairs.sum() == 480
    assert data.word_rho.gt(.9).sum() == 7
    assert np.isclose(data.word_rho, 1, rtol=0, atol=1e-12).sum() == 2
    return data, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PIPELINE.parent / "output/pdf")
    parser.add_argument("--structures", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for structure in args.structures:
        label, limits, ticks = MODELS[structure]
        name = f"option_2_{structure}_top_k_context_improvements_all_16"
        data, sources = collect(structure)
        plot(data, args.output_dir, model_label=label, output_name=name,
             xlimits=limits, xticks=ticks)
        data.to_csv(args.output_dir / f"{name}.csv")
        (args.output_dir / f"{name}_caption.txt").write_text(CAPTION.replace("Set Top-K", label) + "\n")
        (args.output_dir / f"{name}_provenance.json").write_text(json.dumps({
            "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "script": str(Path(__file__).resolve()),
            "selection": "All 16 contexts, alphabetical order; 30 pairs per context, 480 total.",
            "evaluation": f"{label} minus No linking within-context correlations. Spearman uses average ranks for ties. Original corrected held-out predictions; no fitting or resampling. Only top_k linking rows selected.",
            "validation": "All 32 model/context Spearman coefficients match the saved full-data results within 1e-12. All item predictions, human rates, and Pearson coefficients match the separate saved item-level reporting export within 1e-12.",
            "horizontal_limits": limits,
        }, indent=2) + "\n")
        print(f"{label}: verified all 16 contexts; exported PDF, PNG, SVG, data, caption, and provenance.")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
