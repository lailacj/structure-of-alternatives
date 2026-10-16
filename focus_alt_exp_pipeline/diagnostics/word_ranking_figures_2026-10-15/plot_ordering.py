"""Ordering version of the seven-context Option 2 figure."""
import hashlib
import json
import numpy as np
import pandas as pd
from plot_options import OUT, PIPELINE, SOURCE, CAPTIONS, option2


def main():
    word_path = SOURCE / "word_spearman_by_context.csv"
    pred_path = SOURCE / "negation_paired_ranks.csv"
    words = pd.read_csv(word_path).set_index("context")
    pred = pd.read_csv(pred_path)
    selected = words[words.spearman_rho.gt(.9)]
    assert len(selected) == 7
    ordering = pred[pred.structure.eq("ordering")].pivot(index="item_id", columns="variant", values="model_value")
    assert np.array_equal(ordering.top_k, ordering.top_p)
    saved = pd.read_csv(PIPELINE / "diagnostics/word_ranking_gt_0_9_2026-10-08/correlations_by_context.csv").set_index(["context", "model"])
    rows = []
    for context, word in selected.iterrows():
        g = pred[pred.context.eq(context)]
        a = g[g.structure.eq("No linking structure")].sort_values("item_id")
        b = g[g.structure.eq("ordering") & g.variant.eq("top_k")].sort_values("item_id")
        assert len(a) == len(b) == 30
        assert a.item_id.tolist() == b.item_id.tolist()
        assert np.array_equal(a.human_rate.to_numpy(), b.human_rate.to_numpy())
        row = {"context": context, "word_rho": word.spearman_rho, "n_pairs": 30}
        for prefix, model, frame in [("no_linking", "No linking", a), ("ordering", "Ordering", b)]:
            row[prefix + "_pearson"] = frame.model_value.corr(frame.human_rate)
            row[prefix + "_spearman"] = frame.model_value.rank().corr(frame.human_rate.rank())
            assert abs(row[prefix + "_pearson"] - saved.loc[(context, model), "pearson_r"]) < 1e-12
            assert abs(row[prefix + "_spearman"] - saved.loc[(context, model), "spearman_rho"]) < 1e-12
        for metric in ["pearson", "spearman"]:
            row["delta_" + metric] = row["ordering_" + metric] - row["no_linking_" + metric]
        rows.append(row)
    data = pd.DataFrame(rows).set_index("context").sort_index()
    data.to_csv(OUT / "ordering_figure_data.csv")
    option2(data, model_label="Ordering", output_name="option_2_ordering_context_improvements",
            xlimits=(-.30, .30), xticks=(-.3, -.2, -.1, 0, .1, .2, .3))
    caption = CAPTIONS["option_2_context_improvements"].replace("Set Top-K", "Ordering")
    (OUT / "option_2_ordering_caption.txt").write_text(caption + "\n")
    (OUT / "ordering_provenance.json").write_text(json.dumps({
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [word_path, pred_path]},
        "selection": "Word-ranking Spearman strictly greater than .9; seven contexts.",
        "evaluation": "Ordering minus No linking exclusion correlations. 30 equally weighted pairs per context; Spearman uses average ranks for ties. Corrected held-out predictions, no refitting. Identical Ordering variants deduplicated.",
        "validation": "All 28 model/context coefficients match the saved mini-analysis within 1e-12.",
    }, indent=2) + "\n")
    print(data.to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
