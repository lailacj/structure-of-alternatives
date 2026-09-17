"""Exploratory context-level association, historical fixed predictions only.

Writes only beside this script. Rebuilds the two context measures, verifies the
saved outputs and sampling-score provenance, and preserves every input file.
No Qwen scoring, boundary selection, model refitting, or participant resampling.
"""
from pathlib import Path
from html import escape
import hashlib
import itertools
import json
import re
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PIPE = OUT.parents[1]
sys.path.insert(0, str(PIPE / "code"))
from evaluate_focus_spearman import evaluate

SEED = 20260917
PERMUTATIONS = 199999
BOOTSTRAPS = 20000
SPECS = [
    ("No linking structure", "direct", "No linking"),
    ("X but not Y", "direct", "X but not Y"),
    ("ordering", "top_k", "Ordering"),
    ("set", "top_k", "Set Top-K"),
    ("conjunction", "top_k", "Conjunction Top-K"),
    ("disjunction", "top_k", "Disjunction Top-K"),
    ("set", "top_p", "Set Top-p"),
    ("conjunction", "top_p", "Conjunction Top-p"),
    ("disjunction", "top_p", "Disjunction Top-p"),
]
CASES = ("fridge", "beach", "cold")
COLORS = {"fridge": "#147d72", "beach": "#7657a6", "cold": "#bd5b21"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else np.nan


def rho(x, y):
    return pearson(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def row_corr(x, y):
    x, y = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x*x, axis=1) * np.sum(y*y, axis=1))
    return np.divide(np.sum(x*y, axis=1), denominator,
                     out=np.full(len(x), np.nan), where=denominator > 0)


def permutation_p(x, y, permutations):
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    xr, yr = xr - xr.mean(), yr - yr.mean()
    denominator = np.linalg.norm(xr) * np.linalg.norm(yr)
    observed = float(xr @ yr / denominator)
    null = np.sum(xr[permutations] * yr[None, :], axis=1) / denominator
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    p = (extreme + 1) / (len(null) + 1)
    return observed, p, extreme


def bootstrap_rho(x, y, indices):
    # Re-rank within each bootstrap sample: duplicated contexts alter midranks.
    xr = pd.DataFrame(np.asarray(x)[indices]).rank(axis=1, method="average").to_numpy()
    yr = pd.DataFrame(np.asarray(y)[indices]).rank(axis=1, method="average").to_numpy()
    return row_corr(xr, yr)


def self_test():
    x, y = [1, 1, 3, 4], [4, 3, 2, 1]
    assert np.isclose(rho(x, y), pd.Series(x).rank().corr(pd.Series(y).rank()))
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    idx = np.array([[0, 0, 2, 3], [1, 2, 3, 3], [0, 1, 2, 3]])
    fast = bootstrap_rho(np.array(x), np.array(y), idx)
    slow = [rho(np.array(x)[i], np.array(y)[i]) for i in idx]
    assert np.allclose(fast, slow)
    permutations = np.array(list(itertools.permutations(range(4))))
    point, _, count = permutation_p(x, y, permutations)
    exact_count = sum(abs(rho(np.array(x)[i], y)) >= abs(point) - 1e-12 for i in permutations)
    assert count == exact_count
    assert np.isnan(rho([1, 1, 1], [1, 2, 3]))


def load_data():
    results = PIPE / "results/set_variant_qwen"
    source_path = PIPE / "scoring_manifests/set_variant_qwen/source_rows.csv"
    human_path = PIPE / "human_exp_data/sca_dataframe.csv"
    oof_path = results / "cv_results/oof_predictions.csv"
    viewer_path = PIPE / "results_viewer/index.html"
    grid_path = results / "prediction_grid.csv"
    paths = [human_path, source_path, oof_path, viewer_path, grid_path]
    paths += list((results / "spearman").glob("*.csv"))
    original_hashes = {str(p): sha(p) for p in paths}
    human, source, oof = [pd.read_csv(p) for p in (human_path, source_path, oof_path)]
    rebuilt = evaluate(human, source, oof)
    for name, keys in [("word_spearman_by_context", ["context", "model"]),
                       ("negation_spearman_by_context", ["context", "model", "structure", "variant"])]:
        fresh = rebuilt[name].sort_values(keys).reset_index(drop=True)
        saved = pd.read_csv(results / "spearman" / (name + ".csv")).sort_values(keys).reset_index(drop=True)
        assert fresh[keys].equals(saved[keys])
        assert np.allclose(fresh.spearman_rho, saved.spearman_rho, equal_nan=True, atol=1e-12)

    payload = json.loads(re.search(r'<script id="results-data" type="application/json">(.*?)</script>',
                                  viewer_path.read_text(), re.S).group(1))
    for path in (source_path, oof_path, grid_path):
        recorded = {r["sha256"] for r in payload["provenance"] if Path(r["path"]).name == path.name}
        assert recorded == {original_hashes[str(path)]}, f"Viewer provenance mismatch: {path}"
    prompts = {p["id"]: p for p in payload["prompts"] if p["frame"] == "Neutral"}
    focus_source = source.loc[source.dataset_family.eq("novel_focus")].copy()
    words = rebuilt["word_paired_ranks"].copy()
    meta = focus_source[["context_id", "prompt_id", "generation_prompt"]].drop_duplicates()
    assert len(meta) == 16
    words = words.merge(meta.rename(columns={"context_id": "context"}), on="context", validate="many_to_one")
    arrays = []
    for row in words.itertuples():
        prompt = prompts[row.prompt_id]
        assert prompt["text"] == row.generation_prompt
        candidate = {d["word"].strip().lower(): d for d in prompt["distribution"]}[row.word]
        arrays.append({"context": row.context, "word": row.word,
                       "sampling_logp": candidate["logp"], "sampling_probability": candidate["normalized"],
                       "sampling_vocab_rank": candidate["rank"]})
    words = words.merge(pd.DataFrame(arrays), on=["context", "word"], validate="one_to_one")
    words["sampling_word_rank"] = words.groupby("context").sampling_logp.rank(ascending=False) - 1
    # Rank 0 is best in source; present the word tables with intuitive ranks 1--6.
    for col in ("human_rank", "model_rank", "sampling_word_rank"):
        words[col] += 1
    word_stats = []
    for context, g in words.groupby("context"):
        word_stats.append({"context": context,
                           "viewer_word_rho": rho(-g.human_rank, g.model_value),
                           "sampling_word_rho": rho(-g.human_rank, g.sampling_logp)})
    word_stats = pd.DataFrame(word_stats)
    neg = rebuilt["negation_spearman_by_context"].copy()
    pairs = rebuilt["negation_paired_ranks"].copy()
    groups, item_groups = [], []
    for structure, variant, label in SPECS:
        g = neg.loc[neg.structure.eq(structure) & neg.variant.eq(variant)].copy()
        g["predictor"] = label
        groups.append(g)
        p = pairs.loc[pairs.structure.eq(structure) & pairs.variant.eq(variant)].copy()
        p["predictor"] = label
        item_groups.append(p)
    joined = pd.concat(groups).merge(word_stats, on="context", validate="many_to_one")
    # The six-word Spearman values lie on a discrete grid. Collapse numerical
    # floating-point differences only, so mathematically tied values stay tied.
    for col in ("spearman_rho", "viewer_word_rho", "sampling_word_rho"):
        joined[col] = joined[col].round(12)
    assert joined.groupby("predictor").size().eq(16).all()
    return joined, words, pd.concat(item_groups), focus_source, original_hashes, human


def test_associations(joined):
    results, leave_one_out = [], []
    streams = {}
    for n in (15, 16):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, n, 1]))
        permutations = rng.permuted(np.tile(np.arange(n), (PERMUTATIONS, 1)), axis=1)
        indices = np.random.default_rng(np.random.SeedSequence([SEED, n, 2])).integers(n, size=(BOOTSTRAPS, n))
        streams[n] = permutations, indices
    for definition in ("viewer_word_rho", "sampling_word_rho"):
        for _, _, label in SPECS:
            g = joined.loc[joined.predictor.eq(label)].dropna(subset=[definition, "spearman_rho"]).sort_values("context")
            x, y = g[definition].to_numpy(), g.spearman_rho.to_numpy()
            point, p, count = permutation_p(x, y, streams[len(g)][0])
            draws = bootstrap_rho(x, y, streams[len(g)][1])
            valid = draws[np.isfinite(draws)]
            loo = []
            for omitted in range(len(g)):
                keep = np.arange(len(g)) != omitted
                value = rho(x[keep], y[keep])
                loo.append(value)
                leave_one_out.append({"word_definition": definition, "predictor": label,
                                      "omitted_context": g.iloc[omitted].context, "spearman": value})
            results.append({"word_definition": definition, "predictor": label, "n_contexts": len(g),
                            "spearman": point, "pearson_descriptive": pearson(x, y),
                            "permutation_p_two_sided": p, "permutation_extreme_count": count,
                            "permutation_mc_se": np.sqrt(p * (1-p) / (PERMUTATIONS+1)),
                            "bootstrap_95_low": np.quantile(valid, .025), "bootstrap_95_high": np.quantile(valid, .975),
                            "bootstrap_valid": len(valid), "bootstrap_undefined": len(draws)-len(valid),
                            "leave_one_out_min": min(loo), "leave_one_out_max": max(loo)})
    results = pd.DataFrame(results)
    results["holm_p_within_definition_9"] = results.groupby("word_definition").permutation_p_two_sided.transform(holm)
    results["holm_p_all_18_sensitivity"] = holm(results.permutation_p_two_sided)
    # Descriptive common-coverage check; avoids mixing 15-context and 16-context
    # coefficients when comparing predictors. Not an additional test family.
    common = joined.groupby("context").spearman_rho.apply(lambda s: s.notna().all())
    for index, row in results.iterrows():
        g = joined.loc[joined.predictor.eq(row.predictor) & joined.context.isin(common.index[common])]
        results.loc[index, "common_15_spearman"] = rho(g[row.word_definition], g.spearman_rho)
    return results, pd.DataFrame(leave_one_out)


class SVG:
    def __init__(self, width, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24323e}</style>',
                      f'<rect width="{width}" height="{height}" fill="white"/>']
    def text(self, x, y, value, size=16, anchor="start", color="#24323e", weight="normal", extra=""):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" style="fill:{color}" {extra}>{escape(str(value))}</text>')
    def line(self, x1, y1, x2, y2, color="#dfe5ea", width=1):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
    def circle(self, x, y, radius=5, color="#708493"):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="1"/>')
    def save(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]))


def scatterplots(joined, results, definition, filename):
    canvas = SVG(1440, 1290)
    title = "Word-rank agreement and exclusion agreement across contexts"
    subtitle = "Historical viewer word scores" if definition == "viewer_word_rho" else "Sensitivity: word ranks from the actual historical sampling scores"
    canvas.text(55, 40, title, 27, weight="bold")
    canvas.text(55, 71, subtitle + " · exploratory; fixed predictions", 18, color="#596b78")
    for k, case in enumerate(CASES):
        canvas.circle(70 + 170*k, 105, 7, COLORS[case])
        canvas.text(85 + 170*k, 111, case, 17)
    canvas.circle(610, 105, 5)
    canvas.text(625, 111, "other contexts", 17)
    canvas.text(930, 111, "Each point is one context", 17, color="#596b78")
    for i, (_, _, label) in enumerate(SPECS):
        left, top = 90 + (i % 3)*465, 191 + (i // 3)*343
        width, height = 338, 235
        xx = lambda v: left + (v + .1)/1.15*width
        yy = lambda v: top + height - (v + .05)/1.1*height
        g = joined.loc[joined.predictor.eq(label)].dropna(subset=[definition, "spearman_rho"])
        result = results.loc[results.predictor.eq(label) & results.word_definition.eq(definition)].iloc[0]
        canvas.text(left, top-41, label, 21, weight="bold")
        p_label = "< .001" if result.holm_p_within_definition_9 < .001 else f"= {result.holm_p_within_definition_9:.3f}"
        canvas.text(left, top-16, f'R = {result.spearman:.3f}   p(Holm) {p_label}   n = {len(g)}', 16, color="#596b78")
        for tick in (0, .25, .5, .75, 1):
            canvas.line(xx(tick), top, xx(tick), top+height)
            canvas.line(left, yy(tick), left+width, yy(tick))
            canvas.text(xx(tick), top+height+24, f"{tick:g}", 15, "middle")
            canvas.text(left-12, yy(tick)+5, f"{tick:g}", 15, "end")
        canvas.line(left, top+height, left+width, top+height, "#84929d")
        canvas.line(left, top, left, top+height, "#84929d")
        draw_order = pd.concat([g.loc[~g.context.isin(CASES)], g.loc[g.context.isin(CASES)]])
        for row in draw_order.itertuples():
            canvas.circle(xx(getattr(row, definition)), yy(row.spearman_rho), 7 if row.context in CASES else 5,
                          COLORS.get(row.context, "#708493"))
        for row in g.loc[g.context.isin(CASES)].itertuples():
            x, y = xx(getattr(row, definition)), yy(row.spearman_rho)
            right = getattr(row, definition) < .85
            canvas.text(x+10 if right else x-10, y-10, row.context, 14,
                        "start" if right else "end", COLORS[row.context])
        if len(g) < 16:
            canvas.text(left, top+height+47, "mask omitted: constant exclusion predictions", 13, color="#596b78")
    canvas.text(744, 1220, "Word-rank Spearman within context (six alternatives)", 21, "middle")
    canvas.text(24, 690, "Exclusion Spearman within context (30 trigger–query pairs)", 21, "middle",
                extra='transform="rotate(-90 24 690)"')
    canvas.text(55, 1260, "R correlates the two context-level measures. Two-sided permutation tests; Holm adjustment across nine predictors.", 16, color="#596b78")
    canvas.save(OUT / filename)


def exception_outputs(words, pairs, source):
    frames, query_tables, diagnostics = [], [], []
    words = words.loc[words.context.isin(CASES)].copy()
    for context in CASES:
        rows = pairs.loc[pairs.context.eq(context)]
        base = rows.loc[rows.predictor.eq("X but not Y"), ["context", "trigger", "query", "human_rate", "human_total"]]
        wide = rows.pivot(index=["context", "trigger", "query"], columns="predictor", values="model_value").reset_index()
        wide = base.merge(wide, on=["context", "trigger", "query"], validate="one_to_one")
        wide["human_yes"] = (wide.human_rate * wide.human_total).round().astype(int)
        for _, _, label in SPECS:
            wide[label + " absolute_error"] = (wide[label] - wide.human_rate).abs()
        frames.append(wide)
        q = wide.groupby(["context", "query"], as_index=False).agg(
            human_mean_item_rate=("human_rate", "mean"), human_yes=("human_yes", "sum"),
            human_total=("human_total", "sum"), human_min_item_rate=("human_rate", "min"),
            human_max_item_rate=("human_rate", "max"), **{label: (label, "mean") for _, _, label in SPECS})
        q["human_pooled_trial_rate"] = q.human_yes / q.human_total
        q = q.rename(columns={"query": "word"}).merge(words.loc[words.context.eq(context)], on=["context", "word"], validate="one_to_one")
        query_tables.append(q)
        h_res = wide.human_rate - wide.groupby("query").human_rate.transform("mean")
        for _, _, label in SPECS:
            p_res = wide[label] - wide.groupby("query")[label].transform("mean")
            diagnostics.append({"context": context, "predictor": label,
                                "mean_absolute_error": (wide[label]-wide.human_rate).abs().mean(),
                                "trigger_residual_pearson_descriptive": pearson(h_res, p_res) if p_res.std() > 1e-12 else np.nan,
                                "trigger_residual_sd": p_res.std()})
    combined, query = pd.concat(frames), pd.concat(query_tables)
    # Keep every pair, not just illustrative successes or failures.
    combined.to_csv(OUT / "exception_all_90_pairs.csv", index=False)
    query.sort_values(["context", "human_rank"]).to_csv(OUT / "exception_word_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "exception_diagnostics.csv", index=False)
    source.loc[source.context_id.isin(CASES), ["context_id", "generation_prompt"]].drop_duplicates().to_json(
        OUT / "exception_prompts.json", orient="records", indent=2)
    return query, combined


def exception_figure(query):
    canvas = SVG(1440, 845)
    canvas.text(45, 42, "Three contexts: alternative ranks and exclusion predictions", 27, weight="bold")
    canvas.text(45, 74, "Historical scores · words ordered by human norming rank within each context", 18, color="#596b78")
    canvas.text(45, 105, "Exclusion rates average the five trigger-specific items for each query; these are not the 30-pair correlations.", 17, color="#596b78")
    columns = [("human_mean_item_rate", "Human", "#172e40"), ("Set Top-K", "Set K", "#237f83"),
               ("Ordering", "Ordering", "#66789e"), ("X but not Y", "But not", "#bc672c")]
    for col, context in enumerate(CASES):
        left = 45 + col*468
        g = query.loc[query.context.eq(context)].sort_values("human_rank")
        canvas.text(left, 151, context.capitalize(), 23, color=COLORS[context], weight="bold")
        for j, name in enumerate(["Word", "Norm", "Viewer", "Sample"]):
            canvas.text(left + [0, 180, 250, 328][j], 181, name, 16, weight="bold")
        for i, row in enumerate(g.itertuples()):
            y = 211 + 29*i
            canvas.text(left, y, row.word, 18)
            for offset, value in zip([195, 273, 353], [row.human_rank, row.model_rank, row.sampling_word_rank]):
                canvas.text(left+offset, y, int(value), 18, "middle")
        canvas.text(left, 399, "Ranks among the six tested words (1 = highest)", 14, color="#596b78")
        x0, top, height, width = left + 43, 455, 240, 344
        for tick in (0, .25, .5, .75, 1):
            y = top + height*(1-tick)
            canvas.line(x0, y, x0+width, y)
            canvas.text(x0-9, y+5, f"{tick:g}", 14, "end")
        for i, (_, row) in enumerate(g.iterrows()):
            x = x0 + 24 + i*59
            # Small horizontal offsets distinguish coincident values; y is exact.
            for k, (field, _, color) in enumerate(columns):
                canvas.circle(x+(k-1.5)*7, top+height*(1-row[field]), 5, color)
            canvas.text(x+3, top+height+23, row.word, 14, "end", extra=f'transform="rotate(-38 {x+3} {top+height+23})"')
    for i, (_, name, color) in enumerate(columns):
        canvas.circle(70+200*i, 803, 6, color)
        canvas.text(85+200*i, 809, name, 17)
    canvas.text(950, 809, "Vertical scale: exclusion rate / prediction", 16, color="#596b78")
    canvas.save(OUT / "exceptions.svg")


def write_stats_report(results):
    lines = ["# Context-level word-ranking / exclusion association", "", "Historical, exploratory fixed-prediction analysis. Not corrected-score results.", "",
             "## Analysis specification", "",
             f"- Seed {SEED}; {PERMUTATIONS:,} random permutations and {BOOTSTRAPS:,} paired context bootstrap samples.",
             "- Primary: across-context Spearman between viewer word-rank agreement and each predictor's exclusion agreement.",
             "- Secondary sensitivity: replace the word measure with ranks recovered from the actual historical sampling scores; exclusion predictions unchanged.",
             "- Two-sided permutation p = (1 + count(|R_perm| >= |R_observed|))/(B + 1), tolerance 1e-12. One vector is permuted across contexts, retaining ties. The test assumes context pairings are exchangeable under the null, conditional on the saved scores.",
             "- Holm correction over all nine predictors within each word-score definition. The machine-readable table also includes a conservative joint 18-test adjustment.",
             "- Bootstrap: resample paired context records with replacement and re-rank within each draw; report unadjusted 95% percentile intervals. No participant resampling, norming-rank resampling, Monte Carlo rerun, or cross-validation refit.",
             "- 16 contexts, except Set Top-p and Conjunction Top-p: 15 because mask has constant predictions and undefined exclusion correlation. Undefined is never replaced by zero.",
             "- Context coefficients rounded to 12 decimals before ranking to preserve mathematical ties despite floating-point arithmetic. Ordering is included once because its Top-K/Top-p predictions are identical.",
             "- Pearson and leave-one-context-out correlations are descriptive sensitivity checks, not extra hypothesis tests. Common-15-context coefficients are supplied for equal-coverage comparisons.",
             "- These are associations between two estimated statistics, not causal tests. The same 213 participants supplied judgments across the 16 contexts, and fitted predictions share training folds. Fixed-summary resampling does not propagate these dependencies or human measurement uncertainty. Six-word norming ranks are treated as fixed.", ""]
    for definition, heading in [("viewer_word_rho", "Primary: viewer word measure"), ("sampling_word_rho", "Sensitivity: sampling-score word measure")]:
        lines += ["## " + heading, "", "| Predictor | Contexts | Spearman R | 95% context-bootstrap interval | Permutation p | Holm p (9) | Leave-one-out range |", "|---|---:|---:|---|---:|---:|---|"]
        for row in results.loc[results.word_definition.eq(definition)].itertuples():
            lines.append(f"| {row.predictor} | {row.n_contexts} | {row.spearman:.3f} | [{row.bootstrap_95_low:.3f}, {row.bootstrap_95_high:.3f}] | {row.permutation_p_two_sided:.5f} | {row.holm_p_within_definition_9:.5f} | [{row.leave_one_out_min:.3f}, {row.leave_one_out_max:.3f}] |")
        lines.append("")
    lines += ["## Interpretation boundaries", "", "Confidence intervals are marginal, unadjusted intervals, so they need not agree with the Holm-adjusted testing decisions. A significant association for one model and a nonsignificant association for another does not establish a difference between their associations. No such between-model difference test was performed.", "",
              "The viewer-word analysis reproduces the user's observation. The sampling-score sensitivity is better aligned with the sampled models' input, but remains based on the historical scorer. Neither should be labeled as corrected-run evidence.", "",
              "## Method references", "", "- [Permutation-test conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html). The absolute-statistic two-sided convention used here is explicit; it differs from SciPy's twice-the-smaller-tail default.",
              "- [Holm multiple-testing adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).", "",
              "## Reproduction", "", "Run `analyze.py` with NumPy and pandas. Run `render.cjs` with the bundled Node executable to render the SVG files to PNG. All outputs are confined to this new directory. `provenance.json` records input hashes and verification checks."]
    (OUT / "STATISTICAL_RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    self_test()
    joined, words, pairs, source, hashes, human = load_data()
    results, loo = test_associations(joined)
    joined.to_csv(OUT / "context_measures.csv", index=False)
    results.to_csv(OUT / "association_tests.csv", index=False)
    loo.to_csv(OUT / "leave_one_context_out.csv", index=False)
    words.to_csv(OUT / "all_96_word_scores.csv", index=False)
    scatterplots(joined, results, "viewer_word_rho", "scatter_viewer.svg")
    scatterplots(joined, results, "sampling_word_rho", "scatter_sampling_scores.svg")
    query, _ = exception_outputs(words, pairs, source)
    exception_figure(query)
    write_stats_report(results)
    unchanged = all(sha(Path(path)) == digest for path, digest in hashes.items())
    assert unchanged
    provenance = {"status": "historical exploratory fixed-prediction analysis", "seed": SEED,
                  "permutations": PERMUTATIONS, "bootstrap_samples": BOOTSTRAPS,
                  "input_sha256": hashes, "analysis_script_sha256": sha(Path(__file__)),
                  "checks": {"self_tests_passed": True, "saved_story_correlations_reproduced": True,
                             "sampling_viewer_source_oof_grid_hashes_match": True, "all_inputs_unchanged": unchanged},
                  "participants": int(human.id.nunique()), "participant_resampling": False,
                  "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(results[["word_definition", "predictor", "n_contexts", "spearman", "bootstrap_95_low", "bootstrap_95_high", "permutation_p_two_sided", "holm_p_within_definition_9"]].to_string(index=False))
    print("All inputs unchanged; source correlations rebuilt; sampling-score provenance verified.")


if __name__ == "__main__":
    main()
