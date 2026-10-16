"""Three publication figure candidates for the word-ranking sensitivity analysis.

Uses saved corrected held-out probabilities without fitting or resampling.
Requires numpy, pandas, and matplotlib. PNG, PDF, and editable-text SVG outputs.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PIPELINE = OUT.parents[1]
SOURCE = PIPELINE / "results/set_variant_qwen_corrected_2026-09-15/spearman"
BLUE = "#176B92"
GRAY = "#66717D"
LIGHT = "#BEC6CD"
INK = "#24313C"
GRID = "#E8EDF0"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": GRAY,
    "ytick.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.titleweight": "bold", "axes.titlesize": 10,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

CAPTIONS = {
"option_1_subset_comparison": """Exclusion prediction after restricting contexts by model-human word-ranking agreement. (A) Spearman and (B) Pearson correlations between predicted exclusion probabilities and observed human rates, computed across the 30 ordered trigger-query pairs in each context and then averaged equally across contexts. Circles indicate No linking and squares indicate Set Top-K. Rows show the full dataset (16 contexts, 480 pairs), contexts with word-ranking Spearman > .9 (7 contexts, 210 pairs), and contexts with word-ranking Spearman = 1 (2 contexts, 60 pairs). Word rankings compare neutral continuation scores with human norming ranks for six tested alternatives. Subsets are nested; points are descriptive means, not confidence intervals. Predictions retain the original corrected held-out evaluation. Pearson measures numerical association, not calibration.""",
"option_2_context_improvements": """Context-level changes in exclusion correlation after applying Set Top-K. Each row is one of the seven contexts with word-ranking Spearman > .9. Points show Set Top-K minus No linking for (A) Spearman and (B) Pearson exclusion correlations, each calculated across 30 ordered trigger-query pairs. Diamonds identify the two contexts with perfect word-ranking agreement (bakery and fridge); circles identify the five contexts with word-ranking Spearman = .943 (rounded). Positive values favor Set Top-K. Both panels use identical horizontal limits. Values are descriptive point differences, not significance tests or confidence intervals; predictions are unchanged from the corrected held-out evaluation.""",
"option_3_alignment_and_gain": """Word-ranking agreement and the change in exclusion Pearson correlation from No linking to Set Top-K across all 16 contexts. The horizontal axis shows Spearman agreement between the model's ranking and human norming ranks of six tested words. The vertical axis shows the difference in within-context exclusion Pearson correlation (Set Top-K minus No linking), based on 30 ordered trigger-query pairs per context. Blue circles identify contexts with word-ranking Spearman > .9 and < 1; blue diamonds identify the two contexts with perfect agreement. Gray circles show other contexts. The vertical dashed line marks the strict .9 selection cutoff; the horizontal dashed line marks no improvement. Points use their exact coordinates without jitter. Labels use leader lines where needed. The display is descriptive; no trend line or inferential test is fitted.""",
}


def style(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=.7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=6)


def save(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=220 if ext == "png" else 300)
    plt.close(fig)


def collect():
    wp = SOURCE / "word_spearman_by_context.csv"
    pp = SOURCE / "negation_paired_ranks.csv"
    word = pd.read_csv(wp).set_index("context")
    pred = pd.read_csv(pp)
    records = []
    for context, g in pred.groupby("context"):
        a = g[g.structure.eq("No linking structure")].sort_values("item_id")
        b = g[g.structure.eq("set") & g.variant.eq("top_k")].sort_values("item_id")
        assert len(a) == len(b) == 30
        assert a.item_id.tolist() == b.item_id.tolist()
        assert np.array_equal(a.human_rate.to_numpy(), b.human_rate.to_numpy())
        record = {"context": context, "word_rho": word.loc[context, "spearman_rho"]}
        for label, data in [("no_linking", a), ("set_top_k", b)]:
            record[label + "_pearson"] = data.model_value.corr(data.human_rate)
            record[label + "_spearman"] = data.model_value.rank().corr(data.human_rate.rank())
        for metric in ["pearson", "spearman"]:
            record["delta_" + metric] = record["set_top_k_" + metric] - record["no_linking_" + metric]
        records.append(record)
    data = pd.DataFrame(records).set_index("context")
    assert len(data) == 16 and data.word_rho.gt(.9).sum() == 7
    assert np.isclose(data.word_rho, 1, rtol=0, atol=1e-12).sum() == 2
    for folder, selector in [
        ("perfect_word_ranking_2026-10-08", np.isclose(data.word_rho, 1, rtol=0, atol=1e-12)),
        ("word_ranking_gt_0_9_2026-10-08", data.word_rho.gt(.9)),
    ]:
        saved = pd.read_csv(PIPELINE / "diagnostics" / folder / "correlation_summary.csv").set_index("model")
        for model, prefix in [("No linking", "no_linking"), ("Set Top-K", "set_top_k")]:
            for metric in ["pearson", "spearman"]:
                assert abs(data.loc[selector, prefix + "_" + metric].mean() - saved.loc[model, "mean_within_" + metric]) < 1e-12
    data.to_csv(OUT / "figure_data.csv")
    (OUT / "provenance.json").write_text(json.dumps({
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [wp, pp]},
        "evaluation": "Equal-item within-context correlations, arithmetic equal-context means; average ranks for ties. Original held-out predictions; no refitting. Strict word rho > .9; perfect rho uses roundoff tolerance 1e-12.",
        "validation": "Recomputed subset means match both saved mini-analysis summaries within 1e-12.",
    }, indent=2) + "\n")
    return data


def option1(data):
    selections = [data, data[data.word_rho.gt(.9)], data[np.isclose(data.word_rho, 1, rtol=0, atol=1e-12)]]
    labels = ["All contexts\n16 contexts / 480 pairs", "Word-ranking $\\rho > .9$\n7 contexts / 210 pairs", "Word-ranking $\\rho = 1$\n2 contexts / 60 pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.05), sharey=True)
    fig.subplots_adjust(left=.265, right=.975, top=.79, bottom=.30, wspace=.26)
    fig.suptitle("Exclusion prediction when alternative rankings agree", y=.975, fontsize=12, fontweight="bold")
    for ax, metric, title, xlabel in zip(axes, ["spearman", "pearson"],
                                       ["A  Exclusion rankings", "B  Exclusion rates"],
                                       ["Mean within-context\nSpearman $\\rho$", "Mean within-context\nPearson $r$"]):
        style(ax)
        ax.set_title(title, loc="left", pad=16)
        for y, subset in enumerate(selections):
            a, b = [subset[prefix + "_" + metric].mean() for prefix in ["no_linking", "set_top_k"]]
            ax.plot([a, b], [y, y], color=LIGHT, lw=2, zorder=2)
            ax.scatter(a, y, s=61, facecolors="white", edgecolors=GRAY, linewidths=1.4, zorder=3)
            ax.scatter(b, y, s=24, marker="s", color=BLUE, zorder=4)
            ax.annotate(f"{a:.3f}", (a,y), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", color=GRAY, fontsize=8.5)
            ax.annotate(f"{b:.3f}", (b,y), xytext=(0,-11), textcoords="offset points", ha="center", va="top", color=BLUE, fontsize=8.5)
        ax.set_xlim(.45, 1.0)
        ax.set_xticks([.5, .6, .7, .8, .9, 1.0])
        ax.set_ylim(2.5, -.5)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel(xlabel, labelpad=10, fontsize=9)
    handles = [Line2D([],[],marker="o",ls="",markerfacecolor="white",markeredgecolor=GRAY,markersize=6,label="No linking"),
               Line2D([],[],marker="s",ls="",color=BLUE,markersize=5,label="Set Top-K")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.62,.065),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.025,"Equal-context means; subsets overlap. Descriptive estimates without confidence intervals.",ha="center",fontsize=7.5,color=GRAY)
    save(fig,"option_1_subset_comparison")


def option2(data, model_label="Set Top-K", output_name="option_2_context_improvements",
            xlimits=(-.10, .355), xticks=(-.1, 0, .1, .2, .3)):
    subset = data[data.word_rho.gt(.9)].sort_index()
    fig, axes = plt.subplots(1,2,figsize=(8.0,4.55),sharey=True)
    fig.subplots_adjust(left=.16,right=.975,top=.78,bottom=.24,wspace=.19)
    fig.suptitle(f"Where does {model_label} improve exclusion prediction?",y=.975,fontsize=12,fontweight="bold")
    fig.text(.5,.914,"Seven contexts with word-ranking Spearman > .9",ha="center",fontsize=9,color=GRAY)
    for ax,metric,title in zip(axes,["spearman","pearson"],["A  Change in Spearman", "B  Change in Pearson"]):
        style(ax)
        ax.set_title(title,loc="left",pad=12)
        ax.axvline(0,color=GRAY,lw=1,ls=(0,(3,3)))
        for y,(c,row) in enumerate(subset.iterrows()):
            value=row["delta_"+metric]
            perfect=np.isclose(row.word_rho,1,rtol=0,atol=1e-12)
            ax.plot([0,value],[y,y],color=LIGHT,lw=1.5,zorder=2)
            ax.scatter(value,y,s=41 if perfect else 33,marker="D" if perfect else "o",color=BLUE,edgecolors="white",lw=.5,zorder=3)
            ax.annotate(f"{value:+.3f}", (value,y),xytext=(8 if value>=0 else -8,0),textcoords="offset points",ha="left" if value>=0 else "right",va="center",fontsize=8,color=BLUE)
        ax.set_xlim(*xlimits)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: "0" if abs(x)<1e-10 else f"{x:+.2f}"))
        ax.set_ylim(6.65,-.65)
        ax.set_yticks(range(len(subset)),[c.title() for c in subset.index])
        ax.set_xlabel(f"{model_label} minus No linking",labelpad=9)
    handles=[Line2D([],[],marker="o",ls="",color=BLUE,markersize=5,label="Word-ranking $\\rho=.943$"),
             Line2D([],[],marker="D",ls="",color=BLUE,markersize=5,label="Word-ranking $\\rho=1$")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.56,.075),ncol=2,frameon=False,fontsize=8.5)
    fig.text(.5,.035,f"Positive values favor {model_label}. Each correlation uses 30 pairs; both panels share the same scale.",ha="center",fontsize=7.5,color=GRAY)
    save(fig,output_name)


def option3(data):
    fig,ax=plt.subplots(figsize=(8.0,5.0))
    fig.subplots_adjust(left=.12,right=.965,top=.83,bottom=.27)
    fig.suptitle("Does the numerical benefit remain when word rankings agree?",y=.97,fontsize=11.5,fontweight="bold")
    style(ax,axis="both")
    ax.axvspan(.9,1.06,color=BLUE,alpha=.045,zorder=0)
    ax.axvline(.9,color=BLUE,ls=(0,(4,3)),lw=.9,zorder=1)
    ax.axhline(0,color=GRAY,ls=(0,(4,3)),lw=1,zorder=1)
    # Labels have fixed display positions; data points are not jittered.
    label_positions={
        "cold":(-.005,.027),"hot":(.19,-.031),"meat":(.28,-.077),
        "mask":(.34,.216),"mall":(.50,-.042),"cut":(.63,.072),
        "throw":(.64,.025),"restaurant":(.72,-.034),"transport":(.74,.108),
        "bag":(1.09,.049),"bakery":(1.09,-.001),"beach":(.83,-.065),
        "fridge":(1.09,.270),"gym":(.97,.143),"salad":(1.09,-.045),
        "science":(1.09,.095),
    }
    # Additional right margin keeps the closely aligned contexts legible.
    ax.set_xlim(-.08,1.28)
    ax.set_ylim(-.115,.31)
    ax.set_xticks([0,.2,.4,.6,.8,.9,1.0])
    ax.set_yticks([-.1,0,.1,.2,.3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: "0" if abs(x)<1e-10 else f"{x:+.1f}"))
    for c,row in data.iterrows():
        perfect=np.isclose(row.word_rho,1,rtol=0,atol=1e-12)
        selected=row.word_rho>.9
        ax.scatter(row.word_rho,row.delta_pearson,s=48 if perfect else 35,marker="D" if perfect else "o",facecolor=BLUE if selected else LIGHT,edgecolor=INK if perfect else "white",lw=.7,zorder=4)
        pos=label_positions[c]
        ax.annotate(c.title(),xy=(row.word_rho,row.delta_pearson),xytext=pos,textcoords="data",ha="left",va="center",fontsize=8,color=BLUE if selected else GRAY,
                    arrowprops=dict(arrowstyle="-",color=BLUE if selected else LIGHT,lw=.6,shrinkA=2,shrinkB=4),
                    bbox=dict(facecolor="white",edgecolor="none",pad=.6),zorder=5)
    ax.set_xlabel("Model-human word-ranking Spearman $\\rho$",labelpad=10)
    ax.set_ylabel("Change in exclusion Pearson $r$\n(Set Top-K minus No linking)",labelpad=12)
    handles=[Line2D([],[],marker="o",ls="",color=LIGHT,markersize=5,label="$\\rho\\leq .9$ (9 contexts)"),
             Line2D([],[],marker="o",ls="",color=BLUE,markersize=5,label="$.9<\\rho<1$ (5 contexts)"),
             Line2D([],[],marker="D",ls="",color=BLUE,markeredgecolor=INK,markersize=5,label="$\\rho=1$ (2 contexts)")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.55,.095),ncol=3,frameon=False,fontsize=8,handletextpad=.4,columnspacing=1.4)
    fig.text(.5,.05,"All 16 contexts shown. Dashed lines mark no change and the .9 selection cutoff; points are not jittered.",ha="center",fontsize=7.5,color=GRAY)
    save(fig,"option_3_alignment_and_gain")


if __name__ == "__main__":
    data = collect()
    option1(data)
    option2(data)
    option3(data)
    (OUT / "captions.txt").write_text("\n\n".join(name+"\n"+caption for name,caption in CAPTIONS.items())+"\n")
    print("Created three figures in PNG, PDF, and SVG; subset means verified against saved analyses.")
    print(OUT)
