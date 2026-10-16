"""Plot fridge probabilities/ranks and linking-minus-baseline fit in 16 contexts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
RESULT = PIPELINE / 'results/set_variant_qwen_corrected_2026-09-15'
OUTPUT = RESULT / 'advisor_summary/focus_linking_conclusions'
MODELS = [('No linking', 'No linking structure', 'direct'),
          ('X but not Y', 'X but not Y', 'direct'),
          ('Ordering', 'ordering', 'top_k'), ('Set Top-K', 'set', 'top_k')]
METRICS = [('log_score', 'Mean log score'), ('pearson_r', 'Pearson r'), ('spearman_rho', 'Spearman ρ')]
HIGHLIGHTS = [('A', 'water', 'juice', '#0072B2'), ('B', 'juice', 'water', '#D55E00'),
              ('C', 'water', 'milk', '#009E73'), ('D', 'meat', 'milk', '#A44D9E')]
INK, MUTED = '#182B3A', '#546573'
TOL = 1e-12


def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def point_groups(ax, frame, rank=False, small=False, letters=True):
    x, y = ('model_rank', 'human_rank') if rank else ('model_value', 'human_rate')
    grouped = frame.groupby([x, y], sort=False).size()
    unit = 25 if small else 38
    for (px, py), count in grouped.items():
        area = unit + (count-1)*(17 if small else 28)
        ax.scatter(px, py, s=area, color='#718494', edgecolor='white', linewidth=.8, zorder=3)
        if count>1:
            ax.text(px, py, str(count), color='white', ha='center', va='center', fontsize=6 if small else 8, zorder=4)
    # Outlines identify selected pairs without moving their scientific coordinates.
    for letter, trigger, query, color in HIGHLIGHTS:
        r=frame.loc[frame.trigger.eq(trigger)&frame['query'].eq(query)].iloc[0]
        count=grouped.loc[(r[x],r[y])]
        area=unit+(count-1)*(17 if small else 28)+38
        ax.scatter(r[x],r[y],s=area,facecolors='none',edgecolors=color,linewidth=1.8,zorder=5)
        if letters:
            # Keep edge labels within the panel and separate A/C near the upper left.
            offset={'A':(8,-13),'B':(8,6),'C':(8,5),'D':(8,-9)}[letter]
            if not rank and r[x]>.9: offset=(-16,5 if letter=='B' else -11)
            if rank: offset={'A':(9,6),'B':(-14,-13),'C':(8,-12),'D':(8,5)}[letter]
            ax.annotate(letter,(r[x],r[y]),xytext=offset,textcoords='offset points',
                        color=color,weight='bold',fontsize=8 if small else 10,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.1),zorder=6)
    ax.grid(color='#E4E9EE',linewidth=.7,zorder=0)
    ax.set_axisbelow(True)
    for side in ['top','right']:
        ax.spines[side].set_visible(False)
    for side in ['left','bottom']:
        ax.spines[side].set_color('#AFBBC5')
    ax.tick_params(labelsize=8 if small else 10,colors=MUTED)


def draw_fridge(pairs, fits):
    fig=plt.figure(figsize=(17.8,12.9))
    fig.text(.035,.969,'Fridge: linking changes probabilities and sometimes ranks',fontsize=23,weight='bold',color=INK)
    fig.text(.035,.938,'No linking and Set Top-K have identical rank plots, but different probability spacing and Pearson correlations.',fontsize=12.5,color=MUTED)
    fig.text(.035,.910,'Context: “I’m thirsty.” Sam opens the fridge and responds, “Sure, I have …”',fontsize=12,color=INK)
    handles=[Line2D([],[],marker='o',markerfacecolor='none',markeredgecolor=color,markeredgewidth=1.8,
                    linestyle='none',label=f'{letter}   {trigger} → {query}') for letter,trigger,query,color in HIGHLIGHTS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.535,.890),ncol=4,frameon=False,fontsize=11,
               handletextpad=.45,columnspacing=2.7)
    fig.text(.035,.865,'Highlighted pairs',fontsize=10,color=MUTED)
    gs=fig.add_gridspec(2,4,left=.065,right=.97,top=.797,bottom=.285,wspace=.25,hspace=.48)
    axlist=[]
    for col,(label,_,_) in enumerate(MODELS):
        g=pairs.loc[pairs.display_model.eq(label)&pairs.context.eq('fridge')]
        f=fits.loc[fits.context.eq('fridge')&fits.model.eq(label)].iloc[0]
        top=fig.add_subplot(gs[0,col]);bot=fig.add_subplot(gs[1,col]);axlist.append((top,bot))
        point_groups(top,g,letters=col!=0)
        top.plot([0,1],[0,1],linestyle='--',color='#AAB5BD',linewidth=1,zorder=1)
        top.set(xlim=(-.04,1.04),ylim=(-.055,1.055),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Model exclusion probability')
        top.set_title(f'{label}\nPearson r = {f.pearson_r:.3f}',fontsize=14,pad=12,linespacing=1.6,color=INK)
        if col==0:
            top.set_ylabel('Human exclusion rate',fontsize=12)
            inset=top.inset_axes([.37,.11,.59,.62])
            point_groups(inset,g,small=True)
            inset.set(xlim=(-.003,.061),ylim=(-.06,1.06),xticks=[0,.03,.06],yticks=[0,.5,1])
            inset.set_title('Zoom: x = 0–.06',fontsize=9,pad=6)
            inset.tick_params(labelsize=7)
            inset.set_facecolor('#FAFCFD')
        point_groups(bot,g,rank=True)
        bot.plot([0,29],[0,29],linestyle='--',color='#AAB5BD',linewidth=1,zorder=1)
        bot.set(xlim=(-1.5,30.5),ylim=(-1.5,30.5),xticks=[0,10,20,29],yticks=[0,10,20,29],xlabel='Model rank')
        bot.set_title(f'Spearman ρ = {f.spearman_rho:.3f}',fontsize=13,pad=12,color=INK)
        if col==0:bot.set_ylabel('Human rank',fontsize=12)
    fig.text(.065,.238,'Fridge fit measures',fontsize=12,weight='bold',color=INK)
    tabax=fig.add_axes([.065,.106,.905,.115]);tabax.axis('off')
    matrix=fits[fits.context.eq('fridge')].set_index('model').loc[[m[0] for m in MODELS]]
    rows=[[label]+[f'{matrix.loc[m[0],key]:.3f}' for m in MODELS] for key,label in METRICS]
    table=tabax.table(cellText=rows,colLabels=['Fit measure']+[m[0] for m in MODELS],
                      cellLoc='center',colWidths=[.18,.205,.205,.205,.205],bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#DCE3E9');cell.set_linewidth(.6)
        if r==0:cell.set_facecolor('#F0F4F7');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('white')
    for r,(key,_) in enumerate(METRICS,1):
        best=matrix[key].max()
        for c,m in enumerate(MODELS):
            if np.isclose(matrix.loc[m[0],key],best,atol=TOL,rtol=0):
                table[r,c+1].get_text().set_weight('bold')
    fig.text(.035,.066,'Each panel includes all 30 pairs with equal pair weights. Numerals inside circles count coincident pairs; colored outlines mark A–D.',fontsize=10.5,color=MUTED)
    fig.text(.035,.044,'Ranks start at 0 (highest exclusion), with average ranks for ties. Dashed lines mark equal values or ranks. Bold table entries are highest.',fontsize=10.5,color=MUTED)
    fig.text(.035,.022,'Corrected Qwen2-7B results · Set K = 20 · natural-log scores with 10⁻¹⁰ clipping · descriptive estimates, without uncertainty intervals.',fontsize=10.5,color=MUTED)
    save(fig,'fridge_probabilities_and_ranks')


def draw_contexts(deltas):
    fig,axes=plt.subplots(1,3,figsize=(15.8,12.8),sharey=True)
    fig.subplots_adjust(left=.145,right=.965,top=.825,bottom=.255,wspace=.24)
    fig.text(.035,.963,'Linking benefits depend on context and fit measure',fontsize=24,weight='bold',color=INK)
    fig.text(.035,.930,'Novel focus study · each cell is linking fit minus no-linking fit within the same context.',fontsize=13,color=MUTED)
    fig.text(.035,.903,'Blue = higher fit with linking     Red = lower fit with linking     0 (tie) = equal fit',fontsize=12,color=INK)
    contexts=sorted(deltas.context.unique());models=[m[0] for m in MODELS[1:]]
    counts=[]
    for ax,(key,label) in zip(axes,METRICS):
        values=deltas[deltas.metric.eq(key)].pivot(index='context',columns='model',values='difference').loc[contexts,models].to_numpy()
        limit=4 if key=='log_score' else .8
        im=ax.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
        ax.set_title(label,fontsize=16,pad=54,color=INK)
        ax.set_xticks(range(3),['X but not Y','Ordering','Set Top-K'],fontsize=10)
        ax.xaxis.tick_top();ax.tick_params(axis='x',pad=10,length=0)
        ax.set_yticks(range(16),contexts,fontsize=12);ax.tick_params(axis='y',length=0,pad=10)
        for tick in ax.get_yticklabels():
            if tick.get_text()=='fridge':tick.set_weight('bold')
        for y in range(16):
            for x in range(3):
                v=values[y,x]
                text='0 (tie)' if abs(v)<=TOL else f'{v:+.3f}' if abs(v)<.01 else f'{v:+.2f}' if key=='log_score' else f'{v:+.3f}'
                if text in ['+0.000','-0.000']:text=f'{v:+.1e}'
                ax.text(x,y,text,ha='center',va='center',fontsize=10.5,color='white' if abs(v)>.62*limit else INK)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        cax=fig.add_axes([ax.get_position().x0,.204,ax.get_position().width,.014])
        cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-limit,0,limit])
        cb.ax.tick_params(labelsize=10,length=3)
        cb.set_label('Change in nats per pair' if key=='log_score' else 'Change in correlation',fontsize=10)
        for x,m in enumerate(models):
            a=values[:,x];better=int((a>TOL).sum());tied=int((np.abs(a)<=TOL).sum());worse=int((a< -TOL).sum())
            counts.append(dict(metric=key,model=m,better=better,tied=tied,worse=worse))
            frac=(x+.5)/3
            fig.text(ax.get_position().x0+frac*ax.get_position().width,.128,f'{better} / {tied} / {worse}',ha='center',fontsize=11,color=INK)
        fig.text(ax.get_position().x0+ax.get_position().width/2,.151,'Contexts: better / tied / worse',ha='center',fontsize=10,color=MUTED)
    fig.text(.035,.081,'All three measures use the same 30 equally weighted pairs within each context. Pearson and Spearman use matching aggregation here.',fontsize=11,color=MUTED)
    fig.text(.035,.057,'The two correlation panels share a color scale; log score has its own scale. Cell values are rounded; exact ties use tolerance 10⁻¹².',fontsize=10.5,color=MUTED)
    fig.text(.035,.033,'Corrected Qwen2-7B results · saved held-out predictions · Set K = 20 · descriptive differences, without significance or noninferiority tests.',fontsize=10.5,color=MUTED)
    save(fig,'context_fit_differences')
    return pd.DataFrame(counts)


def main():
    OUTPUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,'axes.titlecolor':INK,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    path=RESULT/'spearman/negation_paired_ranks.csv'
    saved=pd.read_csv(path)
    frames=[]; records=[]
    for display,structure,variant in MODELS:
        frame=saved[saved.structure.eq(structure)&saved.variant.eq(variant)].copy()
        frame['display_model']=display
        assert len(frame)==480 and frame.context.nunique()==16
        for context,g in frame.groupby('context'):
            assert len(g)==30 and not g[['trigger','query']].duplicated().any()
            assert g.model_value.between(0,1).all() and g.human_rate.between(0,1).all()
            mr=g.model_value.rank(method='average',ascending=False)-1
            hr=g.human_rate.rank(method='average',ascending=False)-1
            assert np.allclose(mr,g.model_rank,atol=TOL,rtol=0)
            assert np.allclose(hr,g.human_rank,atol=TOL,rtol=0)
            p=g.model_value.clip(1e-10,1-1e-10);h=g.human_rate
            records.append(dict(context=context,model=display,n=30,pearson_r=h.corr(g.model_value),
                                spearman_rho=hr.corr(mr),log_score=(h*np.log(p)+(1-h)*np.log1p(-p)).mean()))
        frames.append(frame)
    pairs=pd.concat(frames,ignore_index=True);fits=pd.DataFrame(records)
    base=frames[0].set_index(['context','trigger','query'])
    for f in frames[1:]:
        aligned=f.set_index(['context','trigger','query']).reindex(base.index)
        assert np.array_equal(aligned.human_rate,base.human_rate)
    fridge=pairs[pairs.context.eq('fridge')]
    pivot=fridge.pivot(index=['trigger','query'],columns='display_model',values='model_rank')
    assert np.array_equal(pivot['No linking'],pivot['Set Top-K'])
    long=fits.melt(id_vars=['context','model','n'],value_vars=[m[0] for m in METRICS],var_name='metric')
    baselines=long[long.model.eq('No linking')][['context','metric','value']].rename(columns={'value':'baseline'})
    deltas=long[~long.model.eq('No linking')].merge(baselines,on=['context','metric'],validate='many_to_one')
    deltas['difference']=deltas.value-deltas.baseline
    previous_path=RESULT/'advisor_summary/baseline_comparison/focus_context_differences.csv'
    previous=pd.read_csv(previous_path).rename(columns={'dataset':'context'})
    matched=deltas.merge(previous,on=['context','model','metric'],suffixes=('','_previous'),validate='one_to_one')
    assert len(matched)==144
    for key in ['value','baseline','difference']:
        assert np.allclose(matched[key],matched[key+'_previous'],atol=TOL,rtol=0)
    pairs.to_csv(OUTPUT/'plotted_pair_values.csv',index=False)
    fits.to_csv(OUTPUT/'context_fit_values.csv',index=False)
    deltas.to_csv(OUTPUT/'context_fit_differences.csv',index=False)
    draw_fridge(pairs,fits)
    counts=draw_contexts(deltas)
    counts.to_csv(OUTPUT/'comparison_counts.csv',index=False)
    sources=[path,previous_path]
    (OUTPUT/'provenance.json').write_text(json.dumps(dict(
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        verified_differences=144,fridge_baseline_set_rank_vectors_identical=True,
        within_context_pairs=30,contexts=16,rank_origin=0,rank_ties='average',clip_epsilon=1e-10,
        highlighted_pairs=[dict(label=a,trigger=b,query=c) for a,b,c,_ in HIGHLIGHTS],
        interpretation='Descriptive saved point estimates; no refitting or inference.'),indent=2)+'\n')
    print(f'Verified all 144 within-context differences. Saved both figures to {OUTPUT}')
    print(counts.to_string(index=False))


if __name__=='__main__':
    main()
