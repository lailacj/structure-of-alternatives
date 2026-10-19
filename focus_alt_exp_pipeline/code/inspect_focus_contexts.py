"""Inspect four corrected novel-focus contexts without refitting or resampling."""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
RESULTS = PIPELINE / 'results/set_variant_qwen_corrected_2026-09-15'
OUT = RESULTS / 'advisor_summary/four_context_inspection'
CONTEXTS = ['fridge', 'throw', 'bag', 'meat']
MODELS = {'No linking': ('No linking structure', 'direct'),
          'X but not Y': ('X but not Y', 'direct'),
          'Ordering': ('ordering', 'top_k'), 'Set Top-K': ('set', 'top_k')}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [RESULTS/'spearman/negation_paired_ranks.csv',
               RESULTS/'spearman/word_paired_ranks.csv', RESULTS/'inputs/source_rows.csv',
               PIPELINE/'results_viewer/index.html']
    pairs, words, source = [pd.read_csv(p) for p in sources[:3]]
    viewer = json.loads(re.search(r'<script id="results-data" type="application/json">(.*?)</script>', sources[3].read_text(), re.S).group(1))
    # Never mix historical distribution exports with corrected predictions.
    for name, sub in [('source_rows.csv', 'inputs'), ('oof_predictions.csv', 'cv_results'), ('prediction_grid.csv', '')]:
        p = RESULTS/sub/name
        assert digest(p) in {x['sha256'] for x in viewer['provenance'] if Path(x['path']).name == name}
    all_trials, all_metrics, all_words, all_candidates, audits = [], [], [], [], []
    for context in CONTEXTS:
        s = source[source.dataset_family.eq('novel_focus') & source.context_id.eq(context)]
        prompt = next(p for p in viewer['prompts'] if p['frame']=='Neutral' and p['contexts']==[context])
        assert s.generation_prompt.nunique()==1 and prompt['text']==s.generation_prompt.iloc[0]
        expected = set(s.neutral_score_array_sha256)
        assert expected=={x['sha256'] for x in viewer['provenance'] if Path(x['path']).name == prompt['id']+'.log_probs.npy'}
        w = words[words.context.eq(context)].sort_values('human_rank').copy()
        order = w.word.tolist()
        n = pairs[pairs.context.eq(context)]
        base = n[n.structure.eq('No linking structure') & n.variant.eq('direct')].set_index(['trigger','query'])
        frame = base[['human_rate','human_total']].copy()
        assert len(frame)==30 and not frame.index.duplicated().any()
        for label, (structure, variant) in MODELS.items():
            g=n[n.structure.eq(structure)&n.variant.eq(variant)].set_index(['trigger','query']).reindex(frame.index)
            assert len(g)==30 and g.model_value.notna().all() and np.array_equal(g.human_rate,frame.human_rate)
            frame[label]=g.model_value
            p=frame[label].clip(1e-10,1-1e-10); h=frame.human_rate
            log=h*np.log(p)+(1-h)*np.log1p(-p)
            all_metrics.append(dict(context=context,model=label,pearson=h.corr(frame[label]),
                spearman=h.rank().corr(frame[label].rank()),log_score=log.sum()))
            frame[label+' rank']=frame[label].rank(method='average',ascending=False)
            frame[label+' log score']=log
            frame[label+' absolute error']=(h-frame[label]).abs()
        frame['Set absolute error reduction']=frame['No linking absolute error']-frame['Set Top-K absolute error']
        frame['Set log score gain']=frame['Set Top-K log score']-frame['No linking log score']
        frame['context']=context
        all_trials.append(frame.reset_index())
        w['raw_continuation_probability']=np.exp(w.model_value)
        q=frame.reset_index().groupby('query')
        assert q['No linking'].nunique().eq(1).all() and q['Set Top-K'].nunique().eq(1).all()
        w['set_probability']=w.word.map(q['Set Top-K'].first())
        all_words.append(w)
        for candidate in prompt['distribution']:
            all_candidates.append(dict(context=context,**candidate))
        audits.append(dict(context=context,exact_prompt=prompt['text'],support_size=prompt['supportSize'],
            baseline_set_identical_rank_vector=bool(np.array_equal(frame['No linking rank'],frame['Set Top-K rank'])),
            baseline_unique_predictions=int(frame['No linking'].nunique()),set_unique_predictions=int(frame['Set Top-K'].nunique()),
            set_zero_queries=w.loc[w.set_probability.eq(0),'word'].tolist()))
        draw(context,prompt,w,frame,pd.DataFrame(all_metrics).query('context == @context'),order)
    pd.concat(all_trials).to_csv(OUT/'all_120_trial_comparisons.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(OUT/'context_metrics.csv',index=False)
    pd.concat(all_words).to_csv(OUT/'tested_word_rankings.csv',index=False)
    pd.DataFrame(all_candidates).to_csv(OUT/'saved_candidate_continuations.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(dict(contexts=audits,sources=[dict(path=str(p),sha256=digest(p)) for p in sources],
        method='30 pairs per context; log scores sum item scores without division; Spearman average ranks for ties; probabilities clipped at 1e-10; no resampling or refitting'),indent=2)+'\n')
    print(pd.DataFrame(all_metrics).round(4).to_string(index=False))
    print('Saved', OUT)


def draw(context,prompt,w,frame,metrics,order):
    fig=plt.figure(figsize=(16,14.5),layout=None)
    fig.patch.set_facecolor('white')
    fig.text(.045,.971,context.capitalize()+' | Prompt, continuations and human judgments',fontsize=22,weight='bold',va='top')
    fig.text(.045,.938,textwrap.fill(prompt['text'],135),fontsize=11.5,va='top',linespacing=1.35)
    fig.text(.045,.855,'Leading scored continuations',fontsize=13,weight='bold')
    candidates=prompt['distribution'][:10]
    # Candidate strings may overlap; these are weights normalized over saved support.
    for i,c in enumerate(candidates):
        col=i//5; row=i%5
        fig.text(.045+col*.19,.832-row*.021,f"{c['rank']:2d}. {c['word']:<14} {c['normalized']*100:6.2f}%",fontfamily='monospace',fontsize=10)
    fig.text(.045,.715,'Percent = sampling mass over 121,301 scored strings,\nincluding overlapping one- and two-word continuations.',fontsize=9,color='#4A5568')
    fig.text(.49,.855,'Six tested alternatives',fontsize=13,weight='bold')
    fig.text(.49,.831,f"{'Word':<12} {'Human rank':>10} {'Qwen rank':>10} {'Raw P':>10} {'Set P':>8}",fontsize=10,fontfamily='monospace')
    for i,r in enumerate(w.itertuples()):
        fig.text(.49,.808-i*.021,f"{r.word:<12} {int(r.human_rank)+1:>10} {int(r.model_rank)+1:>10} {r.raw_continuation_probability:>10.3g} {r.set_probability:>8.3f}",fontsize=10,fontfamily='monospace')
    fig.text(.49,.666,'Ranks refer to the six tested words; 1 = highest.\nHuman ranks are the experiment’s reference relevance ranks.',fontsize=9,color='#4A5568')
    gs=fig.add_gridspec(2,3,left=.08,right=.94,bottom=.15,top=.61,wspace=.39,hspace=.64)
    labels=['Human judgments','No linking','Set Top-K','X but not Y','Ordering','Set minus human']
    for k,label in enumerate(labels):
        ax=fig.add_subplot(gs[k//3,k%3])
        key='human_rate' if label=='Human judgments' else label
        values=frame['Set Top-K']-frame.human_rate if label=='Set minus human' else frame[key]
        m=values.unstack('query').reindex(index=order,columns=order).to_numpy()
        residual=label=='Set minus human'
        cmap=plt.get_cmap('RdBu_r' if residual else 'Blues').copy();cmap.set_bad('#ECEFF2')
        im=ax.imshow(np.ma.masked_invalid(m),cmap=cmap,vmin=-1 if residual else 0,vmax=1)
        ax.set_xticks(range(6),order,rotation=35,ha='right',fontsize=9)
        ax.set_yticks(range(6),order,fontsize=9)
        ax.set_xlabel('Queried alternative',fontsize=10);ax.set_ylabel('Spoken trigger',fontsize=10)
        title=label
        if label in MODELS:
            r=metrics.set_index('model').loc[label]
            title+=f"\nr = {r.pearson:.3f}   ρ = {r.spearman:.3f}   Σ log = {r.log_score:.3f}"
        else:title+='\nRate and response count' if label=='Human judgments' else '\nRed = too high; blue = too low'
        ax.set_title(title,fontsize=11,pad=10,linespacing=1.6)
        for y,t in enumerate(order):
            for x,q in enumerate(order):
                if t==q:continue
                v=m[y,x]
                if label=='Human judgments':
                    annotation=f'{v:.2f}\nn={int(frame.loc[(t,q),"human_total"])}'
                elif residual:annotation=f'{v:+.2f}'
                else:annotation='<.01' if 0<v<.005 else f'{v:.2f}'
                ax.text(x,y,annotation,ha='center',va='center',fontsize=8.5,color='white' if (abs(v)>.65 if residual else v>.65) else '#1B263B')
        ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.045,.038,'All 30 distinct trigger–query pairs are shown. Model scores weight pairs equally; human response counts vary.\nNo linking uses raw continuation probability. Set uses saved Top-K inclusion probability (K=20). Predictions and ranks are distinct from human responses.\nCorrected Qwen2-7B run; descriptive point estimates. Small nonzero probabilities display as <.01; gray diagonal pairs were not tested.',fontsize=10,color='#4A5568',linespacing=1.45)
    for extension in ['png','pdf']:
        fig.savefig(OUT/f'{context}_inspection.{extension}',dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
