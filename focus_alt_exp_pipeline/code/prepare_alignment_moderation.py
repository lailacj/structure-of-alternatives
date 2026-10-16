"""Validate participant responses and join saved novel-focus predictions/moderator."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/set_variant_qwen_corrected_2026-09-15'
OUT=RESULT/'advisor_summary/alignment_moderation'
MODELS={'no_linking':('No linking structure','direct'), 'x_but_not_y':('X but not Y','direct'),
        'ordering':('ordering','top_k'), 'set_top_k':('set','top_k')}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'human_exp_data/sca_dataframe.csv',RESULT/'spearman/negation_paired_ranks.csv',
           RESULT/'spearman/word_paired_ranks.csv',RESULT/'spearman/word_spearman_by_context.csv']
    human,pred,words,alignment=[pd.read_csv(p) for p in files]
    human=human.rename(columns={'id':'participant','story':'context','cleaned_trigger':'trigger_word','cleaned_query':'query_word'})
    for col in ['context','trigger_word','query_word']:
        human[col]=human[col].str.strip().str.lower()
    assert len(human)==3408 and human.participant.nunique()==213 and human.context.nunique()==16
    assert not human[['participant','context']].duplicated().any()
    assert human.groupby('participant').context.nunique().eq(16).all()
    assert human.neg.isin([0,1]).all() and human.participant.notna().all()
    human['item']=human.context+'::'+human.trigger_word+'::'+human.query_word
    assert human.item.nunique()==480
    for context,g in words.groupby('context'):
        ref=human[human.context.eq(context)][['trigger_word','trigger_relevance']].drop_duplicates().set_index('trigger_word').trigger_relevance
        assert len(ref)==6 and set(ref)==set(range(6))
        assert np.array_equal(g.human_rank,ref.reindex(g.word))
        rho=g.human_rank.corr(g.model_rank)
        assert np.isclose(rho,alignment[alignment.context.eq(context)].spearman_rho.iloc[0],atol=1e-12,rtol=0)
    alignment=alignment[['context','spearman_rho']].rename(columns={'spearman_rho':'alignment'})
    alignment['alignment_c']=alignment.alignment-alignment.alignment.mean()
    frames=[];scales=[]; summaries=[]
    counts=human.groupby('item').neg.agg(['size','mean'])
    for model,(structure,variant) in MODELS.items():
        p=pred[pred.structure.eq(structure)&pred.variant.eq(variant)].copy()
        assert len(p)==480 and not p.item_id.duplicated().any()
        p=p.set_index('item_id').reindex(counts.index)
        assert np.array_equal(counts['size'],p.human_total)
        assert np.allclose(counts['mean'],p.human_rate,atol=1e-12,rtol=0)
        assert p.model_value.between(0,1).all()
        p=p.reset_index().rename(columns={'item_id':'item','model_value':'prediction'})
        center=p.prediction.mean();scale=p.prediction.std(ddof=1)
        joined=human[['participant','context','item','trigger_word','query_word','neg']].merge(p[['item','prediction','cv_fold','boundary']],on='item',validate='many_to_one').merge(alignment,on='context',validate='many_to_one')
        assert len(joined)==3408 and joined[['prediction','alignment']].notna().all().all()
        joined['prediction_z']=(joined.prediction-center)/scale
        joined['model']=model
        frames.append(joined)
        scales.append(dict(model=model,prediction_mean=center,prediction_sd=scale,alignment_mean=alignment.alignment.mean(),
                           prediction_scale_basis='480 equally weighted pairs',alignment_scale_basis='16 equally weighted contexts'))
        for ctx,g in joined.groupby('context'):
            summaries.append(dict(model=model,context=ctx,n=len(g),alignment=g.alignment.iloc[0],p_min=g.prediction.min(),p_max=g.prediction.max(),human_rate=g.neg.mean()))
    pd.concat(frames,ignore_index=True).to_csv(OUT/'analysis_data.csv',index=False)
    pd.DataFrame(scales).to_csv(OUT/'scaling.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'context_support.csv',index=False)
    alignment.to_csv(OUT/'context_alignment.csv',index=False)
    (OUT/'input_audit.json').write_text(json.dumps(dict(
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        responses_per_model=3408,participant_ids=213,contexts=16,unique_pairs=480,
        participant_assumption='id is treated as participant: each of 213 IDs appears once in every context',
        human_rate_count_matches=True,word_reference_rank_matches=True,models=list(MODELS),
        moderator_source='six independent reference relevance ranks versus corrected neutral Qwen word ranks',
        filtering='none; all contexts and all binary responses retained',
        prediction_source='saved direct baselines and held-out ordering/Set predictions; no new K selection'),indent=2)+'\n')
    print('Validated 3408 responses per model, 213 participant IDs, 16 contexts and 480 pairs; all joins complete.')


if __name__=='__main__':main()
