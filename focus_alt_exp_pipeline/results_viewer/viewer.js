/* Offline viewer: no fetches, external libraries, or model fitting. */
"use strict";
const ResultsMath = (() => {
  const mean = xs => xs.length ? xs.reduce((a,b) => a+b,0)/xs.length : null;
  function pearson(xs,ys) {
    if(xs.length<2)return null;
    const mx=mean(xs),my=mean(ys),dx=xs.map(x=>x-mx),dy=ys.map(y=>y-my);
    const denominator=Math.sqrt(dx.reduce((s,x)=>s+x*x,0)*dy.reduce((s,y)=>s+y*y,0));
    return denominator ? Math.max(-1,Math.min(1,dx.reduce((s,x,i)=>s+x*dy[i],0)/denominator)) : null;
  }
  function ranks(values){
    const order=values.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v),out=Array(values.length);
    for(let a=0;a<order.length;){let b=a+1;while(b<order.length&&order[b].v===order[a].v)b++;
      for(let j=a;j<b;j++)out[order[j].i]=(a+b-1)/2;a=b;}
    return out;
  }
  const spearman=(xs,ys)=>pearson(ranks(xs),ranks(ys));
  function summary(items,model) {
    const rows=items.filter(r=>r.p[model]!==null);
    const ids=[...new Set(items.map(r=>r.dataset))],eligible=ids.length===1&&/^(hu_|rnx_)/.test(ids[0]);
    return {rho:eligible?spearman(rows.map(r=>r.p[model]),rows.map(r=>r.y)):null,n:rows.length,r:pearson(rows.map(r=>r.p[model]),rows.map(r=>r.y)),
      log:mean(rows.map(r=>r.scores[model])),human:mean(rows.map(r=>r.y)),prediction:mean(rows.map(r=>r.p[model]))};
  }
  function aggregate(items,model,balanced=true) {
    const rows=items.filter(r=>r.p[model]!==null),ids=[...new Set(rows.map(r=>r.dataset))];
    const groups=ids.map(id=>summary(rows.filter(r=>r.dataset===id),model)),stats=summary(rows,model);
    if(balanced){stats.log=mean(groups.map(g=>g.log));stats.r=mean(groups.map(g=>g.r).filter(x=>x!==null));}
    return {...stats,datasets:ids.length,definedR:groups.filter(g=>g.r!==null).length};
  }
  const escape = value => String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function distribution(items,model) {
    const rows=items.filter(r=>r.p[model]!==null);
    function stats(values) {
      const sorted=[...values].sort((a,b)=>a-b),n=values.length,counts=Array(20).fill(0);
      for(const value of values){
        if(!Number.isFinite(value)||value<0||value>1)throw new Error("Distribution values must be in [0, 1]");
        counts[Math.min(19,Math.floor(value*20))]++;
      }
      return {counts,mean:mean(values),median:n?(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2:null,
        zeros:values.filter(v=>v===0).length,ones:values.filter(v=>v===1).length};
    }
    return {n:rows.length,human:stats(rows.map(r=>r.y)),model:stats(rows.map(r=>r.p[model]))};
  }
  return {mean,pearson,ranks,spearman,summary,aggregate,escape,distribution};
})();
if(typeof module!=="undefined")module.exports=ResultsMath;
if(typeof document!=="undefined") (()=>{
const D=JSON.parse(document.getElementById("results-data").textContent);
const {mean,summary,aggregate,escape:esc}=ResultsMath;
const state={view:"overview",wordSource:"sampling",spearmanModel:"set|top_k",distributionModel:"all",distributionContext:"all",metric:"log",dataset:"novel_focus",model:2,context:"bag",balanced:true,coverage:"all",promptDataset:"novel_focus",promptContext:"all",frame:"Neutral",promptId:"",candidateSource:"distribution",wordScale:"logp",wordSearch:"",itemSearch:"",page:0,item:null};
const main=document.getElementById("main");
const labels=Object.fromEntries(D.datasets.map(d=>[d.id,d.label]));
const short=["No linking","X but not Y","Set Top-K","Set Top-p","Ordering","Conj. Top-K","Conj. Top-p","Disj. Top-K","Disj. Top-p"];
const colors=["#155bcc","#087f87","#946721","#8646a7","#3d6897","#c15b53","#5c7993","#7d674a","#666bae","#008389"];
const descriptions=[
  "Uses exp(query continuation log probability) from the neutral prompt directly as the model’s exclusion probability.",
  "Uses exp(query continuation log probability) after the X-but-not-Y prompt. Structurally unavailable for the five R&X conditions.",
  "Probability that the query appears among the first K candidates of a sampled ordering. K is selected using training-fold Set log score.",
  "Probability that the query appears in the shortest sampled prefix whose original normalized candidate probabilities reach p. This is a sampled prefix, not a deterministic list of the most probable words.",
  "Probability that the query precedes the trigger in the sampled ordering. Independent of the Top-K or Top-p boundary.",
  "Probability that the query is in the Top-K set AND precedes the trigger. Reuses the K selected for Set in the training folds.",
  "Probability that the query is in the Top-p set AND precedes the trigger. Reuses the p selected for Set in the training folds.",
  "Probability that the query is in the Top-K set OR precedes the trigger. Reuses the K selected for Set in the training folds.",
  "Probability that the query is in the Top-p set OR precedes the trigger. Reuses the p selected for Set in the training folds."
];
const fmt=(v,n=3)=>v===null||v===undefined?"N/A":Number(v).toFixed(n);
const prob=v=>v===null||v===undefined?"N/A":v!==0&&v<.001?Number(v).toExponential(3):Number(v).toFixed(4);
const integer=v=>Number(v).toLocaleString("en-US");
const title=(eyebrow,heading,sub,action="")=>`<div class="topline"><div><div class="eyebrow">${eyebrow}</div><h1>${heading}</h1><p class="muted">${sub}</p></div>${action}</div>`;
const stat=(value,label,small=false)=>`<div class="stat"><div class="value ${small?"small-value":""}">${value}</div><div class="label">${label}</div></div>`;
const select=(id,label,options,value)=>`<label for="${id}">${label}<select id="${id}">${options.map(([v,l])=>`<option value="${esc(v)}" ${String(value)===String(v)?"selected":""}>${esc(l)}</option>`).join("")}</select></label>`;
const supportsDatasetSpearman=()=>state.view==="structures"||(state.view==="datasets"&&state.dataset!=="novel_focus");
const metricControl=()=>select("metric","Fit measure",[["log","Mean proper log score"],["r","Pearson correlation"],...(supportsDatasetSpearman()?[["rho","Spearman correlation"]]:[])],state.metric);
const modelControl=()=>select("model","Linking structure",D.models.map((m,i)=>[i,m]),state.model);
const datasetControl=()=>select("dataset","Dataset",D.datasets.map(d=>[d.id,d.label]),state.dataset);
const contextControl=()=>select("context","Focus context",D.contexts.map(c=>[c,c]),state.context);
const wordSourceControl=()=>select("wordSource","Word-ranking score source",[["sampling","Actual sampling scores (default)"],["target","Separate direct-target scores (comparison)"]],state.wordSource);
const wordSourceLabel=()=>state.wordSource==="sampling"?"actual sampling-array scores":"separate direct-target scores";
const activeSpearman=()=>state.wordSource==="sampling"?D.spearman:D.targetScoreSpearman;
const metricLabel=()=>({log:"Mean proper log score",r:"Pearson r",rho:"Spearman ρ"})[state.metric];
const downloadButton=(kind,label="Download CSV")=>`<button data-download="${kind}">${label}</button>`;
const scopeItems=()=>state.coverage==="shared"?D.items.filter(r=>r.p[1]!==null):D.items;
const spearmanScopeNote="Spearman ranks scales within each Hu dataset and the 60 items separately within each R&X condition. It uses the same items as Pearson, with average ranks for ties. van Tiel template probabilities are averaged before ranking. Focus Spearman remains within-context on its separate page.";
const metricNote=()=>state.metric==="rho"?`<p class="caption metric-note">${spearmanScopeNote}</p>`:`<p class="caption metric-note">${state.metric==="log"?"Log score measures probability accuracy and penalizes confident errors. Higher (closer to 0) is better. Item probabilities are clipped to [10⁻¹⁰, 1 − 10⁻¹⁰] before scoring.":"Pearson r measures linear association between model exclusion probabilities and human exclusion rates. It does not measure calibration. An undefined correlation is shown as N/A."}</p>`;
function metricColor(value,metric){
  if(value===null)return "#edf0f4";
  const t=metric!=="log"?Math.max(0,Math.min(1,(value+1)/2)):Math.max(0,Math.min(1,(value+8)/8));
  const a=t<.5?[243,189,175]:[240,243,248],b=t<.5?[240,243,248]:[74,148,205],f=t<.5?t*2:(t-.5)*2;
  return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*f)).join(",")})`;
}
function heatmap(rows,kind){
  return `<div class="table-wrap"><table class="heatmap"><caption class="sr-only">${metricLabel()} by ${kind} and linking structure</caption><thead><tr><th scope="col">${kind==="dataset"?"Dataset":"Context"}</th>${short.map(m=>`<th scope="col">${m}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr><th scope="row"><button data-${kind}="${esc(row.id)}">${esc(row.label)} <span class="caption">· ${row.stats[0].n}</span></button></th>${row.stats.map((s,m)=>`<td style="background:${metricColor(s[state.metric],state.metric)}" class="${s[state.metric]===null?"na":""}"><button data-cell="${kind}|${esc(row.id)}|${m}" title="${esc(D.models[m])}: ${s.n===0?"not applicable":s[state.metric]===null?"correlation undefined":fmt(s[state.metric],6)}; N=${s.n}">${s.n===0?"—":fmt(s[state.metric],2)}</button></td>`).join("")}</tr>`).join("")}</tbody></table></div><div class="legend"><span>${state.metric!=="log"?"−1":"−8"}</span><span class="gradient"></span><span>${state.metric!=="log"?"+1":"0"} · higher is better</span><span class="legend-spacer">${state.metric==="log"?"Fixed color scale; values below −8 use the lowest color. ":""}— not applicable · N/A undefined · click a cell to inspect</span></div>`;
}
function bars(records,metric,click="model"){
  const vals=records.map(r=>r.value).filter(v=>v!==null),lo=metric!=="log"?Math.min(0,...vals,-.1):Math.min(-.1,...vals),hi=metric!=="log"?Math.max(.1,...vals):0,range=hi-lo;
  const zero=(0-lo)/range*100;
  return records.map(r=>`<div class="rank-row"><button class="rank-label" data-${click}="${esc(r.id)}">${esc(r.label)}</button><div class="bar-track"><span class="bar-zero" style="left:${zero}%"></span>${r.value!==null?`<span class="bar-fill" style="left:${Math.min(zero,(r.value-lo)/range*100)}%;width:${Math.abs(r.value)/range*100}%;${r.value<0&&metric!=="log"?"background:#b76559":""}"></span>`:""}</div><span class="rank-value">${fmt(r.value)}</span></div>`).join("")+`<p class="caption">${metric==="log"?"Bars extend left from 0. Shorter bars indicate better log scores.":"Bars extend from 0; right is positive, left is negative."}</p>`;
}
function metricTable(stats){const withRho=state.view==="datasets"&&state.dataset!=="novel_focus";return `<div class="table-wrap"><table><thead><tr><th>Linking structure</th><th>N</th><th>Pearson r</th>${withRho?"<th>Spearman ρ</th>":""}<th>Mean log score</th><th>Mean model p</th></tr></thead><tbody>${stats.map((s,m)=>`<tr class="${m===state.model?"selected-row":""}"><th scope="row"><button class="item-link" data-model="${m}">${esc(D.models[m])}</button></th><td>${s.n||"—"}</td><td>${s.n?fmt(s.r):"—"}</td>${withRho?`<td>${s.n?fmt(s.rho):"—"}</td>`:""}<td>${s.n?fmt(s.log):"—"}</td><td>${s.n?fmt(s.prediction):"—"}</td></tr>`).join("")}</tbody></table></div>`;}
function datasetRankPlot(rows,model){
  const valid=rows.filter(r=>r.p[model]!==null);
  if(!valid.length)return `<div class="empty">X-but-not-Y is not applicable to this condition.</div>`;
  const human=ResultsMath.ranks(valid.map(r=>r.y)),prediction=ResultsMath.ranks(valid.map(r=>r.p[model]));
  const paired=valid.map((r,i)=>({...r,human_rank:human[i],model_rank:prediction[i],human_rate:r.y,model_value:r.p[model]}));
  return rankScatter(paired,"items")+`<p class="caption">${valid.length} analysis items · Spearman ρ = ${fmt(ResultsMath.spearman(valid.map(r=>r.y),valid.map(r=>r.p[model])))}. Higher rates rank first (0). Circle counts show tied coordinates; the dashed line marks equal ranks.</p><details><summary>Inspect the ${valid.length} item ranks</summary><div class="table-wrap"><table><thead><tr><th>Item</th><th>Human rate</th><th>Prediction</th><th>Human rank</th><th>Model rank</th></tr></thead><tbody>${paired.map(r=>`<tr><th>${esc(r.id)}</th><td>${fmt(r.y)}</td><td>${fmt(r.model_value)}</td><td>${fmt(r.human_rank,1)}</td><td>${fmt(r.model_rank,1)}</td></tr>`).join("")}</tbody></table></div></details>`;
}
function scatter(rows,model){
  const valid=rows.filter(r=>r.p[model]!==null),s=summary(rows,model);
  if(!valid.length)return `<div class="empty">X-but-not-Y is not applicable to this dataset.</div>`;
  const x=p=>62+p*468,y=p=>350-p*316;
  return `<svg class="chart" viewBox="0 0 560 410" role="img" aria-label="Model exclusion probability on the horizontal axis and human exclusion rate on the vertical axis, with ${valid.length} items. Exact values are in the item table below.">${[0,.25,.5,.75,1].map(v=>`<line class="chart-grid" x1="62" x2="530" y1="${y(v)}" y2="${y(v)}"/><line class="chart-grid" x1="${x(v)}" x2="${x(v)}" y1="34" y2="350"/><text x="${x(v)}" y="374" text-anchor="middle">${v}</text><text x="48" y="${y(v)+4}" text-anchor="end">${v}</text>`).join("")}<line class="reference" x1="62" x2="530" y1="350" y2="34"/>${valid.map(r=>`<circle data-item="${esc(r.dataset+"|"+r.id)}" cx="${x(r.p[model])}" cy="${y(r.y)}" r="4.3"><title>${esc(r.context?r.context+": ":"")}${esc(r.trigger)} → ${esc(r.query)}\nModel: ${fmt(r.p[model],6)} · Human: ${fmt(r.y,6)}\nLog score: ${fmt(r.scores[model],6)}</title></circle>`).join("")}<text x="300" y="403" text-anchor="middle">Model exclusion probability</text><text transform="translate(16 195) rotate(-90)" text-anchor="middle">Human exclusion rate</text></svg><div class="scatter-meta"><span><strong>${fmt(s.r)}</strong>Pearson r</span>${state.view==="datasets"&&state.dataset!=="novel_focus"?`<span><strong>${fmt(s.rho)}</strong>Spearman ρ</span>`:""}<span><strong>${fmt(s.log)}</strong>Mean log score</span><span><strong>${s.n}</strong>Analysis units</span></div><p class="caption">Dashed line: p = human rate. Hover for values or select a point for its prompts. Items at the same coordinates overlap; every item is listed below.</p>`;
}
function distributionPlot(d,model,ymax){
  if(!d.n)return `<div class="empty">Not applicable: X-but-not-Y predictions are unavailable for this dataset.</div>`;
  const color=colors[model],pct=v=>`${(v*100).toFixed(1)}%`;
  const x=i=>62+i*23.4,y=v=>320-v/ymax*260;
  let path=`M ${x(0)} 320 L ${x(0)} ${y(d.model.counts[0]/d.n)}`;
  d.model.counts.forEach((n,i)=>{path+=` L ${x(i+1)} ${y(n/d.n)}`;if(i<19)path+=` L ${x(i+1)} ${y(d.model.counts[i+1]/d.n)}`;});
  path+=` L ${x(20)} 320`;
  return `<p class="caption">Model mean ${pct(d.model.mean)} · median ${pct(d.model.median)}<br>Human mean ${pct(d.human.mean)} · median ${pct(d.human.median)} · ${d.n} matched units</p>
  <svg class="chart distribution-chart" viewBox="0 0 560 390" role="img" aria-label="${esc(D.models[model])}: human and model distributions across ${d.n} matched analysis units, with 5 percentage point bins.">
  ${Array.from({length:6},(_,i)=>{const v=i*ymax/5;return `<line class="chart-grid" x1="62" x2="530" y1="${y(v)}" y2="${y(v)}"/><text x="52" y="${y(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`;}).join("")}
  ${d.human.counts.map((n,i)=>`<rect x="${x(i)}" y="${y(n/d.n)}" width="23.4" height="${320-y(n/d.n)}" fill="#c8cdd3" stroke="white" stroke-width=".6"/>`).join("")}
  <path d="${path}" fill="none" stroke="${color}" stroke-width="3"/>
  ${d.human.counts.map((n,i)=>`<rect x="${x(i)}" y="60" width="23.4" height="260" fill="transparent"><title>${i*5}%–${(i+1)*5}%${i===19?" (includes 100%)":" (upper bound excluded)"}: Human ${n} (${pct(n/d.n)}); Model ${d.model.counts[i]} (${pct(d.model.counts[i]/d.n)})</title></rect>`).join("")}
  ${[0,4,8,12,16,20].map(i=>`<text x="${x(i)}" y="344" text-anchor="middle">${i*5}%</text>`).join("")}
  <rect x="62" y="15" width="20" height="10" fill="#c8cdd3"/><text x="90" y="25">Human rates</text><rect x="250" y="15" width="20" height="10" fill="none" stroke="${color}" stroke-width="3"/><text x="278" y="25">Model predictions</text>
  <text x="296" y="380" text-anchor="middle">Exclusion probability / human exclusion rate</text><text transform="translate(16 190) rotate(-90)" text-anchor="middle">Share of analysis units</text></svg>
  <p class="caption">Model exactly 0: ${d.model.zeros} · exactly 1: ${d.model.ones}</p>`;
}
function distributionRows(){
  return D.items.filter(r=>r.dataset===state.dataset&&(state.dataset!=="novel_focus"||state.distributionContext==="all"||r.context===state.distributionContext));
}
function distributions(){
  const context=state.dataset==="novel_focus"?state.distributionContext:"all";
  const rows=distributionRows(),all=D.models.map((_,m)=>ResultsMath.distribution(rows,m));
  // Keep the scale fixed across all models in this dataset, even when filtering.
  const peak=Math.max(...all.filter(d=>d.n).flatMap(d=>[...d.human.counts,...d.model.counts].map(n=>n/d.n)));
  const ymax=Math.min(1,Math.max(.1,Math.ceil((peak+0.001)*10)/10));
  return title("Model and human distributions","Where do the probabilities concentrate?","Compare each model’s predictions with human exclusion rates on the same analysis units.",downloadButton("distributions","Download histogram data"))+
    `<div class="controls">${datasetControl()}${state.dataset==="novel_focus"?select("distributionContext","Focus context",[["all","All contexts pooled"],...D.contexts.map(c=>[c,c])],state.distributionContext):""}${select("distributionModel","Model",[["all","All nine models"],...D.models.map((m,i)=>[i,m])],state.distributionModel)}</div>
    <p class="dataset-description">${context==="all"?datasetDescription(state.dataset):`Context: <strong>${esc(context)}</strong> · ${rows.length} ordered trigger–query pairs. Human and model distributions use only this context, with equal weight per pair.`}</p>
    <p class="caption">Gray bars: human rates. Colored outlines: model predictions. Bins are 5 percentage points wide: [0%, 5%), …, [95%, 100%]. Each matched analysis unit has equal weight. All models in the selected dataset/context share identical axes, including when filtering by model. Hover over a bin for counts. Human values are item or scale rates, not individual binary responses.</p>
    <div class="grid-two">${all.map((d,m)=>state.distributionModel!=="all"&&Number(state.distributionModel)!==m?"":`<section class="panel"><h2 style="color:${colors[m]}">${esc(D.models[m])}${context!=="all"?` · ${esc(context)}`:""}</h2>${distributionPlot(d,m,ymax)}</section>`).join("")}</div>`;
}
function overview(){
  const rows=scopeItems(),stats=D.models.map((_,m)=>aggregate(rows,m,state.balanced));
  const comparable=stats.map((s,m)=>({...s,m})).filter(s=>s.datasets===new Set(rows.map(r=>r.dataset)).size&&s[state.metric]!==null).sort((a,b)=>b[state.metric]-a[state.metric]);
  const winner=comparable[0],novel=D.datasetSummaries.novel_focus;
  const bestNovel=novel.map((s,m)=>({...s,m})).filter(s=>s[state.metric]!==null).sort((a,b)=>b[state.metric]-a[state.metric])[0];
  const baseIds=[...new Set(rows.map(r=>r.dataset))],base=state.balanced?mean(baseIds.map(id=>D.baselines[id]).filter(v=>v!==undefined)):mean(rows.map(r=>D.baselines[r.dataset]).filter(v=>v!==undefined));
  return title("Cross-dataset comparison","How do the linking structures fit?","Compare the overall pattern, then follow a dataset or context down to individual items.",downloadButton("overview"))+
  `<div class="stats">${stat("10","datasets · 9 linking structures")}${stat(integer(D.items.length),"analysis units · "+integer(D.sourceRows)+" source rows")}${stat("16","focus-study contexts · 480 item pairs")}${stat("360","neutral prompts scored on the cluster")}</div>`+
  `<div class="controls">${metricControl()}${select("aggregation","Aggregate summary",[["balanced","Equal dataset weight (existing log-score method)"],["pooled","Pooled analysis units (descriptive)"]],state.balanced?"balanced":"pooled")}${select("coverage","Comparison coverage",[["all","All 10 datasets"],["shared","5 datasets with all 9 structures"]],state.coverage)}</div>`+
  `<div class="grid-two wide-left"><section class="panel"><div class="panel-heading"><div><h2>${state.balanced?(state.metric!=="log"?"Mean within-dataset Pearson r":"Dataset-balanced log score"):(state.metric!=="log"?"Pooled Pearson r":"Pooled item log score")}</h2><p class="caption">${state.balanced?"Each dataset contributes equally.":"Each analysis unit contributes equally; the focus study contributes 480 units."}</p></div></div>${bars(stats.map((s,m)=>({id:m,label:short[m],value:s[state.metric]})),state.metric)}${state.metric==="log"&&base!==null?`<p class="caption">Fold-safe dataset base-rate reference: <strong>${fmt(base)}</strong> on the selected coverage.</p>`:""}</section><section class="panel"><h2>What these results show</h2><p><strong>${esc(D.models[winner.m])}</strong> has the highest ${state.metric!=="log"?"correlation summary":"log score"} among structures covering every dataset in this comparison: <strong>${fmt(winner[state.metric])}</strong>.</p><p>Within the novel focus study, <strong>${esc(D.models[bestNovel.m])}</strong> has the highest ${metricLabel().toLowerCase()}: <strong>${fmt(bestNovel[state.metric])}</strong>.</p>${state.metric==="log"&&base!==null?`<p>The strongest structure scores ${fmt(winner.log-base)} relative to the fold-safe dataset base rate. A positive difference favors the structure.</p>`:""}<p class="caption">These are descriptive rankings. No confidence intervals or significance tests are supplied by this result bundle.</p>${state.coverage==="all"?`<div class="note amber">X-but-not-Y covers only 5 datasets (693 units). Its aggregate uses those 5 and is excluded from the all-10 ranking. Select shared coverage to compare all nine structures on the same datasets.</div>`:""}${state.metric!=="log"?`<div class="note">The saved pipeline reports Pearson r separately by dataset. The aggregate correlation shown here is an additional descriptive summary: ${state.balanced?"the arithmetic mean of defined dataset correlations, not a pooled correlation or Fisher-z meta-analysis":"one correlation across all selected analysis units"}.</div>`:""}${metricNote()}</section></div>`+
  `<section class="panel"><div class="panel-heading"><div><h2>Datasets × linking structures</h2><p class="caption">All ${D.verifiedCells} saved metric cells are checked against the item-level rebuild.</p></div></div>${heatmap(D.datasets.filter(d=>state.coverage!=="shared"||!d.id.startsWith("rnx_")).map(d=>({id:d.id,label:d.label,stats:D.datasetSummaries[d.id]})),"dataset")}</section>`;
}
function datasetDescription(id){
  if(id==="novel_focus")return "480 ordered trigger–query pairs across 16 contexts. Human exclusion rates come from exact response counts; summary scores weight item pairs equally.";
  if(id==="hu_vt16")return "39 scales retained by the published Hu string-analysis filter. Model probabilities and human rates are averaged over templates within scale before evaluation.";
  if(id.startsWith("rnx_"))return "60 item-condition observations. X-but-not-Y is structurally unavailable. Matched ESI/Eonly and Estrong/Eonlystrong conditions share neutral scoring prompts.";
  return "Published Hu string-analysis subset, evaluated at scale level. The viewer uses the same retained scales and probability aggregation as the current pipeline.";
}
function itemRowsForView(){return state.view==="contexts"?D.items.filter(r=>r.context===state.context&&r.dataset==="novel_focus"):D.items.filter(r=>r.dataset===state.dataset);}
function itemTable(rows){
  const query=state.itemSearch.toLowerCase();
  const filtered=rows.filter(r=>(r.id+" "+r.context+" "+r.trigger+" "+r.query).toLowerCase().includes(query));
  const pages=Math.max(1,Math.ceil(filtered.length/20));state.page=Math.min(state.page,pages-1);
  const visible=filtered.slice(state.page*20,(state.page+1)*20);
  return `<section class="panel"><div class="panel-heading"><div><h2>Individual items</h2><p class="caption">${esc(D.models[state.model])} · select an item to compare all structures and read its prompts.</p></div>${downloadButton("items","Download these items")}</div><label for="itemSearch">Find a trigger, query, context, or item<input id="itemSearch" type="search" value="${esc(state.itemSearch)}" placeholder="e.g. wallet, mask, good"></label><div class="table-wrap"><table><thead><tr><th>Trigger → query</th><th>Context / item</th><th>Human rate</th><th>Model p</th><th>Log score</th><th>Fold</th></tr></thead><tbody>${visible.map(r=>`<tr><td><button class="item-link" data-item="${esc(r.dataset+"|"+r.id)}">${esc(r.trigger)} → ${esc(r.query)}</button></td><td>${esc(r.context||r.id)}</td><td>${fmt(r.y)}</td><td>${fmt(r.p[state.model])}</td><td>${fmt(r.scores[state.model])}</td><td>${r.fold}</td></tr>`).join("")}</tbody></table></div>${!visible.length?`<div class="empty">No matching items.</div>`:""}<div class="pager"><span>${integer(filtered.length)} items · page ${state.page+1} of ${pages}</span><div class="inline"><button data-page="-1" ${state.page===0?"disabled":""}>Previous</button><button data-page="1" ${state.page>=pages-1?"disabled":""}>Next</button></div></div>${itemDetail()}</section>`;
}
function itemDetail(){
  if(!state.item)return "";
  const row=D.items.find(r=>r.dataset+"|"+r.id===state.item);
  if(!row||!itemRowsForView().includes(row))return "";
  return `<div class="detail" id="item-detail"><div class="panel-heading"><div><div class="eyebrow">${esc(labels[row.dataset])} ${esc(row.context)}</div><h2>${esc(row.trigger)} → ${esc(row.query)}</h2></div><button data-close-item="1">Close detail</button></div><p>Human exclusion rate <strong>${fmt(row.y,4)}</strong> · held-out fold ${row.fold} · K=${row.k}, p=${row.topP}</p><div class="table-wrap"><table><thead><tr><th>Structure</th><th>Model exclusion probability</th><th>Item log score</th></tr></thead><tbody>${D.models.map((m,i)=>`<tr><th>${esc(m)}</th><td>${fmt(row.p[i],6)}</td><td>${fmt(row.scores[i],6)}</td></tr>`).join("")}</tbody></table></div><h3 style="margin-top:20px">Source rows & prompts</h3>${row.sources.map(index=>{const s=D.sources[index],p=D.prompts.find(p=>p.id===s.prompt);return `<div class="item-prompt"><div><strong>${esc(s.item)}</strong> · ${s.countStatus==="exact"?`${s.yes} / ${s.total} exclusion responses`:`published rate only (${fmt(s.human)}); no response counts imputed`}</div><p>${esc(p.text)}</p><div class="inline"><button data-prompt="${esc(s.prompt)}">Inspect neutral next-word scores</button>${s.framedPrompt?`<button data-prompt="${esc(s.framedPrompt)}">Inspect X-but-not-Y scores</button>`:""}</div><p class="caption">Human source: ${esc(s.humanFile)}<br>Score source: ${esc(s.scoreFile)}</p></div>`;}).join("")}</div>`;
}
function datasets(){
  const rows=itemRowsForView(),stats=D.datasetSummaries[state.dataset];
  return title("Dataset detail",esc(labels[state.dataset]),datasetDescription(state.dataset))+
  `<div class="controls">${datasetControl()}${modelControl()}${metricControl()}</div><div class="stats">${stat(rows.length,"analysis units")}${state.dataset==="novel_focus"?stat(fmt(mean(rows.map(r=>r.y))),"mean human exclusion rate"):stat(fmt(stats[state.model].rho),"selected structure · Spearman ρ")}${stat(fmt(stats[state.model].r),"selected structure · Pearson r")}${stat(fmt(stats[state.model].log),"selected structure · mean log score")}</div><div class="grid-two"><section class="panel"><h2>${esc(D.models[state.model])}</h2><p class="caption">${descriptions[state.model]}</p>${state.metric==="rho"?datasetRankPlot(rows,state.model):scatter(rows,state.model)}</section><section class="panel"><h2>Compare all nine structures</h2>${bars(stats.map((s,m)=>({id:m,label:short[m],value:s[state.metric]})),state.metric)}${metricNote()}${state.dataset==="novel_focus"?`<button data-view="contexts">Explore all 16 contexts</button>`:""}</section></div><section class="panel"><div class="panel-heading"><h2>${state.dataset==="novel_focus"?"Dataset-level fit measures":"Three fit measures"}</h2>${downloadButton("dataset")}</div>${metricTable(stats)}${state.dataset!=="novel_focus"?`<p class="caption">${spearmanScopeNote}</p>`:""}</section>${itemTable(rows)}`;
}
function structures(){
  const m=state.model,records=D.datasets.filter(d=>state.metric!=="rho"||d.id!=="novel_focus").map(d=>({id:d.id,label:d.label,value:D.datasetSummaries[d.id][m][state.metric]}));
  return title("Linking-structure detail",esc(D.models[m]),descriptions[m],downloadButton("structure"))+
  `<div class="controls">${modelControl()}${metricControl()}</div><div class="grid-two"><section class="panel"><h2>${metricLabel()} by dataset</h2>${bars(records,state.metric,"dataset")}${metricNote()}</section><section class="panel"><h2>Dataset-level results</h2><div class="table-wrap"><table><thead><tr><th>Dataset</th><th>N</th><th>Pearson r</th><th>Spearman ρ</th><th>Mean log score</th></tr></thead><tbody>${D.datasets.map(d=>{const s=D.datasetSummaries[d.id][m];return `<tr><th><button class="item-link" data-dataset="${d.id}">${esc(d.label)}</button></th><td>${s.n||"—"}</td><td>${s.n?fmt(s.r):"—"}</td><td>${d.id==="novel_focus"?"See context page":s.n?fmt(s.rho):"—"}</td><td>${s.n?fmt(s.log):"—"}</td></tr>`;}).join("")}</tbody></table></div><div class="note">${spearmanScopeNote} Correlation and log score answer different questions. A structure can track the ordering of human rates while making poorly calibrated probability predictions.</div></section></div>${state.metric==="rho"?`<section class="panel"><button data-view="spearman">Explore focus within-context Spearman</button></section>`:`<section class="panel"><h2>Within the novel focus study</h2><p class="caption">All 16 contexts · ${esc(D.models[m])} · select a context to inspect its item pairs.</p>${bars(D.contexts.map(c=>({id:c,label:c,value:D.contextSummaries[c][m][state.metric]})),state.metric,"context")}</section>`}`;
}
function contexts(){
  const rows=itemRowsForView(),stats=D.contextSummaries[state.context];
  return title("Novel focus alternative study","Sixteen experimental contexts","Compare contexts, then inspect the 30 trigger–query pairs within each context.",downloadButton("contexts"))+
  `<div class="controls">${contextControl()}${modelControl()}${metricControl()}</div><section class="panel"><h2>Contexts × linking structures</h2>${heatmap(D.contexts.map(c=>({id:c,label:c,stats:D.contextSummaries[c]})),"context")}</section><div class="grid-two"><section class="panel"><div class="eyebrow">${esc(state.context)}</div><h2>${esc(D.models[state.model])}</h2>${scatter(rows,state.model)}</section><section class="panel"><h2>All structures in ${esc(state.context)}</h2>${metricTable(stats)}<p class="caption">Context correlations are computed within this context’s items; context log scores average its item scores. These summaries use the saved held-out predictions.</p><button data-context-prompts="${esc(state.context)}">Read this context’s prompts & next words</button></section></div>${itemTable(rows)}`;
}
function rankScatter(rows,kind){
  const max=rows.length-1, left=64,top=24,size=278;
  const x=r=>left+r/max*size,y=r=>top+size-r/max*size;
  const groups=new Map();
  for(const row of rows){const key=`${row.human_rank}|${row.model_rank}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
  const ticks=kind==="words"?[0,1,2,3,4,5]:kind==="pairs"?[0,5,10,15,20,25,29]:[0,...Array.from({length:5},(_,i)=>Math.round((i+1)*max/5))];
  const label=kind==="words"?"Six word ranks":kind==="pairs"?"Thirty trigger–query pair ranks":`${rows.length} analysis item ranks`;
  return `<svg class="chart spearman-chart" viewBox="0 0 400 370" role="img" aria-label="${label}: human rank horizontally, model rank vertically; zero is best">
    <title>${label}</title>
    ${ticks.map(t=>`<line class="chart-grid" x1="${x(t)}" y1="${top}" x2="${x(t)}" y2="${top+size}"/><line class="chart-grid" x1="${left}" y1="${y(t)}" x2="${left+size}" y2="${y(t)}"/><text x="${x(t)}" y="${top+size+22}" text-anchor="middle">${t}</text><text x="${left-12}" y="${y(t)+4}" text-anchor="end">${t}</text>`).join("")}
    <line class="reference" x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}"/>
    ${[...groups.values()].map(group=>{const r=group[0],n=group.length;
      const details=group.map(v=>kind==="words"?`${v.word}: human rank ${v.human_rank}, model rank ${v.model_rank}`:`${v.trigger} → ${v.query}: human rate ${fmt(v.human_rate)}, prediction ${fmt(v.model_value)}; human rank ${v.human_rank}, model rank ${v.model_rank}`).join("; ");
      return `<g class="rank-point" tabindex="0" role="img" aria-label="${esc(details)}" data-rank-count="${n}"><title>${esc(details)}</title><circle cx="${x(r.human_rank)}" cy="${y(r.model_rank)}" r="${5*Math.sqrt(n)}" style="r:${5*Math.sqrt(n)}px"/>${kind==="words"?`<text class="word-rank-label" x="${x(r.human_rank)+(r.human_rank>3?-10:10)}" y="${y(r.model_rank)-10}" text-anchor="${r.human_rank>3?"end":"start"}">${esc(r.word)}</text>`:n>1?`<text class="rank-count" x="${x(r.human_rank)}" y="${y(r.model_rank)+4}" text-anchor="middle">${n}</text>`:""}</g>`;
    }).join("")}
    <text x="${left+size/2}" y="350" text-anchor="middle">Human rank (0 = highest)</text>
    <text transform="translate(17 163) rotate(-90)" text-anchor="middle">Model rank (0 = highest)</text>
  </svg>`;
}
function spearman(){
  if(!D.spearman)return title("Focus alternatives","Within-context Spearman","Rebuild the viewer to calculate these results.");
  const table=(rows,columns)=>`<div class="table-wrap"><table><thead><tr>${columns.map(([key,label])=>`<th>${esc(label)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${columns.map(([key])=>`<td>${typeof row[key]==="number"?fmt(row[key],Number.isInteger(row[key])?0:3):esc(row[key]??"—")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const S=activeSpearman(),c=state.context;
  const names={set:"Set",ordering:"Ordering",conjunction:"Conjunction",disjunction:"Disjunction"};
  const modelLabel=r=>`${names[r.structure]||r.structure}${r.variant==="direct"?"":r.variant==="top_k"?" · Top-K":" · Top-p"}`;
  const negResults=S.negation_spearman_by_context.filter(r=>r.context===c);
  const selected=negResults.find(r=>`${r.structure}|${r.variant}`===state.spearmanModel)||negResults[0];
  const words=S.word_paired_ranks.filter(r=>r.context===c);
  const neg=S.negation_paired_ranks.filter(r=>r.context===c&&r.structure===selected.structure&&r.variant===selected.variant);
  const wordResult=S.word_spearman_by_context.find(r=>r.context===c);
  const fridgeWord=S.word_spearman_by_context.find(r=>r.context==="fridge");
  const fridgeNeg=S.negation_spearman_by_context.find(r=>r.context==="fridge"&&r.structure==="set"&&r.variant==="top_k");
  return title("Novel focus alternative study","Within-context Spearman","Two questions about the same context: how well are words ranked, and how well are human negation responses ranked?")+
    `<section class="panel spearman-explainer"><h2>Word ranking and exclusion are different comparisons</h2><p>In fridge, word-ranking Spearman from the ${wordSourceLabel()} is <strong>${fmt(fridgeWord.spearman_rho)}</strong>. For Set Top-K, negation Spearman is <strong>${fmt(fridgeNeg.spearman_rho)}</strong>. The first measure ranks six words; the second ranks 30 trigger–query pairs. Agreement about contextual word order does not determine the adequacy of a linking rule.</p><div class="inline"><button data-spearman-example="fridge">Show the fridge example</button><button data-view="association">Compare these measures across all contexts</button></div></section>
    <div class="controls">${contextControl()}${select("spearmanModel","Negation model",negResults.map(r=>[`${r.structure}|${r.variant}`,modelLabel(r)]),state.spearmanModel)}${wordSourceControl()}</div>
    <div class="grid-two spearman-comparison">
      <section class="panel"><div class="eyebrow">${esc(c)} · six words</div><h2>Word-ranking Spearman · ${esc(c)}</h2><div class="spearman-coefficient">ρ = ${fmt(wordResult.spearman_rho)}</div><p>Does the model put the six alternatives in the human relevance order?</p>${words.length===6?rankScatter(words,"words"):`<div class="empty">Sampling scores are missing for this context. Direct-target scores are not substituted.</div>`}<p class="caption">Each dot is one word. Human rank comes from trigger_relevance; model rank comes from the ${wordSourceLabel()}. This comparison stays the same when you change the negation model. Normalization over a common support does not change the within-prompt word order.</p></section>
      <section class="panel"><div class="eyebrow">${esc(c)} · 30 trigger–query pairs</div><h2>Negation Spearman · ${esc(c)}</h2><div class="spearman-coefficient">ρ = ${fmt(selected.spearman_rho)}</div><p>${esc(modelLabel(selected))}: does the model put pairs in the same order as the human negation rates?</p>${rankScatter(neg,"pairs")}<p class="caption">Each circle represents one or more pairs at exactly the same rank coordinates. Numbers inside circles count overlapping pairs; circle area reflects that count. Hover for pair details, or inspect the table below.</p>${selected.status!=="defined"?`<div class="note amber">Undefined correlation: human rates or model predictions are constant within this context, so their ranks have no variation.</div>`:""}</section>
    </div>
    <p class="caption">Each plot uses its own rank range: 0–5 for six words and 0–29 for 30 pairs. The dashed line marks equal human and model ranks. Spearman is the correlation of the two rank columns; ties receive average ranks.</p>
    <section class="panel"><h2>How the negation ranks are calculated</h2><p>For each trigger–query pair, the human rate is the fraction of participants with <code>neg = 1</code>. We match it to that pair’s model prediction, then rank all 30 human rates and all 30 model probabilities separately within the context. The six trigger_relevance ranks are used only for the word-ranking measure.</p><p>Higher rates receive lower ranks, starting at 0. Ties share their average position: in fridge, eleven pairs have human rate 1.000, occupying positions 0–10, so each gets rank 5. A rank of 5 here can mean “tied for highest.” Each pair contributes once, regardless of its participant count.</p><p>Set predictions depend on query membership, so the same query can have the same prediction across different triggers even when the human rates differ. Ordering, Conjunction, and Disjunction can depend on both words. Sampled models use held-out predictions; boundaries are still selected using training log score.</p>
      <details><summary>Inspect the six word ranks in ${esc(c)}</summary>${table(words,[["word","Word"],["human_rank","Human rank"],["model_value","Model log probability"],["model_rank","Model rank"],...(state.wordSource==="sampling"?[["vocabulary_rank","Full-support rank"]]:[])])}</details>
      <details><summary>Inspect all 30 paired values and ranks · ${esc(modelLabel(selected))}</summary>${table(neg,[["trigger","Trigger"],["query","Query"],["human_total","Participants"],["human_rate","Human rate"],["model_value","Prediction"],["human_rank","Human rank"],["model_rank","Model rank"]])}</details>
    </section>
    <section class="panel"><h2>All negation models · ${esc(c)}</h2>${table(negResults,[["structure","Structure"],["variant","Variant"],["n","Pairs"],["spearman_rho","ρ"],["status","Status"]])}</section>
    <section class="panel"><h2>Mean within-context Spearman</h2><p class="caption">These are averages of separately calculated context correlations, rather than the selected context’s score or one pooled correlation. Each valid context has equal weight. Constant predictions give undefined correlations and are excluded from the mean; valid and total context counts remain visible. These are separate from dataset-level Pearson and log score.</p>${table(S.mean_within_context_spearman,[["measure","Measure"],["structure","Structure"],["variant","Variant"],["mean_within_context_spearman","Mean ρ"],["valid_contexts","Valid contexts"],["total_contexts","Total contexts"]])}</section>`;
}
const pText=value=>value===null?"N/A":value<.001?"< .001":fmt(value);
function associationChart(model,width=400){
  const A=D.rankAssociation,definition=state.wordSource;
  const rows=A.points.filter(r=>r.model===model&&r[definition]!==null&&r.negation!==null);
  if(!rows.length)return `<div class="empty">No matched context coefficients. Sampling scores are never replaced by direct-target scores.</div>`;
  const values=A.points.flatMap(r=>[r.sampling,r.target,r.negation]).filter(v=>v!==null);
  const lo=Math.max(-1.08,Math.min(0,...values)-.06),hi=Math.min(1.08,Math.max(1,...values)+.06);
  const left=57,right=width-16,top=22,bottom=width-52,height=width+11;
  const x=v=>left+(v-lo)/(hi-lo)*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top);
  const ticks=width<350?[0,.5,1]:[0,.25,.5,.75,1];
  const highlight={fridge:"fridge",beach:"beach",cold:"cold"};
  const ordered=[...rows].sort((a,b)=>Number(!!highlight[a.context])-Number(!!highlight[b.context]));
  return `<svg class="chart association-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(D.models[model])}: ${rows.length} contexts; word-ranking Spearman horizontally, exclusion Spearman vertically">
    <title>${esc(D.models[model])}: context-level association</title>
    ${ticks.filter(t=>t>=lo&&t<=hi).map(t=>`<line class="chart-grid" x1="${left}" x2="${right}" y1="${y(t)}" y2="${y(t)}"/><line class="chart-grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${bottom}"/><text x="${x(t)}" y="${bottom+21}" text-anchor="middle">${t}</text><text x="${left-10}" y="${y(t)+4}" text-anchor="end">${t}</text>`).join("")}
    ${ordered.map(r=>`<g class="association-point ${highlight[r.context]||"other"}" tabindex="0" role="button" data-rank-context="${esc(r.context)}" data-rank-model="${model}" aria-label="Inspect ${esc(r.context)}: word Spearman ${fmt(r[definition])}, exclusion Spearman ${fmt(r.negation)}"><title>${esc(r.context)} · word ρ ${fmt(r[definition],6)} · exclusion ρ ${fmt(r.negation,6)}</title><circle cx="${x(r[definition])}" cy="${y(r.negation)}" r="${highlight[r.context]?6:4.5}"/>${highlight[r.context]?`<text x="${x(r[definition])+(r[definition]>.7?-9:9)}" y="${y(r.negation)-10}" text-anchor="${r[definition]>.7?"end":"start"}">${esc(r.context)}</text>`:""}</g>`).join("")}
    <text x="${(left+right)/2}" y="${height-7}" text-anchor="middle">Word-ranking Spearman ρ</text>
    <text transform="translate(15 ${(top+bottom)/2}) rotate(-90)" text-anchor="middle">Exclusion Spearman ρ</text>
  </svg>`;
}
function association(){
  if(!D.rankAssociation)return title("Focus alternatives","Across-context association","Rebuild the viewer to calculate this analysis.");
  const A=D.rankAssociation,records=A.tests.filter(r=>r.definition===state.wordSource);
  return title("Exploratory focus analysis","Does better word ranking accompany better exclusion prediction?","Each point is one context. R correlates the two within-context Spearman coefficients across contexts—not words or individual pairs.",downloadButton("association","Download context points"))+
    `<div class="controls">${wordSourceControl()}${downloadButton("association-tests","Download tests & intervals")}</div>
    <p class="caption">Horizontal: ${wordSourceLabel()}, compared with human norming ranks for six tested alternatives. Vertical: saved model predictions compared with human exclusion rates over 30 ordered pairs. Both panels on the within-context page use this same selected word-score source.</p>
    <div class="association-legend"><span class="fridge">● fridge</span><span class="beach">● beach</span><span class="cold">● cold</span><span>● other contexts</span></div>
    <div class="association-grid">${records.map(r=>`<section class="association-panel"><h2>${esc(short[r.model])}</h2><p class="caption">R = ${fmt(r.R)} · p(Holm) ${r.holm9!==null&&r.holm9<.001?"&lt; .001":"= "+fmt(r.holm9)} · n = ${r.n}</p><div data-association-plot="${r.model}">${associationChart(r.model)}</div>${r.omitted.length?`<p class="caption">Omitted: ${esc(r.omitted.join(", "))} — missing scores or undefined correlation, not zero.</p>`:""}</section>`).join("")}</div>
    <p class="caption">The three labeled contexts were selected for closer inspection. Hover or select a point to inspect it; keyboard users can focus a point and press Enter. All plotted values also appear in the table below. Identical coordinates can overlap.</p>
    <section class="panel"><h2>Association tests and uncertainty</h2><p>${integer(A.permutations)} random pairing permutations; two-sided absolute-statistic test with the +1 correction. Holm adjustment covers nine predictors for each score definition, with a second adjustment across all 18 comparisons. Intervals are unadjusted 95% percentile intervals from ${integer(A.bootstraps)} paired context bootstrap samples, re-ranking each sample. Seed: ${A.seed}.</p>
    <div class="table-wrap"><table><thead><tr><th>Predictor</th><th>Contexts</th><th>R</th><th>95% interval</th><th>Raw p</th><th>Holm (9)</th><th>Holm (18)</th><th>Leave-one-out R range</th></tr></thead><tbody>${records.map(r=>`<tr><th>${esc(D.models[r.model])}</th><td>${r.n}</td><td>${fmt(r.R)}</td><td>[${fmt(r.ciLow)}, ${fmt(r.ciHigh)}]</td><td>${esc(pText(r.p))}</td><td>${esc(pText(r.holm9))}</td><td>${esc(pText(r.holm18))}</td><td>${fmt(r.looMin)}–${fmt(r.looMax)}</td></tr>`).join("")}</tbody></table></div>
    <p class="caption">${esc(A.scope)} These associations do not establish causation. A significant association for one predictor and a nonsignificant one for another are not a test of their difference. X but not Y uses framed scores: its horizontal coordinate still measures the neutral generator, not its framed input.</p></section>
    <section class="panel"><h2>Exact context values</h2><p class="caption">These are the same coefficients used in the scatterplots. Select a context to open its word and pair ranks.</p><div class="table-wrap"><table><thead><tr><th>Context</th><th>Word ρ</th>${records.map(r=>`<th>${esc(short[r.model])} exclusion ρ</th>`).join("")}</tr></thead><tbody>${D.contexts.map(c=>{const points=A.points.filter(p=>p.context===c);return `<tr><th><button data-rank-context="${esc(c)}" data-rank-model="4">${esc(c)}</button></th><td>${fmt(points[0][state.wordSource],6)}</td>${records.map(r=>`<td>${fmt(points.find(p=>p.model===r.model).negation,6)}</td>`).join("")}</tr>`;}).join("")}</tbody></table></div></section>`;
}
let associationObserver;
function resizeAssociationCharts(){
  if(associationObserver)associationObserver.disconnect();
  if(state.view!=="association"||!main.querySelectorAll)return;
  const draw=el=>{const width=Math.max(240,Math.floor(el.getBoundingClientRect().width));if(el.dataset.plotWidth!==String(width)){el.dataset.plotWidth=String(width);el.innerHTML=associationChart(Number(el.dataset.associationPlot),width);}};
  const plots=main.querySelectorAll("[data-association-plot]");
  plots.forEach(draw);
  if(typeof ResizeObserver!=="undefined"){associationObserver=new ResizeObserver(entries=>entries.forEach(entry=>draw(entry.target)));plots.forEach(el=>associationObserver.observe(el));}
}
function filteredPrompts(){return D.prompts.filter(p=>(state.promptDataset==="all"||p.datasets.includes(state.promptDataset))&&(state.promptContext==="all"||p.contexts.includes(state.promptContext))&&p.frame===state.frame);}
function chosenPrompt(){const prompts=filteredPrompts();return prompts.find(p=>p.id===state.promptId)||prompts[0];}
function promptCandidates(p){
  if(!p)return [];
  const targets=new Set(p.candidates.map(c=>c.word.trim().toLowerCase()));
  const source=state.candidateSource==="distribution"?p.distribution:state.candidateSource==="samplingTargets"?p.distribution.filter(c=>targets.has(c.word.trim().toLowerCase())):p.candidates;
  const q=state.wordSearch.toLowerCase();
  return source.filter(c=>c.word.toLowerCase().includes(q)).sort((a,b)=>b.logp-a.logp).slice(0,50);
}
function prompts(){
  const list=filteredPrompts(),p=chosenPrompt();if(p)state.promptId=p.id;
  const hasDistribution=p&&p.distribution.length>0;
  if(p&&!hasDistribution)state.candidateSource="targets";
  const candidates=promptCandidates(p),isDistribution=state.candidateSource!=="targets";
  const candidateHeading=state.candidateSource==="samplingTargets"?"Experimental candidates · sampling-array scores":isDistribution?(state.wordSearch?"Matching exported candidates":"Top 50 from the vocabulary"):"Saved experimental candidates · direct-baseline scores";
  const labelMap=list.map((p,i)=>[p.id,`${i+1}. ${p.contexts.length?p.contexts.join(", ")+" · ":""}${p.text.replace(/\s+/g," ").slice(-110)}`]);
  return title("Next-word scores","What follows each prompt?","Inspect up to 50 candidates at a time, with the exact prompt and natural-log continuation scores.",downloadButton("words","Download displayed candidates"))+
  `<div class="controls">${select("promptDataset","Dataset",[["all","All datasets"],...D.datasets.map(d=>[d.id,d.label])],state.promptDataset)}${state.promptDataset==="novel_focus"?select("promptContext","Focus context",[["all","All 16 contexts"],...D.contexts.map(c=>[c,c])],state.promptContext):""}${select("frame","Prompt frame",[["Neutral","Neutral / no linking"],["X but not Y","X but not Y"]],state.frame)}</div>`+
  (!p?`<div class="panel empty">No prompts match these filters. X-but-not-Y is unavailable for the five R&X conditions.</div>`:
  `<section class="panel"><div class="controls">${select("promptId",`Prompt · ${list.length} available`,labelMap,state.promptId)}<button data-prompt-step="-1" ${list.indexOf(p)===0?"disabled":""}>Previous prompt</button><button data-prompt-step="1" ${list.indexOf(p)===list.length-1?"disabled":""}>Next prompt</button></div><div class="caption">${p.datasets.map(d=>`<span class="tag">${esc(labels[d])}</span>`).join(" ")} ${esc(p.id)}</div><div class="prompt-text">${esc(p.text)}<span class="cursor" aria-hidden="true"></span></div>
  ${!hasDistribution?`<div class="note ${p.frame==="X but not Y"?"":"amber"} distribution-head">${p.frame==="X but not Y"?"The X but not Y condition is implemented as a direct-prediction baseline, so it does not need to score the entire vocabulary. This view shows the saved experimental query candidates, ranked by their continuation log probability. These scores are used directly, without vocabulary normalization or sampling sets and orderings.":"Full vocabulary scores are not included for this prompt. This view shows the saved experimental trigger/query candidates, ranked by log probability. These are not the vocabulary’s top 50. Run the build script on the cluster with the full score arrays to populate the top-50 distribution."}</div>`:`<p class="caption">Distribution coverage: ${esc(p.distributionCoverage)}. Total scored support: ${integer(p.supportSize)} candidates.</p>`}
  <div class="controls">${select("candidateSource","Candidate scores",[...(hasDistribution?[["distribution","Top 50 vocabulary candidates"],["samplingTargets","Experimental candidates · sampling-array scores"]]:[]),["targets","Saved experimental candidates · direct-baseline scores"]],state.candidateSource)}${select("wordScale","Plot values",[["logp","Natural log probability"],["raw","Raw continuation probability"],...(isDistribution?[["normalized","Normalized sampling probability"]]:[])],state.wordScale)}<label for="wordSearch">Find a candidate<input id="wordSearch" type="search" value="${esc(state.wordSearch)}" placeholder="Search exported candidates"></label></div>
  <div class="note distribution-head">A score is the sum of next-token log probabilities across the candidate continuation; candidates can be single words or phrases. exp(log p) gives the raw continuation probability. ${isDistribution?"Normalized sampling probability divides each candidate’s raw weight by the total across the full scored vocabulary. It is not normalized over the displayed subset.":"The saved candidate subset is not renormalized into a distribution."} ${p.frame==="Neutral"?(D.scoreAlignment?.matched?"Direct-target and full-vocabulary scores match. Word ranks and linking structures use these same candidate scores.":"The direct-baseline and sampling-array scores are separate historical artifacts. Word-rank analyses now default to the sampling-array scores; the saved No-linking predictions still use the direct-baseline scores."):"Framed vocabulary scores, when supplied, are exported separately from the neutral-prompt scores."}</div>
  <div class="grid-two"><div><h2>${candidateHeading}</h2><p class="caption">${candidates.length} shown · ${state.wordScale==="logp"?"Natural log scale; shorter bars indicate larger probabilities.":state.wordScale==="raw"?"Raw continuation probabilities; bars share a zero origin.":"Probability normalized across the full candidate support."}${state.wordSearch?" Search is limited to the candidates included in this HTML export.":""}</p>${wordBars(candidates)}</div><div><h2>Exact values</h2><div class="table-wrap"><table class="word-table"><thead><tr><th>Candidate</th>${isDistribution?"<th>Rank</th>":"<th>Role</th>"}<th>log p</th><th>exp(log p)</th>${isDistribution?"<th>Sampling p</th>":"<th>Tokens</th>"}</tr></thead><tbody>${candidates.map(c=>`<tr><th>${esc(c.word)}</th><td>${isDistribution?c.rank:esc(c.roles.join(", "))}</td><td class="score-cell">${fmt(c.logp,5)}</td><td>${prob(Math.exp(c.logp))}</td><td>${isDistribution?prob(c.normalized):c.tokens??"N/A"}</td></tr>`).join("")}</tbody></table></div>${!candidates.length?`<div class="empty">No matching candidates in this export.</div>`:""}<p class="caption">Ranking applies to ${isDistribution?"the scored candidate support, which contains unigrams and bounded bigrams":"the saved experimental candidates only"}. Candidate continuations can overlap (e.g. a word and a phrase beginning with that word), so raw probabilities need not sum to 1.</p></div></div></section>`);
}
function wordBars(candidates){
  const key=state.wordScale,values=candidates.map(c=>key==="logp"?c.logp:key==="raw"?Math.exp(c.logp):c.normalized);
  const max=Math.max(1e-20,...values.map(v=>Math.abs(v??0)));
  return `<div class="word-chart">${candidates.map((c,i)=>`<div class="rank-row"><span class="rank-label">${esc(c.word)}</span><div class="bar-track"><span class="bar-fill" style="${key==="logp"?"right:0":"left:0"};width:${Math.abs(values[i]??0)/max*100}%"></span></div><span class="rank-value">${key==="logp"?fmt(values[i],3):prob(values[i])}</span></div>`).join("")}</div>`;
}
function methods(){
  const counts=D.folds.reduce((out,r)=>{const key=r.variant+"|"+r.selected_boundary;out[key]=(out[key]||0)+1;return out;},{});
  return title("Reading the analysis","Methods, coverage & provenance","Every view is generated from saved results. No model is fitted in the browser.",downloadButton("provenance","Download provenance"))+
  `<section class="panel"><h2>Score sources and the new rank analysis</h2><p>Word-ranking Spearman defaults to the full-support scores used by the sampled linking structures, restricted to the six tested alternatives for evaluation. The within-context and across-context pages use the same word ranks and held-out pair predictions. The separate direct-target comparison is retained and labeled explicitly.</p><p>${D.scoreAlignment?`${integer(D.scoreAlignment.compared)} neutral prompt–candidate scores compared; ${integer(D.scoreAlignment.mismatched)} differ by more than 10⁻⁸; ${integer(D.scoreAlignment.missing)} lack a sampling export. ${D.scoreAlignment.matched?"The direct and sampling candidate scores agree.":"The historical No-linking baseline has not been silently recalculated from the sampling scores."}`:""}</p><p>The association tests are recomputed from this export’s context coefficients during the build. No static p values or figure from a different run are attached. Each association uses 15 or 16 contexts, not 480 independent pairs. Missing or constant-vector correlations remain undefined. See <button data-view="association">Across-context association</button> for permutation tests, bootstrap intervals, and their limitations.</p></section>`+
  `<section class="panel"><h2>Two fit measures</h2><div class="grid-two"><div><h3>Pearson correlation</h3><p>Linear association between the model’s exclusion probability and the observed human exclusion rate across analysis units. Correlation is undefined when either variable is constant or fewer than two units are available.</p><p>The neutral and X-but-not-Y correlations use exponentiated continuation scores, matching the active linking tables. They do not correlate raw log probabilities with human rates.</p></div><div><h3>Mean proper log score</h3><p class="formula">score(y, p) = y × ln(p) + (1 − y) × ln(1 − p)</p><p>The saved analysis clips p to [10⁻¹⁰, 1 − 10⁻¹⁰]. Larger scores are better. Dataset summaries average items; the existing aggregate takes an equal-weight mean of the ten dataset means. Published rates are not turned into invented participant counts.</p></div></div><div class="note">The optional pooled view weights analysis units equally. The mean dataset correlation is a new, explicitly descriptive viewer summary; the saved pipeline supplies dataset correlations, not an aggregate correlation.</div></section>
  <section class="panel"><h2>Nine linking structures</h2><div class="method-grid">${D.models.map((m,i)=>`<article><h3>${esc(m)}</h3><p class="caption">${descriptions[i]}</p></article>`).join("")}</div></section>
  <section class="panel"><h2>Cross-validation & boundaries</h2><p>Grouped ten-fold cross-validation chooses K and p separately by maximizing the equal-dataset mean Set log score on training folds. The selected boundary is then reused for Conjunction and Disjunction in the held-out fold. Groups preserve focus contexts, Hu scales, and matched R&X items. These context views summarize the same held-out predictions without refitting.</p><p>The current run samples 500 orderings with seed 7. Top-p is the shortest <em>sampled</em> prefix reaching the mass boundary, not a sorted nucleus distribution.</p><p>${Object.entries(counts).map(([key,count])=>{const [v,b]=key.split("|");return `${v==="top_k"?"K":"p"} = ${Number(b)} in ${count} folds`;}).join(" · ")}</p><p class="caption">Tested K: ${D.boundaries.top_k.join(", ")}<br>Tested p: ${D.boundaries.top_p.join(", ")}</p>${D.folds.some(r=>r.variant==="top_k"&&Number(r.selected_boundary)===Math.max(...D.boundaries.top_k))?`<div class="note amber">Top-K reaches the largest tested boundary. This is a search-grid limit, not evidence of an interior optimum; the repository calls for expanding the K grid before freezing the analysis.</div>`:""}<details><summary>Show fold selections</summary><div class="table-wrap"><table class="fold-table"><thead><tr><th>Variant</th><th>Held-out fold</th><th>Boundary</th><th>Training score</th><th>Training N</th><th>Held-out N</th></tr></thead><tbody>${D.folds.map(r=>`<tr><th>${esc(r.variant)}</th><td>${r.heldout_fold}</td><td>${Number(r.selected_boundary)}</td><td>${fmt(Number(r.balanced_training_set_log_score))}</td><td>${r.training_unit_count}</td><td>${r.heldout_unit_count}</td></tr>`).join("")}</tbody></table></div></details></section>
  <section class="panel"><h2>Coverage & source grain</h2><p>${integer(D.sourceRows)} canonical rows become ${integer(D.items.length)} analysis units. Hu rows follow the published inclusion filter and aggregate templates within scale; the four Hu datasets contribute 39, 67, 50, and 57 scales. Five R&X conditions contribute 60 items each. Novel focus contributes 480 ordered pairs in 16 contexts.</p><p>X-but-not-Y is unavailable for the five R&X conditions. The overview labels its five-dataset aggregate and offers shared coverage for a like-for-like comparison. The base-rate reference uses each dataset’s training-fold human mean.</p><p>Prompt browsing includes all saved canonical source rows, including Hu rows outside the analysis subset. Fit statistics use retained analysis units only. Neutral prompts shared across conditions appear once and carry every associated dataset label.</p><p>Model: ${esc(D.modelName)} · revision <span class="formula">${esc(D.revision)}</span>.</p></section>
  <section class="panel"><h2>Rebuild & full distributions</h2><p>Rebuild locally with the repository’s Python script. On the cluster, pass the score-array directory using <code>--log-probs-dir</code>; the exporter includes the top 50 plus experimental candidates from each neutral prompt. Relocated vocabulary files can be supplied with <code>--vocab-dir</code>. The generated HTML remains self-contained.</p><p>Without arrays, only saved experimental candidates are shown, plus the checked-in mask diagnostic where available. No missing probabilities are estimated. Search is limited to exported candidates; <code>--top-candidates 0</code> exports the full support at the cost of a much larger HTML file.</p></section>
  <section class="panel"><h2>Input provenance</h2><p class="caption">Generated ${esc(D.generated)} · ${D.verifiedCells} saved metric cells verified. File hashes describe the exact inputs used for this export.</p><div class="table-wrap"><table><thead><tr><th>Source file</th><th>SHA-256</th><th>Bytes</th></tr></thead><tbody>${D.provenance.map(p=>`<tr><td class="path">${esc(p.path)}</td><td class="path">${esc(p.sha256)}</td><td>${integer(p.bytes)}</td></tr>`).join("")}</tbody></table></div></section>`;
}
function render(){
  if(state.metric==="rho"&&!supportsDatasetSpearman())state.metric="log";
  const active=document.activeElement,id=active&&active.id,pos=active&&active.selectionStart;
  document.querySelectorAll("#navigation button").forEach(b=>{if(b.dataset.view===state.view)b.setAttribute("aria-current","page");else b.removeAttribute("aria-current");});
  const runNote=D.run?`<div class="note ${D.run.corrected?"":"amber"} run-status"><strong>${esc(D.run.label)}</strong> · ${esc(D.run.id)}.${D.run.corrected?" Direct neutral scores and sampled structures are verified against the same candidate-score artifact.":" This is not the corrected single-space rerun. Word-rank analyses default to the actual scores behind these saved linking predictions. The historical No-linking baseline retains its separate direct-target scores; X but not Y retains its framed scores."}</div>`:"";
  main.innerHTML=runNote+({overview,datasets,distributions,structures,contexts,spearman,association,prompts,methods})[state.view]();
  resizeAssociationCharts();
  const el=id&&document.getElementById(id);if(el){el.focus();if(typeof pos==="number"&&el.setSelectionRange)el.setSelectionRange(pos,pos);}
  document.getElementById("footer").innerHTML=`<span>${esc(D.run?.id||"Local snapshot")} · ${esc(D.modelName)} · ${D.verifiedCells} metric cells verified</span><span>Built ${esc(D.generated.slice(0,10))} · No network connection required</span>`;
}
function csvDownload(kind){
  let rows=[];
  if(kind==="overview")rows=D.models.map((m,i)=>{const s=aggregate(scopeItems(),i,state.balanced);return {model:m,aggregation:state.balanced?"equal_dataset_means":"pooled_units",coverage:state.coverage,n:s.n,datasets:s.datasets,correlation:s.r,mean_log_score:s.log};});
  if(kind==="dataset")rows=D.models.map((m,i)=>({dataset:labels[state.dataset],model:m,...D.datasetSummaries[state.dataset][i]}));
  if(kind==="structure")rows=D.datasets.map(d=>({dataset:d.label,model:D.models[state.model],...D.datasetSummaries[d.id][state.model]}));
  if(kind==="contexts")rows=D.contexts.flatMap(c=>D.models.map((m,i)=>({context:c,model:m,...D.contextSummaries[c][i]})));
  if(kind==="association")rows=D.rankAssociation.points.map(r=>({run:D.run.id,word_score_source:state.wordSource,context:r.context,predictor:D.models[r.model],word_spearman:r[state.wordSource],exclusion_spearman:r.negation,word_status:state.wordSource==="sampling"?r.wordStatus:"direct_target",exclusion_status:r.negationStatus}));
  if(kind==="association-tests")rows=D.rankAssociation.tests.filter(r=>r.definition===state.wordSource).map(r=>({...r,run:D.run.id,predictor:D.models[r.model],omitted:r.omitted.join("; "),seed:D.rankAssociation.seed,permutations:D.rankAssociation.permutations,bootstraps:D.rankAssociation.bootstraps}));
  if(kind==="items")rows=itemRowsForView().filter(r=>(r.id+" "+r.context+" "+r.trigger+" "+r.query).toLowerCase().includes(state.itemSearch.toLowerCase())).flatMap(r=>D.models.map((m,i)=>({dataset:labels[r.dataset],item:r.id,context:r.context,trigger:r.trigger,query:r.query,fold:r.fold,model:m,human_rate:r.y,probability:r.p[i],log_score:r.scores[i]})));
  if(kind==="words"){const p=chosenPrompt();rows=promptCandidates(p).map(c=>({prompt_id:p.id,prompt:p.text,source:state.candidateSource,candidate:c.word,log_probability:c.logp,raw_probability:Math.exp(c.logp),normalized_sampling_probability:c.normalized??null,rank:c.rank??null,token_count:c.tokens??null}));}
  if(kind==="distributions")rows=D.models.flatMap((model,m)=>{
    if(state.distributionModel!=="all"&&Number(state.distributionModel)!==m)return [];
    const d=ResultsMath.distribution(distributionRows(),m);
    return d.n?d.human.counts.map((n,i)=>({dataset:labels[state.dataset],context:state.dataset==="novel_focus"?state.distributionContext:"all",model,n:d.n,bin_lower:i/20,bin_upper:(i+1)/20,upper_inclusive:i===19,human_count:n,model_count:d.model.counts[i],human_share:n/d.n,model_share:d.model.counts[i]/d.n})):[];
  });
  if(kind==="provenance")rows=D.provenance;
  if(!rows.length)return;
  const columns=[...new Set(rows.flatMap(r=>Object.keys(r)))],quote=v=>'"'+String(v??"").replace(/"/g,'""')+'"';
  const csv=[columns,...rows.map(r=>columns.map(c=>r[c]))].map(r=>r.map(quote).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"}));
  const link=document.createElement("a");link.href=url;link.download=`focus-alternatives-${kind}.csv`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.addEventListener("change",e=>{
  const id=e.target.id,v=e.target.value;
  if(id==="aggregation")state.balanced=v==="balanced";
  else if(id==="model")state.model=Number(v);
  else if(id in state)state[id]=v;
  else return;
  if(["dataset","context"].includes(id)){state.page=0;state.item=null;state.itemSearch="";}
  if(["promptDataset","promptContext","frame","promptId"].includes(id)){state.wordSearch="";state.candidateSource="distribution";state.wordScale="logp";}
  if(id==="promptDataset")state.promptContext="all";
  if(id==="candidateSource")state.wordScale="logp";
  render();
});
document.addEventListener("input",e=>{if(["itemSearch","wordSearch"].includes(e.target.id)){state[e.target.id]=e.target.value;state.page=0;render();}});
document.addEventListener("click",e=>{
  const target=e.target.closest("[data-view],[data-model],[data-dataset],[data-context],[data-cell],[data-item],[data-page],[data-prompt],[data-close-item],[data-context-prompts],[data-prompt-step],[data-download],[data-spearman-example],[data-rank-context]");
  if(!target||target.disabled)return;
  const d=target.dataset;
  if(d.download){csvDownload(d.download);return;}
  let navigate=false;
  if(d.rankContext){const test=D.rankAssociation.tests.find(r=>r.model===Number(d.rankModel));state.view="spearman";state.context=d.rankContext;state.spearmanModel=`${test.structure}|${test.variant}`;navigate=true;}
  if(d.spearmanExample){state.view="spearman";state.context=d.spearmanExample;state.spearmanModel="set|top_k";navigate=true;}
  if(d.view){state.view=d.view;state.page=0;state.item=null;navigate=true;}
  if(d.cell){const [kind,id,m]=d.cell.split("|");state.model=Number(m);if(kind==="dataset"){state.dataset=id;state.view="datasets";}else{state.context=id;state.view="contexts";}state.item=null;state.page=0;state.itemSearch="";navigate=true;}
  if(d.dataset){state.dataset=d.dataset;state.view="datasets";state.item=null;state.page=0;state.itemSearch="";navigate=true;}
  if(d.context){state.context=d.context;state.view="contexts";state.item=null;state.page=0;state.itemSearch="";navigate=true;}
  if(d.model!==undefined){state.model=Number(d.model);if(state.view==="overview"){state.view="structures";navigate=true;}}
  if(d.item)state.item=d.item;
  if(d.closeItem)state.item=null;
  if(d.page)state.page+=Number(d.page);
  if(d.contextPrompts){state.promptDataset="novel_focus";state.promptContext=d.contextPrompts;state.frame="Neutral";state.promptId="";state.view="prompts";state.candidateSource="distribution";state.wordScale="logp";state.wordSearch="";navigate=true;}
  if(d.prompt){const p=D.prompts.find(p=>p.id===d.prompt);state.promptId=p.id;state.promptDataset=p.datasets.includes("novel_focus")?"novel_focus":p.datasets[0];state.promptContext=p.contexts[0]||"all";state.frame=p.frame;state.view="prompts";state.candidateSource="distribution";state.wordScale="logp";state.wordSearch="";navigate=true;}
  if(d.promptStep){const prompts=filteredPrompts(),index=prompts.findIndex(p=>p.id===state.promptId);state.promptId=prompts[index+Number(d.promptStep)].id;state.candidateSource="distribution";state.wordScale="logp";state.wordSearch="";}
  render();
  if(navigate){main.focus({preventScroll:true});window.scrollTo({top:0,behavior:"smooth"});}
  if(d.item)document.getElementById("item-detail")?.scrollIntoView({behavior:"smooth",block:"start"});
});
document.addEventListener("keydown",e=>{
  if(e.key!=="Enter"&&e.key!==" ")return;
  const target=e.target.closest("g[data-rank-context]");
  if(target){e.preventDefault();target.dispatchEvent(new MouseEvent("click",{bubbles:true}));}
});
render();
})();
