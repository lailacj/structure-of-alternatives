"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const math=require("./viewer.js");
const html=fs.readFileSync(path.join(__dirname,"index.html"),"utf8");
const data=JSON.parse(html.match(/<script id="results-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);

test("aggregate weighting and unavailable predictions",()=>{
  const rows=[{dataset:"a",y:0,p:[.1,null],scores:[-1,null]},{dataset:"b",y:0,p:[.2,.3],scores:[-3,-4]},{dataset:"b",y:1,p:[.8,.6],scores:[-3,-5]}];
  assert.equal(math.aggregate(rows,0,true).log,-2);
  assert.equal(math.aggregate(rows,0,false).log,-7/3);
  assert.equal(math.aggregate(rows,1,true).datasets,1);
  assert.equal(math.aggregate(rows,1,true).n,2);
  assert.equal(math.aggregate(rows,1,true).log,-4.5);
  assert.equal(math.pearson([1,1],[0,1]),null);
  assert.ok(Math.abs(math.pearson([0,1,2],[2,1,0])+1)<1e-12);
});

test("JavaScript metrics agree with every Python-generated dataset and context summary",()=>{
  for(const [id,stats] of Object.entries(data.datasetSummaries)){
    for(let model=0;model<9;model++){
      const actual=math.summary(data.items.filter(r=>r.dataset===id),model);
      assert.equal(actual.n,stats[model].n);
      for(const metric of ["r","log","rho"]){
        if(stats[model][metric]===null)assert.equal(actual[metric],null);
        else assert.ok(Math.abs(actual[metric]-stats[model][metric])<1e-10,`${id} / ${model} / ${metric}`);
      }
    }
  }
  for(const [context,stats] of Object.entries(data.contextSummaries)){
    for(let model=0;model<9;model++){
      const actual=math.summary(data.items.filter(r=>r.context===context),model);
      for(const metric of ["r","log","rho"]){
        if(stats[model][metric]===null)assert.equal(actual[metric],null);
        else assert.ok(Math.abs(actual[metric]-stats[model][metric])<1e-10);
      }
    }
  }
});

function harness(payload=data){
  const listeners={},elements={},downloads=[];
  const element=id=>elements[id]||(elements[id]={id,innerHTML:"",textContent:"",focus(){},setAttribute(){},removeAttribute(){},scrollIntoView(){}});
  element("results-data").textContent=JSON.stringify(payload);
  const document={activeElement:null,getElementById:element,querySelector:()=>null,querySelectorAll:()=>[],addEventListener:(name,fn)=>{listeners[name]=fn;},createElement:()=>({click(){}})};
  const sandbox={document,window:{scrollTo(){},addEventListener(){}},console,Blob,
    URL:{createObjectURL(blob){downloads.push(blob);return "blob:test";},revokeObjectURL(){}},setTimeout:fn=>fn()};
  vm.createContext(sandbox);
  // Exercise the scripts actually shipped in the HTML, not only the JS source.
  for(const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script[1],sandbox);
  return {html:()=>element("main").innerHTML,downloads:()=>Promise.all(downloads.map(blob=>blob.text())),
    click:(dataset,table)=>listeners.click({target:{closest:()=>({dataset,closest:()=>table})}}),
    change:(id,value)=>listeners.change({target:{id,value}}),
    input:(id,value)=>listeners.input({target:{id,value}})};
}

test("pair contributions match saved context coefficients and tied ranks for all models",()=>{
  for(const context of data.contexts)for(let model=0;model<data.models.length;model++){
    const rows=data.items.filter(r=>r.context===context),actual=math.pairCorrelations(rows,model);
    assert.equal(actual.n,30);
    const expected=data.contextSummaries[context][model].r;
    if(expected===null)assert.equal(actual.r,null);else assert.ok(Math.abs(actual.r-expected)<1e-10);
    const rho=math.spearman(rows.map(r=>r.p[model]),rows.map(r=>r.y));
    if(rho===null)assert.equal(actual.rho,null);else assert.ok(Math.abs(actual.rho-rho)<1e-10);
    for(const [field,coefficient] of [["pearsonContribution","r"],["spearmanContribution","rho"]]){
      if(actual[coefficient]===null)assert.ok(actual.rows.every(r=>r[field]===null));
      else assert.ok(Math.abs(actual.rows.reduce((s,r)=>s+r[field],0)-actual[coefficient])<1e-12);
    }
  }
  const cold=math.pairCorrelations(data.items.filter(r=>r.context==="cold"),2);
  const blanket=cold.rows.find(r=>r.trigger==="heater"&&r.query==="blankets");
  assert.equal(blanket.modelRank,25.5);assert.equal(blanket.humanRank,4);
  assert.ok(Math.abs(blanket.pearsonContribution+.023058796586561767)<1e-12);
  const bag=math.pairCorrelations(data.items.filter(r=>r.context==="bag"),0);
  const chalk=bag.rows.find(r=>r.trigger==="cash"&&r.query==="chalk");
  assert.equal(chalk.modelRank,13);assert.equal(chalk.humanRank,6);
  assert.ok(chalk.pearsonContribution<0&&chalk.spearmanContribution>0);
  assert.equal(math.pairCorrelations([{id:"a",y:0,p:[null]}],0).r,null);
  assert.equal(math.pairCorrelations([{id:"a",y:0,p:[.2]},{id:"b",y:1,p:[.2]}],0).rho,null);
});

test("paired plots select the same item across contexts, models, and both interaction routes",()=>{
  const app=harness();app.click({view:"contexts"});
  for(const context of data.contexts){
    app.change("context",context);
    for(let model=0;model<data.models.length;model++){
      app.change("model",String(model));
      assert.equal((app.html().match(/class="chart pair-chart"/g)||[]).length,2);
      assert.match(app.html(),/All 30 pairs and their contributions/);
      assert.ok(!/\bNaN\b|\bInfinity\b/.test(app.html()));
    }
    for(const row of data.items.filter(r=>r.context===context)){
      app.change("comparisonPair",row.id);
      assert.equal(app.html().split(`data-selected-pair="${row.id}"`).length-1,2);
    }
  }
  app.change("context","bag");app.change("model","0");
  app.click({comparisonPair:"bag::cash::chalk"});
  assert.equal(app.html().split('data-selected-pair="bag::cash::chalk"').length-1,2);
  assert.match(app.html(),/-0\.016575/);assert.match(app.html(),/\+0\.011061/);
  app.click({item:"novel_focus|bag::cash::pen"});
  assert.equal(app.html().split('data-selected-pair="bag::cash::pen"').length-1,2);
  app.change("model","2");
  assert.equal(app.html().split('data-selected-pair="bag::cash::pen"').length-1,2);
  app.change("context","mask");app.change("model","3");
  assert.match(app.html(),/Undefined correlation \(N\/A\)/);
  assert.match(app.html(),/Pearson r = N\/A/);assert.match(app.html(),/Spearman ρ = N\/A/);
});

test("every dataset, structure, context, and fit measure renders with actual data",()=>{
  const app=harness();
  assert.match(app.html(),/How do the linking structures fit/);
  for(const metric of ["r","log"]){
    app.change("metric",metric);
    for(const d of data.datasets){app.click({dataset:d.id});assert.match(app.html(),/Individual items/);assert.ok(!/\bNaN\b|>undefined</.test(app.html()));}
    app.click({view:"structures"});
    for(let i=0;i<9;i++){app.change("model",String(i));assert.match(app.html(),/Dataset-level results/);}
    for(const c of data.contexts){app.click({context:c});assert.match(app.html(),/Contexts × linking structures/);assert.match(app.html(),/Individual items/);}
  }
  app.click({view:"methods"});assert.match(app.html(),/SHA-256/);
  app.click({view:"overview"});app.change("coverage","shared");app.change("aggregation","pooled");assert.match(app.html(),/Pooled item log score/);
});

test("item drill-down, prompt navigation, source coverage, and search",()=>{
  const app=harness();
  app.click({dataset:"novel_focus"});app.click({item:"novel_focus|bag::cash::chalk"});assert.match(app.html(),/Source rows &amp; prompts|Source rows & prompts/);
  const bag=data.prompts.find(p=>p.frame==="Neutral"&&p.contexts.includes("bag"));
  app.click({prompt:bag.id});
  assert.match(app.html(),bag.distribution.length?/Top 50 from the vocabulary/:/not the vocabulary’s top 50/);
  app.input("wordSearch","no-such-candidate-123");assert.match(app.html(),/No matching candidates/);
  const mask=data.prompts.find(p=>p.frame==="Neutral"&&p.contexts.includes("mask"));
  app.click({prompt:mask.id});assert.match(app.html(),/Top 50 from the vocabulary/);
  app.change("wordScale","normalized");assert.match(app.html(),/0\.7922/);
  app.change("candidateSource","targets");assert.match(app.html(),/Saved experimental candidates/);
  app.change("promptDataset","rnx_esi");app.change("frame","X but not Y");assert.match(app.html(),/No prompts match/);
  app.change("frame","Neutral");assert.match(app.html(),/ESI/);assert.match(app.html(),/Eonly/);
  app.click({view:"datasets"});app.input("itemSearch","<script>bad</script>");assert.match(app.html(),/&lt;script&gt;/);assert.ok(!app.html().includes("<script>bad"));
});

test("candidate-only export still explains missing full vocabulary coverage",()=>{
  const subset=JSON.parse(JSON.stringify(data));
  const bag=subset.prompts.find(p=>p.frame==="Neutral"&&p.contexts.includes("bag"));
  bag.distribution=[];bag.distributionCoverage="unavailable";
  const app=harness(subset);app.click({prompt:bag.id});
  assert.match(app.html(),/not the vocabulary’s top 50/);
  const framed=subset.prompts.find(p=>p.frame==="X but not Y");
  app.click({prompt:framed.id});
  assert.match(app.html(),/direct-prediction baseline/);
  assert.ok(!app.html().includes("Run the build script on the cluster"));
});

test("both HTML entry points lead to results and retain a script-free fallback",()=>{
  const entry=fs.readFileSync(path.join(__dirname,"viewer.html"),"utf8");
  assert.match(entry,/url=index\.html/);
  assert.match(entry,/href="index\.html"/);
  assert.ok(!entry.includes("__VIEWER_DATA__"));
  const fallback=html.match(/<main[^>]*>([\s\S]*?)<\/main>/)[1];
  assert.match(fallback,/viewer-status/);
  assert.match(fallback,/Pearson correlation/);
  assert.match(fallback,/Mean proper log score/);
  assert.match(fallback,/van Tiel et al/);
  assert.match(fallback,/-3\.217/);
  assert.ok(!html.includes("__VIEWER_FALLBACK__"));
});

test("histogram boundaries, matched coverage, and exact endpoints",()=>{
  const d=math.distribution([0,.049,.05,.5,.95,1].map(y=>({y,p:[y]})),0);
  assert.equal(d.n,6);assert.equal(d.human.counts[0],2);assert.equal(d.model.counts[1],1);
  assert.equal(d.model.counts[19],2);assert.equal(d.model.zeros,1);assert.equal(d.model.ones,1);
  assert.equal(d.model.median,.275);
  const missing=math.distribution([{y:1,p:[null]},{y:0,p:[.5]}],0);
  assert.equal(missing.n,1);assert.equal(missing.human.mean,0);
  for(const dataset of data.datasets)for(let m=0;m<9;m++){
    const actual=math.distribution(data.items.filter(r=>r.dataset===dataset.id),m);
    assert.equal(actual.n,data.datasetSummaries[dataset.id][m].n);
    for(const source of [actual.human,actual.model])assert.equal(source.counts.reduce((a,b)=>a+b,0),actual.n);
  }
});

test("exact Wasserstein handles unequal sample sizes, ties, and constant distributions",()=>{
  assert.equal(math.wasserstein([0,1],[1,0]),0);
  assert.equal(math.wasserstein([0,0],[1,1,1]),1);
  assert.equal(math.wasserstein([0,1],[.5]),.5);
  assert.ok(Math.abs(math.wasserstein([0,.5,1],[0,0,1])-1/6)<1e-14);
  // Identical histogram bins must still yield a nonzero exact distance.
  assert.ok(Math.abs(math.wasserstein([.001],[.049])-.048)<1e-14);
  assert.equal(math.wasserstein([],[1]),null);
  assert.throws(()=>math.wasserstein([NaN],[1]),/finite/);
  assert.throws(()=>math.wasserstein([1],[Infinity]),/finite/);
});

test("all 153 focus distances agree with independent quantile transport",()=>{
  const focus=data.items.filter(r=>r.dataset==="novel_focus");
  const distances=math.focusWasserstein(data.items,data.models.length);
  assert.equal(distances.length,153);
  // Independent oracle: split each empirical quantile into a common number
  // of equal-mass pieces (480 is divisible by both 30 and 480).
  const quantiles=values=>[...values].sort((a,b)=>a-b).flatMap(v=>Array(480/values.length).fill(v));
  const expected=(a,b)=>{
    const x=quantiles(a),y=quantiles(b);
    return x.reduce((s,v,i)=>s+Math.abs(v-y[i]),0)/480;
  };
  for(const row of distances){
    const scope=focus.filter(r=>row.context==="all"||r.context===row.context);
    const x=scope.map(r=>r.p[row.model]),y=scope.map(r=>r.y);
    assert.equal(row.n,row.context==="all"?480:30);
    assert.equal(row.human_n,row.n);
    assert.equal(row.scope,row.context==="all"?"pooled_contexts":"within_context");
    assert.ok(Math.abs(row.wasserstein_distance-expected(x,y))<1e-12);
    assert.ok(row.wasserstein_distance>=0&&row.wasserstein_distance<=1);
    assert.ok(!("w1_pooled" in row));
  }
  const ordering=distances.filter(r=>r.model===4);
  const contextMean=math.mean(ordering.filter(r=>r.context!=="all").map(r=>r.wasserstein_distance));
  assert.ok(Math.abs(ordering.find(r=>r.context==="all").wasserstein_distance-contextMean)>.01);
  const missing=math.focusWasserstein([{dataset:"novel_focus",context:"test",y:.5,p:[null]}],1);
  assert.equal(missing[1].wasserstein_distance,null);
});

test("Wasserstein tables and CSV report within-context and pooled-to-pooled comparisons only",async()=>{
  const app=harness();app.click({view:"distributions"});
  assert.match(app.html(),/Wasserstein distance · focus study/);
  assert.match(app.html(),/Compare all 16 within-context distances/);
  assert.match(app.html(),/Wasserstein distance \(W₁\)/);
  assert.ok(!app.html().includes("W₁ · pooled humans"));
  app.change("distributionContext","cold");app.change("distributionModel","2");
  const row=math.focusWasserstein(data.items,9).find(r=>r.context==="cold"&&r.model===2);
  assert.ok(app.html().includes(row.wasserstein_distance.toFixed(6)));
  assert.ok(!app.html().includes("Compare all 16 within-context distances"));
  app.click({download:"wasserstein"});
  const [csv]=await app.downloads();
  assert.equal(csv.split("\r\n").length,154);
  assert.match(csv,/wasserstein_distance/);assert.match(csv,/"480"/);
  assert.match(csv,/within_context/);assert.match(csv,/pooled_contexts/);
  assert.ok(!/w1_pooled|w1_context|pooled_human_n/.test(csv));
  assert.match(csv,/"bag"/);assert.match(csv,/"all"/);
  app.change("dataset","hu_vt16");
  assert.ok(!app.html().includes("Wasserstein distance · focus study"));
});

test("distribution gallery renders every available model and dataset",()=>{
  const app=harness();app.click({view:"distributions"});
  for(const dataset of data.datasets){
    app.change("dataset",dataset.id);
    assert.equal((app.html().match(/class="chart distribution-chart"/g)||[]).length,dataset.id.startsWith("rnx_")?8:9);
    assert.ok(!/\bNaN\b|>undefined</.test(app.html()));
  }
  app.change("distributionModel","1");assert.match(app.html(),/X but not Y/);
  app.change("dataset","rnx_esi");assert.match(app.html(),/Not applicable/);
  app.change("dataset","novel_focus");app.change("distributionModel","4");
  assert.equal((app.html().match(/class="chart distribution-chart"/g)||[]).length,1);
  assert.match(app.html(),/Model mean 50.0%/);assert.match(app.html(),/480 matched units/);
});

test("focus distributions cover all 16 contexts by all nine models",()=>{
  const app=harness();app.click({view:"distributions"});
  assert.match(app.html(),/All contexts pooled/);
  for(const context of data.contexts){
    app.change("distributionContext",context);
    assert.equal((app.html().match(/30 matched units/g)||[]).length,9);
    const items=data.items.filter(r=>r.dataset==="novel_focus"&&r.context===context);
    for(let m=0;m<9;m++){
      const dist=math.distribution(items,m);
      assert.equal(dist.n,30);
      assert.equal(dist.model.counts.reduce((a,b)=>a+b,0),30);
      app.change("distributionModel",String(m));
      assert.match(app.html(),new RegExp(` · ${context}</h2>`));
      assert.ok(app.html().includes(`Model mean ${(dist.model.mean*100).toFixed(1)}%`));
    }
    app.change("distributionModel","all");
  }
  app.change("dataset","hu_vt16");
  assert.ok(!app.html().includes('id="distributionContext"'));
  assert.equal((app.html().match(/39 matched units/g)||[]).length,9);
  app.change("dataset","novel_focus");app.change("distributionContext","all");
  assert.equal((app.html().match(/480 matched units/g)||[]).length,9);
});

test("Spearman page exposes both measures and context ranks",()=>{
  const app=harness();app.click({view:"spearman"});
  assert.match(app.html(),/Mean within-context Spearman/);
  assert.match(app.html(),/Valid contexts/);
  app.change("context","fridge");
  assert.match(app.html(),/Word-ranking Spearman · fridge/);
  assert.match(app.html(),/Negation Spearman · fridge/);
  assert.match(app.html(),/water/);
  assert.match(app.html(),/Human rank/);
});

test("rank plots preserve every observation across context and model changes",()=>{
  const app=harness();app.click({view:"spearman"});app.click({spearmanExample:"fridge"});
  assert.match(app.html(),/ρ = 1.000/);assert.match(app.html(),/ρ = 0.860/);
  for(const context of data.contexts){
    app.change("context",context);
    for(const model of data.spearman.negation_spearman_by_context.filter(r=>r.context===context)){
      app.change("spearmanModel",`${model.structure}|${model.variant}`);
      const plots=[...app.html().matchAll(/<svg class="chart spearman-chart"[\s\S]*?<\/svg>/g)].map(m=>m[0]);
      assert.equal(plots.length,2);
      const counts=plots.map(plot=>[...plot.matchAll(/data-rank-count="(\d+)"/g)].reduce((s,m)=>s+Number(m[1]),0));
      assert.deepEqual(counts,[6,30]);
      assert.ok(!plots.some(plot=>/NaN|undefined/.test(plot)));
      if(model.status!=="defined")assert.match(app.html(),/Undefined correlation:/);
    }
  }
});

test("across-context plots use the actual sampled inputs and link to the same rank details",()=>{
  const app=harness();app.click({view:"association"});
  assert.match(app.html(),/Corrected single-space run/);
  assert.equal((app.html().match(/class="chart association-chart"/g)||[]).length,9);
  const tests=data.rankAssociation.tests.filter(r=>r.definition==="sampling");
  assert.equal(tests.length,9);
  assert.equal((app.html().match(/class="association-point /g)||[]).length,142);
  for(const t of tests){
    const rows=data.rankAssociation.points.filter(r=>r.model===t.model&&r.sampling!==null&&r.negation!==null);
    assert.equal(rows.length,t.n);
    assert.ok(Math.abs(math.spearman(rows.map(r=>r.sampling),rows.map(r=>r.negation))-t.R)<1e-12);
    if([3,6].includes(t.model))assert.deepEqual(t.omitted,["mask"]);
  }
  assert.match(app.html(),/R = 0.744/);assert.match(app.html(),/p\(Holm\) = 0.012/);
  app.click({rankContext:"beach",rankModel:"4"});
  assert.match(app.html(),/Word-ranking Spearman · beach/);
  assert.match(app.html(),/ρ = 0.943/);assert.match(app.html(),/ρ = 0.498/);
  app.change("wordSource","target");assert.match(app.html(),/ρ = 0.943/);
  app.click({view:"association"});assert.match(app.html(),/Separate direct-target scores \(comparison\)/);
  assert.match(app.html(),/R = 0.661/);
  app.change("wordSource","sampling");assert.match(app.html(),/R = 0.661/);
  assert.ok(!/\bNaN\b|>undefined</.test(app.html()));
});

test("word ranks agree with exported sampling candidates, without changing exclusion predictions",()=>{
  for(const r of data.spearman.word_paired_ranks){
    const prompt=data.prompts.find(p=>p.id===r.prompt_id);
    const candidate=prompt.distribution.find(c=>c.word===r.word);
    assert.equal(candidate.logp,r.model_value);
    assert.equal(candidate.rank,r.vocabulary_rank);
  }
  assert.deepEqual(data.spearman.negation_paired_ranks,data.targetScoreSpearman.negation_paired_ranks);
  assert.equal(data.run.corrected,true);
  assert.deepEqual(data.scoreAlignment,{compared:947,mismatched:0,missing:0,matched:true});
  const app=harness();app.click({view:"spearman"});app.change("context","fridge");
  assert.match(app.html(),/ρ = 1.000/);
  app.change("wordSource","target");assert.match(app.html(),/ρ = 1.000/);
});

test("sampling-target browser and CSV exports reflect the selected score source",async()=>{
  const app=harness();
  const fridge=data.prompts.find(p=>p.frame==="Neutral"&&p.contexts.includes("fridge"));
  app.click({prompt:fridge.id});app.change("candidateSource","samplingTargets");
  assert.match(app.html(),/<h2>Experimental candidates · sampling-array scores<\/h2>/);
  assert.match(app.html(),/6 shown/);
  app.click({download:"words"});
  app.click({view:"association"});app.click({download:"association"});app.click({download:"association-tests"});
  const [words,points,tests]=await app.downloads();
  assert.equal(words.split("\r\n").length,7);
  assert.match(words,/samplingTargets/);
  assert.equal(points.split("\r\n").length,145);
  assert.equal(tests.split("\r\n").length,10);
  assert.match(points,/word_score_source/);assert.match(points,/"sampling"/);
  assert.match(tests,/holm18/);assert.match(tests,/199999/);
});

test("Overview Spearman matrix preserves saved values, coverage, exports, and drill-down",async()=>{
  const app=harness();
  assert.match(app.html(),/<option value="rho"/);
  app.change("aggregation","pooled");app.change("metric","rho");
  assert.match(app.html(),/Datasets × linking structures/);
  assert.match(app.html(),/Spearman ρ by dataset and linking structure/);
  assert.ok(!app.html().includes('id="aggregation"'));
  assert.ok(!/\bNaN\b|>undefined</.test(app.html()));
  assert.equal([...app.html().matchAll(/data-cell="dataset\|/g)].length,90);
  for(const d of data.datasets){
    for(let m=0;m<9;m++){
      const cell=app.html().match(new RegExp(`data-cell="dataset\\|${d.id}\\|${m}"[^>]*>([^<]*)</button>`));
      const model=data.rankAssociation.tests.find(r=>r.definition==="sampling"&&r.model===m);
      const focus=data.spearman.mean_within_context_spearman.find(r=>r.measure==="negation"&&r.structure===model.structure&&r.variant===model.variant);
      const s=data.datasetSummaries[d.id][m],rho=d.id==="novel_focus"?focus.mean_within_context_spearman:s.rho;
      assert.equal(cell[1],!s.n?"—":rho===null?"N/A":rho.toFixed(2));
    }
  }
  assert.match(app.html(),/Set Top-p: 15\/16/);
  app.click({download:"overview"});
  const [csv]=await app.downloads();
  assert.match(csv,/spearman_rho/);assert.match(csv,/mean_within_context/);
  assert.equal(csv.split("\r\n").length,91);
  app.change("coverage","shared");
  assert.equal([...app.html().matchAll(/data-cell="dataset\|/g)].length,45);
  assert.ok(!app.html().includes('data-cell="dataset|rnx_'));
  app.click({cell:"dataset|hu_vt16|2"});assert.match(app.html(),/Inspect the 39 item ranks/);
  app.click({view:"overview"});assert.match(app.html(),/Download Spearman matrix/);
  app.click({cell:"dataset|novel_focus|3"});
  assert.match(app.html(),/Negation Spearman/);assert.match(app.html(),/value="set\|top_p" selected/);
  app.click({view:"overview"});app.change("metric","r");assert.match(app.html(),/Pooled Pearson r/);
  app.change("metric","log");assert.match(app.html(),/Pooled item log score/);
});

test("non-focus Spearman uses the same saved items and is available alongside Pearson",()=>{
  const app=harness();app.click({view:"datasets"});
  for(const dataset of data.datasets.filter(d=>d.id!=="novel_focus")){
    app.change("dataset",dataset.id);app.change("metric","rho");
    assert.match(app.html(),/Three fit measures/);
    assert.match(app.html(),/Spearman ρ/);
    for(let m=0;m<9;m++){
      app.change("model",String(m));
      const rows=data.items.filter(r=>r.dataset===dataset.id&&r.p[m]!==null),s=data.datasetSummaries[dataset.id][m];
      if(!rows.length){assert.match(app.html(),/not applicable/);continue;}
      const counts=[...app.html().matchAll(/data-rank-count="(\d+)"/g)].reduce((s,m)=>s+Number(m[1]),0);
      assert.equal(counts,rows.length);
      assert.ok(Math.abs(math.spearman(rows.map(r=>r.y),rows.map(r=>r.p[m]))-s.rho)<1e-10);
    }
  }
  app.change("dataset","novel_focus");
  assert.ok(!app.html().includes('<option value="rho"'));
  assert.equal(data.datasetSummaries.novel_focus[0].rho,null);
  app.click({view:"structures"});app.change("metric","rho");
  assert.match(app.html(),/Explore focus within-context Spearman/);
  app.click({view:"contexts"});assert.ok(!app.html().includes('<option value="rho"'));
});

// Exercise the shipped click handler with DOM-shaped rows and real fridge data.
function sortableTable(values){
  const headers=["Trigger","Query","Prediction"].map(label=>{
    const arrow={textContent:""},button={dataset:{label},setAttribute(){},querySelector(){return arrow;}};
    return {attributes:{},setAttribute(k,v){this.attributes[k]=v;},querySelector(){return button;}};
  });
  const body={rows:values.map(row=>({cells:row.map(value=>({dataset:{sortValue:value===null?"":String(value)},textContent:"0.000"}))})),
    append(...rows){this.rows=rows;}};
  return {dataset:{sortKey:"spearman:paired"},tHead:{rows:[{cells:headers}]},tBodies:[body]};
}
test("paired table sorts Query in both directions and preserves every fridge pair",()=>{
  const app=harness(),rows=data.items.filter(r=>r.context==="fridge"&&r.dataset==="novel_focus");
  const table=sortableTable(rows.map(r=>[r.trigger,r.query,r.p[0]]));
  app.click({sortColumn:"1"},table);
  const queries=()=>table.tBodies[0].rows.map(r=>r.cells[1].dataset.sortValue);
  assert.deepEqual(queries(),rows.map(r=>r.query).sort());
  assert.equal(table.tHead.rows[0].cells[1].attributes["aria-sort"],"ascending");
  app.click({sortColumn:"1"},table);
  assert.deepEqual(queries(),rows.map(r=>r.query).sort().reverse());
  assert.equal(table.tHead.rows[0].cells[1].attributes["aria-sort"],"descending");
  assert.equal(table.tBodies[0].rows.length,30);
  // Raw values, not rounded display strings, control probability sorting.
  app.click({sortColumn:"2"},table);
  const values=table.tBodies[0].rows.map(r=>Number(r.cells[2].dataset.sortValue));
  assert.deepEqual(values,rows.map(r=>r.p[0]).sort((a,b)=>a-b));
  assert.equal(table.tHead.rows[0].cells[1].attributes["aria-sort"],"none");
});
test("numeric sorting handles scientific notation and keeps missing values last both ways",()=>{
  const app=harness(),table=sortableTable([["a","a",null],["b","b",10],["c","c",2],["d","d",1e-7],["e","e",-0.4]]);
  const values=()=>table.tBodies[0].rows.map(r=>r.cells[2].dataset.sortValue);
  app.click({sortColumn:"2"},table);
  assert.deepEqual(values(),["-0.4","1e-7","2","10",""]);
  app.click({sortColumn:"2"},table);
  assert.deepEqual(values(),["10","2","1e-7","-0.4",""]);
});
test("item sorting applies before pagination and remains active on the next page",()=>{
  const app=harness();app.click({view:"contexts"});app.change("context","fridge");
  app.click({sortColumn:"2"},{dataset:{sortKey:"contexts:items",sortId:"items"}});
  const ids=()=>[...app.html().match(/<table data-sort-id="items">([\s\S]*?)<\/table>/)[1].matchAll(/data-item="novel_focus\|([^"]+)"/g)].map(m=>m[1]);
  const expected=data.items.filter(r=>r.context==="fridge"&&r.dataset==="novel_focus").sort((a,b)=>a.y-b.y).map(r=>r.id);
  assert.deepEqual(ids(),expected.slice(0,20));
  app.click({page:"1"});
  assert.deepEqual(ids(),expected.slice(20));
  app.click({sortColumn:"2"},{dataset:{sortKey:"contexts:items",sortId:"items"}});
  assert.match(app.html(),/page 1 of 2/);
});
