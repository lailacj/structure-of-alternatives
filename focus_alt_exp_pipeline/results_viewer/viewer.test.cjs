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
  const document={activeElement:null,getElementById:element,querySelectorAll:()=>[],addEventListener:(name,fn)=>{listeners[name]=fn;},createElement:()=>({click(){}})};
  const sandbox={document,window:{scrollTo(){},addEventListener(){}},console,Blob,
    URL:{createObjectURL(blob){downloads.push(blob);return "blob:test";},revokeObjectURL(){}},setTimeout:fn=>fn()};
  vm.createContext(sandbox);
  // Exercise the scripts actually shipped in the HTML, not only the JS source.
  for(const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script[1],sandbox);
  return {html:()=>element("main").innerHTML,downloads:()=>Promise.all(downloads.map(blob=>blob.text())),
    click:dataset=>listeners.click({target:{closest:()=>({dataset})}}),
    change:(id,value)=>listeners.change({target:{id,value}}),
    input:(id,value)=>listeners.input({target:{id,value}})};
}

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
  app.change("wordScale","normalized");assert.match(app.html(),/0\.6522/);
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
  assert.match(fallback,/-3\.219/);
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
  assert.match(app.html(),/ρ = 0.886/);assert.match(app.html(),/ρ = 0.784/);
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
  assert.match(app.html(),/Historical sampled-prefix run/);
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
  assert.match(app.html(),/R = 0.851/);assert.match(app.html(),/p\(Holm\) &lt; .001/);
  app.click({rankContext:"beach",rankModel:"4"});
  assert.match(app.html(),/Word-ranking Spearman · beach/);
  assert.match(app.html(),/ρ = 0.714/);assert.match(app.html(),/ρ = 0.303/);
  app.change("wordSource","target");assert.match(app.html(),/ρ = 0.943/);
  app.click({view:"association"});assert.match(app.html(),/Separate direct-target scores \(comparison\)/);
  assert.match(app.html(),/R = 0.356/);
  app.change("wordSource","sampling");assert.match(app.html(),/R = 0.183/);
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
  assert.equal(data.scoreAlignment.matched,false);
  const app=harness();app.click({view:"spearman"});app.change("context","fridge");
  assert.ok(!app.html().includes("model ranks all six words in the human order"));
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
