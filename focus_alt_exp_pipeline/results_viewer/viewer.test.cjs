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
      for(const metric of ["r","log"]){
        if(stats[model][metric]===null)assert.equal(actual[metric],null);
        else assert.ok(Math.abs(actual[metric]-stats[model][metric])<1e-10,`${id} / ${model} / ${metric}`);
      }
    }
  }
  for(const [context,stats] of Object.entries(data.contextSummaries)){
    for(let model=0;model<9;model++){
      const actual=math.summary(data.items.filter(r=>r.context===context),model);
      for(const metric of ["r","log"]){
        if(stats[model][metric]===null)assert.equal(actual[metric],null);
        else assert.ok(Math.abs(actual[metric]-stats[model][metric])<1e-10);
      }
    }
  }
});

function harness(){
  const listeners={},elements={};
  const element=id=>elements[id]||(elements[id]={id,innerHTML:"",textContent:"",focus(){},setAttribute(){},removeAttribute(){},scrollIntoView(){}});
  element("results-data").textContent=JSON.stringify(data);
  const document={activeElement:null,getElementById:element,querySelectorAll:()=>[],addEventListener:(name,fn)=>{listeners[name]=fn;},createElement:()=>({click(){}})};
  const sandbox={document,window:{scrollTo(){},addEventListener(){}},console,Blob,URL,setTimeout};
  vm.createContext(sandbox);
  // Exercise the scripts actually shipped in the HTML, not only the JS source.
  for(const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(script[1],sandbox);
  return {html:()=>element("main").innerHTML,
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
  app.click({prompt:bag.id});assert.match(app.html(),/not the vocabulary’s top 50/);
  app.input("wordSearch","no-such-candidate-123");assert.match(app.html(),/No matching candidates/);
  const mask=data.prompts.find(p=>p.frame==="Neutral"&&p.contexts.includes("mask"));
  app.click({prompt:mask.id});assert.match(app.html(),/Top 50 from the vocabulary/);
  app.change("wordScale","normalized");assert.match(app.html(),/0\.6522/);
  app.change("candidateSource","targets");assert.match(app.html(),/Saved experimental candidates/);
  app.change("promptDataset","rnx_esi");app.change("frame","X but not Y");assert.match(app.html(),/No prompts match/);
  app.change("frame","Neutral");assert.match(app.html(),/ESI/);assert.match(app.html(),/Eonly/);
  app.click({view:"datasets"});app.input("itemSearch","<script>bad</script>");assert.match(app.html(),/&lt;script&gt;/);assert.ok(!app.html().includes("<script>bad"));
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
