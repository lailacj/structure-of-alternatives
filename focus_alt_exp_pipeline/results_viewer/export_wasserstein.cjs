"use strict";
// Export the same exact distances as the offline viewer, using its saved inputs.
// Usage: node export_wasserstein.cjs [viewer.html] > focus_wasserstein.csv
const fs=require("node:fs"),path=require("node:path");
const {focusWasserstein}=require("./viewer.js");
const html=fs.readFileSync(process.argv[2]||path.join(__dirname,"index.html"),"utf8");
const data=JSON.parse(html.match(/<script id="results-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const rows=focusWasserstein(data.items,data.models.length).map(row=>({...row,run:data.run.id,
  dataset:"Novel Focus Alternative Study",model:data.models[row.model]}));
const columns=Object.keys(rows[0]),quote=value=>'"'+String(value??"").replace(/"/g,'""')+'"';
process.stdout.write([columns,...rows.map(row=>columns.map(column=>row[column]))]
  .map(row=>row.map(quote).join(",")).join("\n")+"\n");
