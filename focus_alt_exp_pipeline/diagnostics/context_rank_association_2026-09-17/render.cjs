// Render only the three SVGs generated in this analysis directory.
const path = require('node:path');
const sharp = require('/Users/lailajohnston/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
(async () => {
  for (const name of ['scatter_viewer', 'scatter_sampling_scores', 'exceptions']) {
    const input = path.join(__dirname, name + '.svg');
    const output = path.join(__dirname, name + '.png');
    await sharp(input, {density: 144}).png().toFile(output);
    console.log(output);
  }
})().catch(error => {console.error(error); process.exit(1);});
