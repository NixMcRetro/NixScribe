// Test the actual inline UI functions with lightweight DOM stand-ins.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('templates/index.html', 'utf8');
function setup() {
  const elements = {};
  const $ = id => elements[id] ||= {
    value: '', textContent: '', disabled: false, classList: {toggle() {}}, focus() {}
  };
  const start = html.indexOf('function updateModelRepository()');
  const end = html.indexOf('function updateDiarizationDevice()', start);
  assert.ok(start >= 0 && end > start);
  const source = html.slice(start, end).replace(/^[^\n]*\.addEventListener\([^\n]*\);\s*$/gm, '')
    .replace(/^updateModelRepository\(\);$/m, '')
    .replace(/^updatePromptCounts\(\);$/m, '');
  const ctx = vm.createContext({$});
  vm.runInContext(source, ctx);
  return {$, ctx};
}
test('MLX disables hotwords without losing text; CPU restores both inputs', () => {
  const {$, ctx} = setup();
  $('hotwords').value = 'unusual terminology';
  $('initialPrompt').value = 'Technical interview';
  $('model').selectedOptions = [{dataset: {engine:'mlx'}}];
  ctx.updateModelRepository();
  assert.equal($('hotwords').disabled, true);
  assert.match($('hotwordsHint').textContent, /not sent/);
  $('model').selectedOptions = [{dataset: {engine:'cpu'}}];
  ctx.updateModelRepository();
  assert.equal($('hotwords').disabled, false);
  assert.equal($('hotwords').value, 'unusual terminology');
  assert.equal($('initialPrompt').value, 'Technical interview');
  assert.match(html, /hotwords: \$\('hotwords'\)\.disabled \? '' :/);
});
test('counts and optional Australian spelling hint preserve original context', () => {
  const {$, ctx} = setup();
  $('initialPrompt').value = '  café  repair  ';
  $('hotwords').value = 'one two';
  ctx.updatePromptCounts();
  assert.equal($('promptCount').textContent, '2 words · 12 characters');
  assert.equal($('hotwordsCount').textContent, '2 words · 7 characters');
  ctx.addAustralianSpelling();
  ctx.addAustralianSpelling();
  assert.equal($('initialPrompt').value, 'Use Australian English spelling.\n\n  café  repair  ');
  assert.match($('promptCount').textContent, /^6 words/);
});
