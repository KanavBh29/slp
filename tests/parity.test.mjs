// Checks that the browser (pure JS) BiLSTM reproduces the Keras predictions
// and that preprocessing / entity extraction behave as expected.
// Run: node tests/parity.test.mjs
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { IntentModel } from '../web/js/model.js';
import { tokenize, encode, normalize, EntityExtractor, parseBudget } from '../web/js/nlp.js';

const root = new URL('../', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('web/model/model.json', root)));
const bin = readFileSync(new URL('web/model/model.bin', root));
const model = new IntentModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const ref = JSON.parse(readFileSync(new URL('tests/reference_predictions.json', root)));

let maxDiff = 0;
for (const r of ref) {
  const probs = model.predictIds(encode(tokenize(r.text), model.wordIndex, model.maxlen));
  const label = model.labels[probs.indexOf(Math.max(...probs))];
  assert.equal(label, r.label, `label mismatch for "${r.text}"`);
  probs.forEach((p, i) => { maxDiff = Math.max(maxDiff, Math.abs(p - r.probs[i])); });
}
assert.ok(maxDiff < 1e-3, `probabilities differ by ${maxDiff}`);
console.log(`parity ok: ${ref.length} sentences, max |p_js - p_keras| = ${maxDiff.toExponential(2)}`);

// Python and JS preprocessing must agree (same cases as tests/test_nlp.py).
const cases = JSON.parse(readFileSync(new URL('tests/tokenize_cases.json', root)));
for (const c of cases) assert.deepEqual(tokenize(c.text), c.tokens, `tokenize("${c.text}")`);
console.log(`tokenizer ok: ${cases.length} cases match Python`);

// Entity extraction, including speech-recognition style misspellings.
const db = JSON.parse(readFileSync(new URL('web/data/cars.json', root)));
const ex = new EntityExtractor(db, new Set(meta.vocab));
const ent = s => ex.extract(normalize(s));
const expect = [
  ['How fast is the Bugatti Chiron?', ['chiron-super-sport']],
  ['compare the LaFerrari vs the McLaren P1', ['laferrari', 'p1']],
  ['how much is the bugatti sheeran', ['chiron-super-sport']],
  ['koenig seg jesko price', ['jesko-absolut']],
  ['tell me about the rimac never a', ['nevera']],
  ['Porsche 918 Spyder top speed', ['918-spyder']],
  ['where can i buy a lamborghini revuelto', ['revuelto']],
  ['what is the weather today', []],
  ['can i afford a ford gt', ['ford-gt']],
];
for (const [s, cars] of expect) assert.deepEqual(ent(s).cars, cars, `cars in "${s}"`);
assert.deepEqual(ent('show me ferrari cars').brands, ['Ferrari']);
assert.equal(ent('tell me about the bugatti chiron').masked, 'tell me about the xcar');
console.log(`entities ok: ${expect.length + 2} cases`);

const budgets = [
  ['what can i buy with $300k', 300000], ['budget of 2.5 million dollars', 2500000],
  ['i have two million', 2000000], ['under 500,000', 500000], ['half a million', 500000],
  ['five hundred thousand dollars', 500000], ['cars under 3 crore rupees', Math.round(3e7 / 84)],
  ['what is the top speed', null],
];
for (const [s, v] of budgets) assert.equal(parseBudget(s), v, `budget "${s}"`);
console.log(`budget parser ok: ${budgets.length} cases`);
