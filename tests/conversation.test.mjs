// End-to-end dialogue test: text -> entities -> BiLSTM intent -> response,
// including multi-turn context and speech-recognition style misspellings.
// Run: node tests/conversation.test.mjs
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { IntentModel } from '../web/js/model.js';
import { Chatbot } from '../web/js/chatbot.js';

const root = new URL('../', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('web/model/model.json', root)));
const bin = readFileSync(new URL('web/model/model.bin', root));
const db = JSON.parse(readFileSync(new URL('web/data/cars.json', root)));
const bot = new Chatbot(new IntentModel(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)), db);

// [user utterance, expected intent, expected car ids (in order) or null]
const dialogue = [
  ['hello', 'greeting', []],
  ['tell me about the bugatti chiron', 'car_overview', ['Bugatti Chiron Super Sport']],
  ['how fast is it', 'car_top_speed', ['Bugatti Chiron Super Sport']],
  ['what about the veyron', 'car_overview', ['Bugatti Veyron 16.4']],
  ['how much does it cost', 'car_price', ['Bugatti Veyron 16.4']],
  ['where can i buy one', 'where_to_buy', ['Bugatti Veyron 16.4']],
  ['compare it with the koenigsegg jesko', 'compare', ['Bugatti Veyron 16.4', 'Koenigsegg Jesko Absolut']],
  ["what's the top speed of the Rimac Nevera?", 'car_top_speed', ['Rimac Nevera']],
  ['is the lamborghini revuelto a hybrid', 'car_engine', ['Lamborghini Revuelto']],
  ['how much horsepower does the mc20 have', 'car_power', ['Maserati MC20']],
  ['zero to sixty of the corvette zr1', 'car_acceleration', ['Chevrolet Corvette ZR1']],
  ['how many porsche 918 were made', 'car_production', ['Porsche 918 Spyder']],
  ['specs of the gordon murray t50', 'car_specs', ['Gordon Murray T.50']],
  ['I want to buy a Lamborghini Temerario', 'where_to_buy', ['Lamborghini Temerario']],
  ['where can i buy a mclaren', 'brand_buy', []],
  ['tell me about ferrari', 'brand_info', []],
  ['what can I buy with 300k', 'budget_recommend', []],
  ['recommend me a car', 'budget_recommend', []],
  ['500 thousand', 'budget_recommend', []],
  ['what is the fastest car', 'fastest_car', []],
  ['most powerful ferrari', 'most_powerful', []],
  ['cheapest supercar', 'cheapest', []],
  ['show me electric hypercars', 'electric_hybrid', []],
  ['what is a hypercar', 'definition', []],
  ['what cars do you know', 'list_cars', []],
  ['what is the price of tesla roadster', 'unknown_car', []],
  ['bugatti sheeran price', 'car_price', ['Bugatti Chiron Super Sport']],   // misheard "Chiron"
  ['pagani wire a top speed', 'car_top_speed', ['Pagani Huayra']],          // misheard "Huayra"
  ['which is quicker the nevera or the evija', 'compare', ['Rimac Nevera', 'Lotus Evija']],
  ['tell me a joke', 'out_of_scope', []],
  ['thanks', 'thanks', []],
  ['bye', 'goodbye', []],
];

let ok = 0;
for (const [text, intent, cars] of dialogue) {
  const r = bot.reply(text);
  assert.equal(r.intent, intent, `"${text}" -> ${r.intent}, expected ${intent}`);
  assert.deepEqual(r.entities.cars, cars, `"${text}" cars`);
  assert.ok(r.html && r.speech, `"${text}" produced an empty answer`);
  ok++;
}
console.log(`conversation ok: ${ok} turns`);
