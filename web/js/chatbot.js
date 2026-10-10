// Dialogue manager: speech text -> entities -> BiLSTM intent -> response.
import { normalize, maskNumbers, encode, EntityExtractor, parseBudget } from './nlp.js';

const CONFIDENCE_THRESHOLD = 0.5;
const CAR_INTENTS = new Set(['car_overview', 'car_specs', 'car_engine', 'car_power', 'car_top_speed',
  'car_acceleration', 'car_price', 'where_to_buy', 'car_production']);

const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const mph = kmh => Math.round(kmh * 0.621371);
const kw = hp => Math.round(hp * 0.7457);

export function fmtPrice(usd) {
  if (usd >= 1e6) return `$${(usd / 1e6).toFixed(usd % 1e6 === 0 ? 0 : 1).replace(/\.0$/, '')} million`;
  return `$${usd.toLocaleString('en-US')}`;
}
const spokenPrice = usd => fmtPrice(usd).replace('$', '') + (usd >= 1e6 ? ' dollars' : ' dollars');
const priceLabel = c => (['available', 'limited'].includes(c.status) ? 'approx. base price' : 'approx. market value');
const statusText = {
  available: 'in production and orderable new',
  limited: 'a limited-production model',
  'sold out': 'sold out (all build slots allocated)',
  discontinued: 'out of production (pre-owned only)',
};

// Follow-up questions ("how fast is it?", "price please") reuse the last car.
const BACK_REFS = new Set(['it', 'its', 'itself', 'one', 'this', 'that', 'them', 'they', 'same', 'car', 'he', 'she']);
const refersBack = tokens => tokens.length <= 4 || tokens.some(t => BACK_REFS.has(t));

export class Chatbot {
  constructor(model, db) {
    this.model = model;
    this.db = db;
    this.cars = new Map(db.cars.map(c => [c.id, c]));
    this.extractor = new EntityExtractor(db, new Set(model.vocab));
    this.lastCars = [];
    this.awaiting = null;
  }

  analyse(text) {
    const norm = normalize(text);
    const ents = this.extractor.extract(norm);
    const masked = maskNumbers(ents.masked);
    const tokens = masked ? masked.split(' ') : [];
    const ids = encode(tokens, this.model.wordIndex, this.model.maxlen);
    const probs = this.model.predictIds(ids);
    const ranked = probs.map((p, i) => ({ intent: this.model.labels[i], p })).sort((a, b) => b.p - a.p);
    return {
      norm, tokens, ids: ids.filter(Boolean), ranked,
      oov: tokens.filter((t, i) => ids[i] === 1),
      cars: ents.cars, brands: ents.brands, spans: ents.spans, budget: parseBudget(text),
    };
  }

  reply(text) {
    const a = this.analyse(text);
    let intent = a.ranked[0].intent;
    const confidence = a.ranked[0].p;
    let note = null;

    // Dialogue state: the previous turn asked for a budget / a car.
    if (this.awaiting === 'budget' && a.budget) intent = 'budget_recommend';
    else if (this.awaiting?.startsWith('car:') && a.cars.length && !CAR_INTENTS.has(intent) && intent !== 'compare') {
      intent = this.awaiting.slice(4);
    }
    this.awaiting = null;

    if (confidence < CONFIDENCE_THRESHOLD) {
      if (a.cars.length > 1) intent = 'compare';
      else if (a.cars.length) intent = 'car_overview';
      else if (a.brands.length) intent = 'brand_info';
      else if (a.budget) intent = 'budget_recommend';
      else intent = 'fallback';
      note = 'low confidence → rule-based fallback';
    }
    if (intent === 'out_of_scope' && a.cars.length && confidence < 0.9) intent = 'car_overview';

    // Resolve which car(s) the user means, using conversation memory.
    let cars = a.cars.map(id => this.cars.get(id));
    if (CAR_INTENTS.has(intent) && !cars.length) {
      const last = this.cars.get(this.lastCars[0]);
      if (a.brands.length) {
        // "where can I buy a McLaren": only reuse the remembered car if it is that brand.
        if (last && a.brands.includes(last.brand)) cars = [last];
        else if (intent === 'where_to_buy') intent = 'brand_buy';
        else intent = 'brand_info';
      } else if (last && refersBack(a.tokens)) {
        cars = [last];
      } else if (a.tokens.length > 4) {
        intent = 'unknown_car';
      }
    }
    if (intent === 'compare' && cars.length === 1 && this.lastCars.length && this.lastCars[0] !== cars[0].id) {
      cars = [this.cars.get(this.lastCars[0]), cars[0]];
    } else if (intent === 'compare' && !cars.length && this.lastCars.length >= 2) {
      cars = this.lastCars.slice(0, 2).map(id => this.cars.get(id));
    }
    if (a.cars.length) this.lastCars = [...a.cars, ...this.lastCars.filter(id => !a.cars.includes(id))].slice(0, 4);

    const res = this.respond(intent, cars, a);
    return {
      ...res, intent, confidence, top: a.ranked.slice(0, 3), note,
      tokens: a.tokens, oov: a.oov, entities: { cars: cars.map(c => c.name), brands: a.brands, budget: a.budget },
      spans: a.spans,
    };
  }

  respond(intent, cars, a) {
    const car = cars[0];
    if (CAR_INTENTS.has(intent) && !car) {
      this.awaiting = `car:${intent}`;
      return { html: 'Which car do you mean? Try something like <em>“the Bugatti Chiron”</em> or <em>“Ferrari F80”</em>.', speech: 'Which car do you mean?' };
    }
    const many = (fn) => {
      const parts = cars.slice(0, 3).map(fn);
      return { html: parts.map(p => p.html).join('<br>'), speech: parts.map(p => p.speech).join(' ') };
    };
    switch (intent) {
      case 'greeting':
        return this.say(pick([
          "Hey! I'm HyperBot. Ask me about any supercar or hypercar: specs, top speed, price or where to buy one.",
          'Hello! Which supercar or hypercar are you curious about today?',
          'Hi there! Tap the mic and ask me something like “How fast is the Koenigsegg Jesko?”',
        ]));
      case 'goodbye':
        return this.say(pick(['Goodbye! Keep it pinned. 🏁', 'See you later. Drive fast, stay safe!', 'Bye! Come back when you’re ready to configure your hypercar.']));
      case 'thanks':
        return this.say(pick(['You’re welcome! Anything else you want to know?', 'Happy to help! Want to compare it with another car?', 'Anytime! Ask me about another hypercar.']));
      case 'bot_identity':
        return this.say('I’m HyperBot, a voice-enabled chatbot. Your speech is turned into text by a speech recognition engine, then a Bidirectional LSTM neural network (with GloVe word embeddings) classifies your intent, and I answer from a database of ' + this.db.cars.length + ' supercars and hypercars.');
      case 'help':
        return {
          html: `I can answer questions about ${this.db.cars.length} supercars and hypercars from ${Object.keys(this.db.brands).length} brands. Try:<ul>
            <li>“Tell me about the Pagani Utopia”</li><li>“What are the specs of the Rimac Nevera?”</li>
            <li>“How much does a Ferrari F80 cost?”</li><li>“Where can I buy a McLaren 750S?”</li>
            <li>“Compare the Chiron and the Jesko”</li><li>“What is the fastest car?”</li>
            <li>“What can I buy with $400k?”</li><li>“Show me electric hypercars”</li></ul>`,
          speech: 'You can ask me about specs, top speed, acceleration, price, where to buy, comparisons, rankings, or recommendations for your budget.',
        };
      case 'car_overview': return this.overview(car);
      case 'car_specs': return { ...this.say(`Here is the spec sheet for the ${car.name}.`), card: this.specCard(car) };
      case 'car_engine':
        return many(c => this.say(`The ${c.name} is powered by a ${c.engine} (${c.powertrain.toLowerCase()} powertrain), producing ${c.power_hp.toLocaleString()} hp.`));
      case 'car_power':
        return many(c => this.say(`The ${c.name} produces ${c.power_hp.toLocaleString()} hp (${kw(c.power_hp).toLocaleString()} kW)` +
          (c.torque_nm ? ` and ${c.torque_nm.toLocaleString()} Nm of torque.` : '. Torque has not been officially disclosed.')));
      case 'car_top_speed':
        return many(c => this.say(`The ${c.name} has a top speed of ${c.top_speed_kmh} km/h (${mph(c.top_speed_kmh)} mph).`));
      case 'car_acceleration':
        return many(c => this.say(`The ${c.name} sprints from 0 to 100 km/h in ${c.zero_to_100_s} seconds.`));
      case 'car_price':
        return many(c => ({
          html: `The <b>${esc(c.name)}</b> costs about <b>${fmtPrice(c.price_usd)}</b> (${priceLabel(c)}). It is ${statusText[c.status]}.`,
          speech: `The ${c.name} costs about ${spokenPrice(c.price_usd)}.`,
        }));
      case 'where_to_buy': return this.whereToBuy(car);
      case 'brand_buy': return this.brandBuy(a.brands[0]);
      case 'unknown_car':
        this.awaiting = null;
        return {
          html: 'I don’t have that car in my database yet. I cover ' + this.db.cars.length +
            ' supercars and hypercars, e.g. the Bugatti Tourbillon, Ferrari F80, McLaren W1 or Rimac Nevera. Say <em>“what cars do you know”</em> for the full list.',
          speech: 'Sorry, I don’t have that car in my database yet. Ask me what cars I know to hear the full list.',
        };
      case 'car_production':
        return many(c => this.say(`The ${c.name}: ${c.production}. It is ${statusText[c.status]}.`));
      case 'compare': return this.compare(cars);
      case 'fastest_car': return this.ranking('top_speed_kmh', 'desc', 'Top speed', c => `${c.top_speed_kmh} km/h`, a.brands);
      case 'quickest_car': return this.ranking('zero_to_100_s', 'asc', 'Quickest 0–100 km/h', c => `${c.zero_to_100_s} s`, a.brands);
      case 'most_powerful': return this.ranking('power_hp', 'desc', 'Most powerful', c => `${c.power_hp.toLocaleString()} hp`, a.brands);
      case 'most_expensive': return this.ranking('price_usd', 'desc', 'Most expensive', c => fmtPrice(c.price_usd), a.brands);
      case 'cheapest': return this.ranking('price_usd', 'asc', 'Most affordable', c => fmtPrice(c.price_usd), a.brands);
      case 'budget_recommend': return this.budget(a.budget);
      case 'electric_hybrid': return this.electrified();
      case 'list_cars': return this.listAll();
      case 'brand_info': return this.brand(a.brands[0] || (car && car.brand));
      case 'definition':
        return this.say('There is no official rule, but a supercar is a high-performance exotic car (think Ferrari 296 GTB or McLaren 750S, roughly $200k–$600k, 600–1,000 hp). A hypercar is the top 1% of that: very limited production, usually over $1 million and 1,000+ hp, like the Bugatti Chiron, Koenigsegg Jesko or Rimac Nevera. The McLaren P1, Ferrari LaFerrari and Porsche 918 Spyder are called the “holy trinity” of hybrid hypercars.');
      case 'out_of_scope':
        return this.say(pick([
          'I’m a car specialist, so that’s outside my lane! Ask me about supercars or hypercars instead.',
          'I only know about supercars and hypercars. Try asking “What is the fastest car?”',
        ]));
      default:
        return this.say('Sorry, I didn’t quite catch that. Ask me about a supercar’s specs, price, top speed or where to buy it. Say “help” for examples.');
    }
  }

  say(text) { return { html: esc(text), speech: text }; }

  overview(c) {
    return {
      html: `<b>${esc(c.name)}</b> (${c.year}) is a ${c.class.toLowerCase()} from ${esc(c.brand)} (${esc(this.db.brands[c.brand].country)}). ${esc(c.description)}`,
      speech: `The ${c.name} is a ${c.year} ${c.class.toLowerCase()} from ${c.brand}. ${c.description}`,
      card: this.specCard(c),
    };
  }

  specCard(c) {
    const rows = [
      ['Class', `${c.class} · ${c.powertrain}`], ['Engine', c.engine],
      ['Power', `${c.power_hp.toLocaleString()} hp (${kw(c.power_hp).toLocaleString()} kW)`],
      ['Torque', c.torque_nm ? `${c.torque_nm.toLocaleString()} Nm` : 'Not disclosed'],
      ['0–100 km/h', `${c.zero_to_100_s} s`], ['Top speed', `${c.top_speed_kmh} km/h (${mph(c.top_speed_kmh)} mph)`],
      ['Price', `${fmtPrice(c.price_usd)} <small>(${priceLabel(c)})</small>`], ['Production', c.production],
      ['Availability', statusText[c.status]],
    ];
    return `<div class="spec-card"><div class="spec-head"><span class="spec-brand">${esc(c.brand)}</span>
      <span class="spec-name">${esc(c.name.replace(c.brand, '').trim())}</span><span class="spec-year">${c.year}</span></div>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${k === 'Price' ? v : esc(v)}</dd>`).join('')}</dl></div>`;
  }

  links(list) {
    return `<ul class="links">${list.map(l => `<li><a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)} ↗</a></li>`).join('')}</ul>`;
  }

  whereToBuy(c) {
    const b = this.db.brands[c.brand];
    const official = { label: `${c.brand} official site & dealer locator`, url: b.website };
    const used = this.db.marketplaces.used;
    let html, speech;
    switch (c.status) {
      case 'available':
        html = `The <b>${esc(c.name)}</b> is on sale now. Order it new through an authorised ${esc(c.brand)} dealer (use the configurator and dealer locator on the official site). For pre-owned examples, try the marketplaces below.`;
        speech = `The ${c.name} is on sale now. You can configure and order one through an official ${c.brand} dealer, or find pre-owned cars on duPont Registry and JamesEdition.`;
        return { html: html + this.links([official, ...used.slice(0, 2)]), speech };
      case 'limited':
        html = `The <b>${esc(c.name)}</b> is built in limited numbers (${esc(c.production)}). Contact ${esc(c.brand)} directly to ask about remaining build slots, or look for one on the secondary market:`;
        speech = `The ${c.name} is a limited model. Contact ${c.brand} directly about remaining build slots, or check secondary markets like duPont Registry and JamesEdition.`;
        return { html: html + this.links([official, ...used]), speech };
      case 'sold out':
        html = `All ${esc(c.production)} of the <b>${esc(c.name)}</b> were allocated to selected clients, so you’ll need the secondary market. Expect to pay a premium over the original ${fmtPrice(c.price_usd)}. Authorised ${esc(c.brand)} dealers also broker resales:`;
        speech = `The ${c.name} is sold out, so you'd have to buy one second-hand, through a specialist like duPont Registry, JamesEdition, or an auction house like RM Sotheby's. ${c.brand} dealers also broker resales.`;
        return { html: html + this.links([...used, official]), speech };
      default:
        html = `The <b>${esc(c.name)}</b> is no longer in production. Look for certified pre-owned cars at ${esc(c.brand)} dealers, specialist marketplaces, or collector-car auctions:`;
        speech = `The ${c.name} is out of production, so look for a pre-owned one at ${c.brand} dealers, on duPont Registry or JamesEdition, or at auctions like Bring a Trailer and RM Sotheby's.`;
        return { html: html + this.links([official, ...used]), speech };
    }
  }

  brandBuy(brand) {
    const b = this.db.brands[brand];
    return {
      html: `You can find authorised ${esc(brand)} dealers through the official dealer locator. For pre-owned or sold-out models, use specialist marketplaces:` +
        this.links([{ label: `${brand} official site & dealer locator`, url: b.website }, ...this.db.marketplaces.used]),
      speech: `Use the dealer locator on the official ${brand} website, or check specialist marketplaces for pre-owned cars.`,
    };
  }

  compare(cars) {
    if (cars.length < 2) {
      this.awaiting = cars.length ? null : 'car:compare';
      return this.say(cars.length ? `What should I compare the ${cars[0].name} with?` : 'Which two cars should I compare? For example: “Chiron vs Jesko”.');
    }
    const [x, y] = cars;
    const rows = [
      ['Power', 'power_hp', 'desc', c => `${c.power_hp.toLocaleString()} hp`],
      ['Torque', 'torque_nm', 'desc', c => c.torque_nm ? `${c.torque_nm.toLocaleString()} Nm` : 'n/a'],
      ['0–100 km/h', 'zero_to_100_s', 'asc', c => `${c.zero_to_100_s} s`],
      ['Top speed', 'top_speed_kmh', 'desc', c => `${c.top_speed_kmh} km/h`],
      ['Price', 'price_usd', 'asc', c => fmtPrice(c.price_usd)],
    ];
    const win = (key, dir) => {
      if (x[key] == null || y[key] == null || x[key] === y[key]) return 0;
      return (dir === 'desc' ? x[key] > y[key] : x[key] < y[key]) ? 1 : 2;
    };
    const table = `<table class="cmp"><thead><tr><th></th><th>${esc(x.name)}</th><th>${esc(y.name)}</th></tr></thead><tbody>
      ${rows.map(([label, key, dir, f]) => { const w = win(key, dir); return `<tr><th>${label}</th><td class="${w === 1 ? 'win' : ''}">${esc(f(x))}</td><td class="${w === 2 ? 'win' : ''}">${esc(f(y))}</td></tr>`; }).join('')}
      <tr><th>Engine</th><td>${esc(x.engine)}</td><td>${esc(y.engine)}</td></tr></tbody></table>`;
    const wins = [[], []];
    for (const [label, key, dir] of rows) {
      const w = win(key, dir);
      if (w) wins[w - 1].push(label === 'Price' ? 'price' : label.toLowerCase());
    }
    const list = arr => arr.length > 1 ? `${arr.slice(0, -1).join(', ')} and ${arr.at(-1)}` : arr[0];
    const parts = [x, y].map((c, i) => wins[i].length ? `the ${c.name} wins on ${list(wins[i])}` : null).filter(Boolean);
    const text = parts.length ? parts.join(', while ').replace(/^t/, 'T') + '.' : 'They are evenly matched.';
    return { html: `<b>${esc(x.name)}</b> vs <b>${esc(y.name)}</b>: ${esc(text)}`, speech: text, card: table };
  }

  ranking(key, dir, title, fmt, brands = []) {
    let list = this.db.cars.filter(c => c[key] != null);
    if (brands.length) list = list.filter(c => brands.includes(c.brand));
    list = [...list].sort((p, q) => dir === 'desc' ? q[key] - p[key] : p[key] - q[key]).slice(0, 5);
    const scope = brands.length ? ` (${brands.join(', ')})` : '';
    const top = list[0];
    return {
      html: `<b>${title}${esc(scope)}</b> in my database:<ol class="rank">${list.map(c => `<li><span>${esc(c.name)}</span><b>${esc(fmt(c))}</b></li>`).join('')}</ol>` +
        (key === 'top_speed_kmh' && !brands.length ? '<small>Top speeds for the Jesko Absolut and Venom F5 are manufacturer targets; the fastest verified two-way production-car run is the SSC Tuatara’s ~455 km/h.</small>' : ''),
      speech: `Number one is the ${top.name} with ${fmt(top).replace('km/h', 'kilometres per hour').replace(' s', ' seconds')}, followed by the ${list[1]?.name ?? ''}.`,
    };
  }

  budget(usd) {
    if (!usd) {
      this.awaiting = 'budget';
      return this.say('Sure! What is your budget? For example “300 thousand dollars” or “2 million”.');
    }
    const fits = this.db.cars.filter(c => c.price_usd <= usd).sort((p, q) => q.price_usd - p.price_usd);
    if (!fits.length) {
      const cheapest = [...this.db.cars].sort((p, q) => p.price_usd - q.price_usd)[0];
      return this.say(`With ${fmtPrice(usd)} nothing in my database fits yet. The most affordable is the ${cheapest.name} at about ${fmtPrice(cheapest.price_usd)}.`);
    }
    const best = fits.slice(0, 5);
    return {
      html: `With a budget of <b>${fmtPrice(usd)}</b> you could consider:<ol class="rank">${best.map(c => `<li><span>${esc(c.name)} <small>(${c.power_hp.toLocaleString()} hp, ${c.status})</small></span><b>${fmtPrice(c.price_usd)}</b></li>`).join('')}</ol>` +
        (fits.length > 5 ? `<small>…plus ${fits.length - 5} more under budget.</small>` : ''),
      speech: `With ${spokenPrice(usd)}, my top pick is the ${best[0].name} at about ${spokenPrice(best[0].price_usd)}.`,
    };
  }

  electrified() {
    const ev = this.db.cars.filter(c => c.powertrain === 'Electric');
    const hy = this.db.cars.filter(c => c.powertrain === 'Hybrid');
    const li = arr => `<ul>${arr.map(c => `<li>${esc(c.name)} <small>${c.power_hp.toLocaleString()} hp</small></li>`).join('')}</ul>`;
    return {
      html: `<b>Fully electric (${ev.length}):</b>${li(ev)}<b>Hybrid (${hy.length}):</b>${li(hy)}`,
      speech: `The fully electric hypercars I know are the ${ev.map(c => c.name).join(' and the ')}. There are also ${hy.length} hybrids, like the ${hy.slice(0, 3).map(c => c.name).join(', ')}.`,
    };
  }

  listAll() {
    const byBrand = {};
    for (const c of this.db.cars) (byBrand[c.brand] ??= []).push(c.name.replace(c.brand, '').trim());
    return {
      html: `I know ${this.db.cars.length} cars:<ul class="brandlist">${Object.entries(byBrand).map(([b, cs]) => `<li><b>${esc(b)}</b>: ${cs.map(esc).join(', ')}</li>`).join('')}</ul>`,
      speech: `I know ${this.db.cars.length} cars from ${Object.keys(byBrand).length} brands, including Bugatti, Koenigsegg, Ferrari, Lamborghini, McLaren and Pagani.`,
    };
  }

  brand(name) {
    if (!name) return this.listAll();
    const b = this.db.brands[name];
    const cars = this.db.cars.filter(c => c.brand === name);
    return {
      html: `<b>${esc(name)}</b> is based in ${esc(b.country)}. Models I know:<ul>${cars.map(c => `<li>${esc(c.name)} <small>(${c.year}, ${c.power_hp.toLocaleString()} hp, ${fmtPrice(c.price_usd)})</small></li>`).join('')}</ul>` + this.links([{ label: `Official ${name} website`, url: b.website }]),
      speech: `${name} is from ${b.country}. I know about the ${cars.map(c => c.name).join(', ')}.`,
    };
  }
}
