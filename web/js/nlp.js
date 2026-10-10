// Text preprocessing + entity extraction.
// normalize / maskNumbers / tokenize / encode mirror training/nlp.py exactly.

export const MAXLEN = 24;

const NUM_RE = new RegExp(
  String.raw`(?<![a-z0-9])\$?\d+(?:[.,]\d+)*` +
  String.raw`(?:\s*(?:k|m|mn|mil|million|millions|thousand|grand|lakh|lakhs|crore|crores|usd|dollars)\b|[a-z]+)?`,
  'g'
);

export function normalize(text) {
  let t = text.toLowerCase().replaceAll('&', ' and ');
  t = t.replace(/['’]/g, '');
  t = t.replace(/[^a-z0-9$.]+/g, ' ');
  t = t.replace(/(?<![a-z0-9])\.|\.(?![a-z0-9])/g, ' ');
  return t.replace(/\s+/g, ' ').trim();
}

export function maskNumbers(text) {
  let t = text.replace(NUM_RE, 'xnum');
  t = t.replaceAll('$', ' ').replaceAll('.', ' ');
  return t.replace(/\s+/g, ' ').trim();
}

export function tokenize(text) {
  const t = maskNumbers(normalize(text));
  return t ? t.split(' ') : [];
}

export function encode(tokens, wordIndex, maxlen = MAXLEN) {
  const ids = tokens.slice(0, maxlen).map(t => wordIndex.get(t) ?? 1);
  while (ids.length < maxlen) ids.push(0);
  return ids;
}

// ---------------------------------------------------------------------------
// Entity extraction: cars, brands and budgets.

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function levenshtein(a, b) {
  const dp = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return dp[b.length];
}

export function similarity(a, b) {
  return 1 - levenshtein(a, b) / Math.max(a.length, b.length, 1);
}

export class EntityExtractor {
  constructor(db, commonWords = new Set()) {
    this.db = db;
    this.common = commonWords;
    this.entries = [];
    for (const car of db.cars) {
      const names = new Set([car.name, ...car.aliases].map(normalize));
      for (const a of names) this.entries.push({ alias: a, type: 'car', id: car.id });
    }
    for (const [brand, info] of Object.entries(db.brands)) {
      const names = new Set([brand, ...info.aliases].map(normalize));
      for (const a of names) this.entries.push({ alias: a, type: 'brand', id: brand });
    }
    // Longest alias first so "bugatti chiron" wins over "bugatti".
    this.entries.sort((x, y) => y.alias.length - x.alias.length);
    for (const e of this.entries) {
      e.re = new RegExp(`(?<![a-z0-9])${escapeRe(e.alias)}(?![a-z0-9])`, 'g');
    }
  }

  /** Returns {masked, cars, brands, spans} for a normalised sentence. */
  extract(norm) {
    const spans = [];
    const free = (s, e) => spans.every(sp => e <= sp.start || s >= sp.end);
    const exact = (type) => {
      for (const ent of this.entries) {
        if (ent.type !== type) continue;
        ent.re.lastIndex = 0;
        let m;
        while ((m = ent.re.exec(norm))) {
          const s = m.index, e = s + m[0].length;
          if (free(s, e)) spans.push({ start: s, end: e, type, id: ent.id, text: m[0], fuzzy: false });
        }
      }
    };
    // Cars first (exact, then fuzzy) so "bugatti sharon" -> Chiron, not brand Bugatti.
    exact('car');
    this.fuzzy(norm, spans, free, 'car');
    exact('brand');
    this.fuzzy(norm, spans, free, 'brand');
    spans.sort((a, b) => a.start - b.start);

    let masked = '', pos = 0;
    for (const sp of spans) {
      masked += norm.slice(pos, sp.start) + (sp.type === 'car' ? 'xcar' : 'xbrand');
      pos = sp.end;
    }
    masked += norm.slice(pos);
    const uniq = arr => [...new Set(arr)];
    return {
      masked: masked.replace(/\s+/g, ' ').trim(),
      cars: uniq(spans.filter(s => s.type === 'car').map(s => s.id)),
      brands: uniq(spans.filter(s => s.type === 'brand').map(s => s.id)),
      spans,
    };
  }

  // Speech recognisers often mishear exotic names ("bugatti sheeran",
  // "pagani wire a"). Unclaimed 1-3 word windows containing at least one
  // uncommon word are matched against the aliases by spelling (edit distance)
  // and by sound (a simplified phonetic key).
  fuzzy(norm, spans, free, type) {
    const words = [];
    const re = /[a-z0-9.]+/g;
    let m;
    while ((m = re.exec(norm))) words.push({ w: m[0], s: m.index, e: m.index + m[0].length });
    const cands = [];
    for (let n = 3; n >= 1; n--) {
      for (let i = 0; i + n <= words.length; i++) {
        const win = words.slice(i, i + n);
        const s = win[0].s, e = win[n - 1].e;
        if (!free(s, e)) continue;
        if (win.every(x => this.common.has(x.w) || /^\d/.test(x.w))) continue;
        const text = win.map(x => x.w).join(' ');
        const joined = text.replaceAll(' ', '');
        if (joined.length < 4) continue;
        const ph = phonetic(text);
        let best = null;
        for (const ent of this.entries) {
          if (ent.type !== type || ent.alias.length < 4) continue;
          ent.ph ??= phonetic(ent.alias);
          const sim = Math.max(
            similarity(text, ent.alias),
            similarity(joined, ent.alias.replaceAll(' ', '')),
            ph.length >= 3 && ent.ph.length >= 3 ? similarity(ph, ent.ph) - 0.05 : 0,
          );
          if (sim >= 0.8 && (!best || sim > best.sim)) best = { ...ent, sim };
        }
        if (best) cands.push({ start: s, end: e, type, id: best.id, text, fuzzy: true, sim: best.sim });
      }
    }
    cands.sort((a, b) => b.sim - a.sim || (b.end - b.start) - (a.end - a.start));
    for (const c of cands) if (free(c.start, c.end)) spans.push(c);
  }
}

/** Simplified Metaphone-style key: "sheeran" and "chiron" -> "xrn". */
export function phonetic(text) {
  return text.split(' ').map(w => {
    if (/^\d/.test(w)) return w;
    let t = w.replace(/ph/g, 'f').replace(/[cs]h/g, 'x').replace(/ck/g, 'k').replace(/c(?=[eiy])/g, 's')
      .replace(/[cq]/g, 'k').replace(/z/g, 's').replace(/gh/g, 'g').replace(/dg/g, 'j').replace(/wh/g, 'w')
      .replace(/([^aeiou])h/g, '$1').replace(/^kn/, 'n');
    t = t[0] + t.slice(1).replace(/[aeiouyhw]/g, '');
    if (/^[aeiouy]/.test(t)) t = 'a' + t.slice(1);
    return t.replace(/(.)\1+/g, '$1');
  }).join(' ');
}

// ---------------------------------------------------------------------------
// Budget parsing: "$300k", "300,000", "2.5 million", "two million dollars",
// "half a million", "5 crore".

const WORD_NUMS = {
  zero: 0, one: 1, a: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8,
  nine: 9, ten: 10, eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30,
  forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};
const MULT = {
  k: 1e3, grand: 1e3, thousand: 1e3, m: 1e6, mn: 1e6, mil: 1e6, million: 1e6, millions: 1e6,
  lakh: 1e5, lakhs: 1e5, crore: 1e7, crores: 1e7, hundred: 100,
};
const INR_PER_USD = 84;

export function parseBudget(raw) {
  let t = raw.toLowerCase().replace(/(\d),(\d{3})/g, '$1$2');
  t = t.replace(/half a million/g, '500000');
  // word numbers -> digits ("two point five million", "five hundred thousand")
  t = t.replace(/\b(?:(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+)?(?:(one|two|three|four|five|six|seven|eight|nine)\s+)?hundred\b/g,
    (_, a, b) => String(((WORD_NUMS[a] ?? 0) + (WORD_NUMS[b] ?? 0) || 1) * 100));
  t = t.replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-]+(one|two|three|four|five|six|seven|eight|nine))?\b/g,
    (_, a, b) => String(WORD_NUMS[a] + (WORD_NUMS[b] ?? 0)));
  t = t.replace(/(\d+)\s+point\s+(\d+)/g, '$1.$2');
  t = t.replace(/(\d+)\s+(\d+)\s*(thousand|k)\b/g, (_, a, b, u) => `${Number(a) * Number(b)} ${u}`);
  const re = /(\$|usd\s*|rs\.?\s*|₹\s*)?(\d+(?:\.\d+)?)\s*(k|grand|thousand|mn|mil|million|millions|m|lakh|lakhs|crore|crores)?\b(?:\s*(thousand))?/g;
  let best = null, m;
  while ((m = re.exec(t))) {
    let v = parseFloat(m[2]);
    if (m[3]) v *= MULT[m[3]];
    if (m[4]) v *= 1e3;
    const inr = /rs|₹/.test(m[1] || '') || /lakh|crore/.test(m[3] || '') || /\b(rupees|inr)\b/.test(t);
    if (inr) v /= INR_PER_USD;
    if (v >= 10000 && (!best || v > best)) best = v;
  }
  return best ? Math.round(best) : null;
}
