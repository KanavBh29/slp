import { IntentModel } from './model.js';
import { Chatbot } from './chatbot.js';
import { normalize } from './nlp.js';
import { WebSpeechEngine, WhisperEngine, speak } from './speech.js';

const $ = (id) => document.getElementById(id);
const els = {
  messages: $('messages'), interim: $('interim'), mic: $('mic'), text: $('text'), send: $('send'),
  form: $('composer'), status: $('status'), statusText: $('status-text'), engine: $('engine'),
  lang: $('lang'), tts: $('tts'), debug: $('debug'), hint: $('hint'), suggestions: $('suggestions'),
};
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pretty = s => s.replaceAll('_', ' ');

let bot;
const engines = {
  webspeech: new WebSpeechEngine(),
  whisper: new WhisperEngine((msg) => setStatus(msg, 'busy')),
};
const settings = loadSettings();

function loadSettings() {
  try { return JSON.parse(localStorage.getItem('hyperbot-settings')) || {}; } catch { return {}; }
}
function saveSettings() {
  try {
    localStorage.setItem('hyperbot-settings', JSON.stringify({
      engine: els.engine.value, lang: els.lang.value, tts: els.tts.checked, debug: els.debug.checked,
    }));
  } catch { /* storage unavailable */ }
}

function setStatus(text, state = '') {
  els.statusText.textContent = text;
  els.status.className = `status ${state}`;
}

function addMessage(role, html, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="meta">${meta}</div><div class="bubble">${html}</div>`;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function handle(text, source) {
  text = text.trim();
  if (!text) return;
  const sourceLabel = source.kind === 'voice'
    ? `🎤 Recognised speech${source.confidence ? ` · ${(source.confidence * 100).toFixed(0)}% ASR confidence` : ''}`
    : '⌨️ Typed';
  addMessage('user', esc(text), sourceLabel);

  const typing = addMessage('bot', '<span class="typing"><span></span><span></span><span></span></span>', 'HyperBot');
  setTimeout(() => {
    const t0 = performance.now();
    const r = bot.reply(text);
    const ms = performance.now() - t0;
    let html = r.html + (r.card ? `<div class="table-wrap">${r.card}</div>` : '');
    if (els.debug.checked) {
      const ents = [...r.entities.cars, ...r.entities.brands.filter(b => !r.entities.cars.some(c => c.startsWith(b)))];
      html += `<div class="analysis">
        <span class="chip intent ${r.confidence < 0.5 ? 'low' : ''}">intent: ${esc(pretty(r.intent))} · ${(r.confidence * 100).toFixed(1)}%</span>
        ${ents.map(e => `<span class="chip">🏎️ ${esc(e)}</span>`).join('')}
        ${r.entities.budget ? `<span class="chip">💰 $${r.entities.budget.toLocaleString()}</span>` : ''}
        ${r.note ? `<span class="chip">${esc(r.note)}</span>` : ''}
        <span class="chip">${ms.toFixed(0)} ms</span></div>`;
    }
    typing.querySelector('.bubble').innerHTML = html;
    typing.querySelector('.meta').textContent = 'HyperBot · BiLSTM response';
    els.messages.scrollTop = els.messages.scrollHeight;
    updatePipeline(text, source, r);
    if (els.tts.checked) speak(r.speech, els.lang.value);
  }, 250);
}

function updatePipeline(text, source, r) {
  $('p-asr').innerHTML = `${source.kind === 'voice' ? '🎤' : '⌨️'} “${esc(text)}”` +
    (source.alternatives?.length > 1 ? `<br><small>alternatives: ${source.alternatives.slice(1).map(a => `“${esc(a.transcript)}”`).join(', ')}</small>` : '');
  const spans = r.spans.map(s => `<code>${esc(s.text)}</code> → ${s.type === 'car' ? 'xcar' : 'xbrand'}${s.fuzzy ? ' <small>(fuzzy)</small>' : ''}`);
  if (r.entities.budget) spans.push(`budget = $${r.entities.budget.toLocaleString()}`);
  $('p-ent').innerHTML = spans.length ? spans.join('<br>') : 'no entities found';
  $('p-tok').innerHTML = r.tokens.map(t => r.oov.includes(t) ? `<s>${esc(t)}</s>` : `<code>${esc(t)}</code>`).join(' ') || '–';
  $('p-probs').innerHTML = r.top.map(t => `<div class="prob"><span>${esc(pretty(t.intent))}</span><div class="bar"><i style="width:${(t.p * 100).toFixed(1)}%"></i></div><em>${(t.p * 100).toFixed(1)}%</em></div>`).join('');
  $('p-resp').textContent = `${pretty(r.intent)}${r.entities.cars.length ? ` → ${r.entities.cars.join(', ')}` : ''}`;
}

// --- Voice input -----------------------------------------------------------

function currentEngine() {
  return els.engine.value === 'whisper' ? engines.whisper : engines.webspeech;
}

function chooseAlternative(alts) {
  // Prefer the recogniser's top guess, unless only a lower-ranked alternative
  // mentions a known car (exotic names are often misheard).
  const known = alts.find(a => bot.extractor.extract(normalize(a.transcript)).cars.length);
  return alts[0] && bot.extractor.extract(normalize(alts[0].transcript)).cars.length ? alts[0] : (known || alts[0]);
}

function toggleMic() {
  const engine = currentEngine();
  if (engine.active) { engine.stop(); return; }
  if (engine === engines.webspeech && !WebSpeechEngine.supported()) {
    addMessage('bot', 'This browser does not support the Web Speech API (e.g. Firefox). Switch the engine to <b>Whisper</b> in Settings, or use Chrome / Edge / Safari.', 'HyperBot');
    return;
  }
  window.speechSynthesis?.cancel();
  els.mic.classList.add('listening');
  els.interim.hidden = false;
  els.interim.textContent = 'Listening…';
  setStatus('Listening…', 'listening');
  engine.start(els.lang.value, {
    onInterim: (t) => { els.interim.textContent = t; },
    onFinal: (alts) => {
      const best = chooseAlternative(alts);
      handle(best.transcript, { kind: 'voice', confidence: best.confidence, alternatives: alts });
    },
    onError: (msg) => addMessage('bot', esc(msg), 'HyperBot · system'),
    onEnd: () => {
      els.mic.classList.remove('listening');
      els.interim.hidden = true;
      setStatus('Ready', 'ready');
    },
  });
}

// --- Boot ------------------------------------------------------------------

async function init() {
  if (settings.engine) els.engine.value = settings.engine;
  else if (!WebSpeechEngine.supported()) els.engine.value = 'whisper';
  if (settings.lang) els.lang.value = settings.lang;
  if (settings.tts === false) els.tts.checked = false;
  if (settings.debug === false) els.debug.checked = false;
  for (const el of [els.engine, els.lang, els.tts, els.debug]) el.addEventListener('change', saveSettings);

  try {
    const [model, db] = await Promise.all([
      IntentModel.load('model'),
      fetch('data/cars.json').then(r => r.json()),
    ]);
    bot = new Chatbot(model, db);
    $('n-cars').textContent = db.cars.length;
  } catch (err) {
    setStatus('Failed to load model', '');
    addMessage('bot', `Could not load the model: ${esc(err.message)}`, 'HyperBot · system');
    return;
  }

  els.mic.disabled = els.text.disabled = els.send.disabled = false;
  setStatus('Ready', 'ready');
  addMessage('bot', `Hi! I’m <b>HyperBot</b> 🏁 Tap the <b>mic</b> and ask me about any of ${bot.db.cars.length} supercars and hypercars: details, specs, price or where to buy.`, 'HyperBot');
  if (!WebSpeechEngine.supported()) {
    els.hint.textContent = '🎙️ Your browser has no built-in speech recognition, so the in-browser Whisper engine will be used (tap mic to start, tap again to stop).';
  }

  els.mic.addEventListener('click', toggleMic);
  els.form.addEventListener('submit', (e) => {
    e.preventDefault();
    handle(els.text.value, { kind: 'text' });
    els.text.value = '';
  });
  els.suggestions.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') handle(e.target.textContent, { kind: 'text' });
  });
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.ctrlKey) { e.preventDefault(); toggleMic(); }
  });
}

init();
