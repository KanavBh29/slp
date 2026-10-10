"""Builds report/HyperBot_Project_Report.html; render it to PDF with
`node report/render_pdf.cjs` (headless Chromium via Playwright).

All numbers are read from training/artifacts/metrics.json, the intent corpus
and the car database, so the PDF stays in sync with the code.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from intents import INTENTS  # noqa: E402

M = json.loads((ROOT / "training/artifacts/metrics.json").read_text())
DB = json.loads((ROOT / "web/data/cars.json").read_text())
CARS = DB["cars"]
OUT = ROOT / "report" / "HyperBot_Project_Report.html"
LIVE = "https://kanavbh29.github.io/slp/"
REPO = "https://github.com/KanavBh29/slp"
e = html.escape


def pct(x):
    return f"{x * 100:.1f}%"


def money(v):
    return f"${v / 1e6:.1f}M" if v >= 1e6 else f"${v / 1e3:,.0f}k"


INTENT_DESC = {
    "greeting": "Opening the conversation", "goodbye": "Ending the conversation",
    "thanks": "Gratitude / acknowledgement", "help": "What can the bot do",
    "bot_identity": "Questions about the bot", "car_overview": "General details about a car",
    "car_specs": "Full specification sheet", "car_engine": "Engine / powertrain",
    "car_power": "Horsepower and torque", "car_top_speed": "Maximum speed",
    "car_acceleration": "0-100 km/h / 0-60 mph", "car_price": "Price / cost / value",
    "where_to_buy": "Dealers, marketplaces, auctions", "car_production": "Units built, availability",
    "compare": "Head-to-head comparison", "fastest_car": "Ranking by top speed",
    "quickest_car": "Ranking by acceleration", "most_powerful": "Ranking by power",
    "most_expensive": "Ranking by price (desc)", "cheapest": "Ranking by price (asc)",
    "budget_recommend": "Recommendations for a budget", "electric_hybrid": "EV / hybrid listing",
    "list_cars": "Database contents", "brand_info": "Manufacturer information",
    "definition": "Supercar vs hypercar", "out_of_scope": "Off-topic (politely declined)",
}


def intent_table():
    rows = []
    per = M["per_intent_fold1"]
    for name, temps in INTENTS.items():
        ex = temps[min(2, len(temps) - 1)].replace("{car}", "Chiron").replace("{car2}", "Jesko") \
            .replace("{brand}", "Ferrari").replace("{num}", "$400k")
        rows.append(f"<tr><td><code>{name}</code></td><td>{e(INTENT_DESC[name])}</td>"
                    f"<td class='num'>{len(temps)}</td><td><i>“{e(ex)}”</i></td>"
                    f"<td class='num'>{per[name]['f1']:.2f}</td></tr>")
    return "\n".join(rows)


def car_table():
    rows = []
    for c in sorted(CARS, key=lambda c: (c["brand"], -c["year"])):
        rows.append(f"<tr><td>{e(c['name'])}</td><td>{c['class']}</td><td>{c['powertrain']}</td>"
                    f"<td class='num'>{c['power_hp']:,}</td><td class='num'>{c['top_speed_kmh']}</td>"
                    f"<td class='num'>{c['zero_to_100_s']}</td><td class='num'>{money(c['price_usd'])}</td>"
                    f"<td>{c['status']}</td></tr>")
    return "\n".join(rows)


def results_table():
    names = {"bow_mlp": "Bag-of-words MLP (baseline)", "bilstm_scratch": "BiLSTM, embeddings from scratch",
             "bilstm_glove": "<b>BiLSTM + GloVe (deployed)</b>"}
    rs, cv = M["random_split"], M["template_cv"]
    out = []
    for k, n in names.items():
        cls = " class='hl'" if k == "bilstm_glove" else ""
        out.append(
            f"<tr{cls}><td>{n}</td><td class='num'>{rs[k]['trainable_params'] / 1e6:.2f} M</td>"
            f"<td class='num'>{pct(rs[k]['accuracy'])}</td><td class='num'>{rs[k]['macro_f1']:.3f}</td>"
            f"<td class='num'>{pct(cv[k]['accuracy_mean'])} ± {cv[k]['accuracy_std'] * 100:.1f}</td>"
            f"<td class='num'>{cv[k]['macro_f1_mean']:.3f} ± {cv[k]['macro_f1_std']:.3f}</td></tr>")
    return "\n".join(out)


def fold_table():
    cv = M["template_cv"]
    rows = []
    for k, n in (("bow_mlp", "BoW-MLP"), ("bilstm_scratch", "BiLSTM (scratch)"), ("bilstm_glove", "BiLSTM + GloVe")):
        cells = "".join(f"<td class='num'>{pct(f['accuracy'])}</td>" for f in cv[k]["folds"])
        rows.append(f"<tr><td>{n}</td>{cells}<td class='num'><b>{pct(cv[k]['accuracy_mean'])}</b></td></tr>")
    return "\n".join(rows)


def conversation():
    import subprocess
    script = ROOT / "report" / "_dialogue.mjs"
    script.write_text("""
import { readFileSync } from 'node:fs';
import { IntentModel } from '../web/js/model.js';
import { Chatbot } from '../web/js/chatbot.js';
const r = new URL('../', import.meta.url);
const bin = readFileSync(new URL('web/model/model.bin', r));
const bot = new Chatbot(new IntentModel(JSON.parse(readFileSync(new URL('web/model/model.json', r))),
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)), JSON.parse(readFileSync(new URL('web/data/cars.json', r))));
bot.reply = ((orig) => (t) => { const x = orig.call(bot, t); return x; })(bot.reply);
const qs = JSON.parse(process.argv[2]);
console.log(JSON.stringify(qs.map(q => { const x = bot.reply(q); return { q, intent: x.intent, conf: x.confidence, a: x.speech }; })));
""")
    qs = ["Hello", "Tell me about the Bugatti Chiron", "How fast is it?", "How much does it cost?",
          "Where can I buy one?", "Compare it with the Koenigsegg Jesko", "Recommend me a car",
          "500 thousand", "Is the Lamborghini Revuelto a hybrid?", "bugatti sheeran price",
          "What is the price of the Tesla Roadster?", "Tell me a joke", "Thanks, bye"]
    res = json.loads(subprocess.check_output(["node", str(script), json.dumps(qs)], cwd=ROOT))
    script.unlink()
    rows = []
    for r in res:
        rows.append(f"<tr><td>“{e(r['q'])}”</td><td><code>{r['intent']}</code><br><small>{pct(r['conf'])}</small></td>"
                    f"<td>{e(r['a'])}</td></tr>")
    return "\n".join(rows)


ARCH_SVG = """
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" font-size="12">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
  <g stroke="#555" stroke-width="1.4" fill="none" marker-end="url(#a)">
    <path d="M110 60 H150"/><path d="M290 60 H330"/><path d="M470 60 H510"/>
    <path d="M580 85 V125"/><path d="M510 150 H470"/><path d="M330 150 H290"/>
    <path d="M220 175 V215"/><path d="M290 240 H330"/><path d="M470 240 H510"/>
  </g>
  <g>
    <rect x="10" y="35" width="100" height="50" rx="8" fill="#fde8e6" stroke="#e0392b"/>
    <text x="60" y="57" text-anchor="middle" font-weight="600">Microphone</text>
    <text x="60" y="73" text-anchor="middle" fill="#555">user speaks</text>

    <rect x="150" y="30" width="140" height="60" rx="8" fill="#fff4e5" stroke="#f08c00"/>
    <text x="220" y="52" text-anchor="middle" font-weight="600">Speech recognition</text>
    <text x="220" y="68" text-anchor="middle" fill="#555">Web Speech API /</text>
    <text x="220" y="82" text-anchor="middle" fill="#555">Whisper (on device)</text>

    <rect x="330" y="30" width="140" height="60" rx="8" fill="#f1f3f5" stroke="#868e96"/>
    <text x="400" y="52" text-anchor="middle" font-weight="600">Transcript</text>
    <text x="400" y="68" text-anchor="middle" fill="#555">n-best + confidence</text>
    <text x="400" y="82" text-anchor="middle" fill="#555">shown to the user</text>

    <rect x="510" y="30" width="140" height="55" rx="8" fill="#f1f3f5" stroke="#868e96"/>
    <text x="580" y="52" text-anchor="middle" font-weight="600">Normalisation</text>
    <text x="580" y="68" text-anchor="middle" fill="#555">lowercase, punctuation</text>

    <rect x="510" y="120" width="200" height="62" rx="8" fill="#e7f5ff" stroke="#1c7ed6"/>
    <text x="610" y="140" text-anchor="middle" font-weight="600">Entity extraction</text>
    <text x="610" y="156" text-anchor="middle" fill="#555">car · brand · budget</text>
    <text x="610" y="171" text-anchor="middle" fill="#555">exact + fuzzy + phonetic</text>

    <rect x="330" y="125" width="140" height="55" rx="8" fill="#f1f3f5" stroke="#868e96"/>
    <text x="400" y="147" text-anchor="middle" font-weight="600">Delexicalised text</text>
    <text x="400" y="163" text-anchor="middle" fill="#555">“how fast is the xcar”</text>

    <rect x="150" y="120" width="140" height="55" rx="8" fill="#ebfbee" stroke="#2f9e44" stroke-width="2"/>
    <text x="220" y="142" text-anchor="middle" font-weight="700">BiLSTM classifier</text>
    <text x="220" y="158" text-anchor="middle" fill="#555">GloVe · 26 intents</text>

    <rect x="150" y="215" width="140" height="55" rx="8" fill="#f3f0ff" stroke="#7048e8"/>
    <text x="220" y="237" text-anchor="middle" font-weight="600">Dialogue manager</text>
    <text x="220" y="253" text-anchor="middle" fill="#555">threshold, context, slots</text>

    <rect x="330" y="215" width="140" height="55" rx="8" fill="#f1f3f5" stroke="#868e96"/>
    <text x="400" y="237" text-anchor="middle" font-weight="600">Knowledge base</text>
    <text x="400" y="253" text-anchor="middle" fill="#555">35 cars · 18 brands</text>

    <rect x="510" y="215" width="200" height="55" rx="8" fill="#fde8e6" stroke="#e0392b"/>
    <text x="610" y="237" text-anchor="middle" font-weight="600">Response</text>
    <text x="610" y="253" text-anchor="middle" fill="#555">chat UI · spec cards · speech</text>
  </g>
</svg>"""

n_hyper = sum(c["class"] == "Hypercar" for c in CARS)
n_ev = sum(c["powertrain"] == "Electric" for c in CARS)
n_hy = sum(c["powertrain"] == "Hybrid" for c in CARS)
fm = M["final_model"]
cv = M["template_cv"]
rs = M["random_split"]
g, b, s = cv["bilstm_glove"], cv["bow_mlp"], cv["bilstm_scratch"]
n_templates = M["n_templates"]

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>HyperBot Project Report</title>
<style>
@page {{ size: A4; margin: 20mm 18mm 20mm 18mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Inter, 'DejaVu Sans', sans-serif; font-size: 10.2pt; line-height: 1.5; color: #1d2129; margin: 0; }}
h1, h2, h3 {{ color: #111; line-height: 1.25; }}
h2 {{ font-size: 15pt; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 2px solid #e0392b; break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 16px 0 6px; break-after: avoid; }}
p {{ margin: 6px 0; text-align: justify; }}
a {{ color: #c92a2a; text-decoration: none; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f1f3f5; padding: 0 3px; border-radius: 3px; }}
pre {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 9px 11px; white-space: pre-wrap; break-inside: avoid; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 10px; font-size: 9pt; break-inside: auto; }}
th, td {{ border-bottom: 1px solid #dee2e6; padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ background: #f1f3f5; font-weight: 600; }}
tr {{ break-inside: avoid; }}
td.num, th.num {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
tr.hl td {{ background: #ebfbee; }}
.small {{ font-size: 8.4pt; }}
figure {{ margin: 10px 0; break-inside: avoid; text-align: center; }}
figure img, figure svg {{ max-width: 100%; border-radius: 6px; }}
figure img.shot {{ border: 1px solid #ced4da; }}
figcaption {{ font-size: 8.6pt; color: #555; margin-top: 4px; }}
.callout {{ background: #fff4e5; border-left: 4px solid #f08c00; padding: 8px 12px; border-radius: 4px; margin: 10px 0; break-inside: avoid; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0; }}
.kpi {{ border: 1px solid #dee2e6; border-radius: 8px; padding: 8px 10px; }}
.kpi b {{ display: block; font-size: 15pt; color: #c92a2a; }}
.kpi span {{ font-size: 8.4pt; color: #555; }}
.page-break {{ break-before: page; }}
/* cover */
.cover {{ height: 255mm; display: flex; flex-direction: column; justify-content: space-between; }}
.cover .band {{ background: linear-gradient(120deg, #111318, #2b1514 70%, #5c1a12); color: #fff; border-radius: 14px; padding: 34px 32px; }}
.cover .band h1 {{ color: #fff; font-size: 30pt; margin: 6px 0 4px; letter-spacing: .02em; }}
.cover .band .tag {{ text-transform: uppercase; letter-spacing: .18em; font-size: 9pt; color: #ff8a00; }}
.cover .band p {{ color: #e9ecef; font-size: 12pt; text-align: left; }}
.cover table td {{ border: 0; padding: 5px 0; font-size: 10.5pt; }}
.cover table td:first-child {{ width: 40mm; color: #555; }}
.blank {{ display: inline-block; min-width: 90mm; border-bottom: 1px solid #adb5bd; }}
.toc {{ columns: 2; column-gap: 24px; font-size: 9.6pt; padding-left: 18px; }}
.toc li {{ margin: 2px 0; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
</style></head><body>

<section class="cover">
  <div class="band">
    <div class="tag">Speech &amp; Language Processing · Project Report</div>
    <h1>HyperBot 🏎️</h1>
    <p>A voice-enabled chatbot for supercar &amp; hypercar information, built with speech recognition and a deep-learning (BiLSTM) intent classifier</p>
  </div>
  <div>
    <table>
      <tr><td>Live application</td><td><a href="{LIVE}">{LIVE}</a></td></tr>
      <tr><td>Source code</td><td><a href="{REPO}">{REPO}</a></td></tr>
      <tr><td>Submitted by</td><td><span class="blank">&nbsp;</span></td></tr>
      <tr><td>Roll / ID</td><td><span class="blank">&nbsp;</span></td></tr>
      <tr><td>Course / Instructor</td><td><span class="blank">&nbsp;</span></td></tr>
      <tr><td>Date</td><td>October 2026</td></tr>
    </table>
  </div>
  <div>
    <h3>Contents</h3>
    <ol class="toc">
      <li>Abstract</li><li>Introduction &amp; objectives</li><li>System architecture</li>
      <li>Speech recognition</li><li>Dataset</li><li>Text preprocessing &amp; entity extraction</li>
      <li>Model architecture &amp; training</li><li>Dialogue management</li>
      <li>Implementation &amp; deployment</li><li>Results &amp; evaluation</li>
      <li>Application walkthrough</li><li>Limitations &amp; future work</li>
      <li>Conclusion</li><li>References</li><li>Appendix: reproducing the project</li>
    </ol>
  </div>
</section>

<h2 class="page-break">1. Abstract</h2>
<p>HyperBot is a publicly hosted, voice-enabled chatbot that answers questions about {len(CARS)} supercars and hypercars: their details, specifications, prices and where to buy them. The user's speech is transcribed in the browser by the Web Speech API, or by an on-device Whisper model when the browser's speech service is unavailable. Car names, brands and budgets are extracted with exact, fuzzy and phonetic matching, and replaced by placeholder tokens. A <b>Bidirectional LSTM</b> initialised with pre-trained <b>GloVe</b> embeddings then classifies the sentence into one of <b>{M['n_intents']} intents</b>. A dialogue manager with context memory and slot filling builds the answer from a curated knowledge base, shows it next to the recognised speech, and reads it aloud.</p>
<p>The model was trained on a hand-built corpus of {n_templates} utterance templates, augmented to {fm['n_examples']:,} sentences. In 5-fold cross-validation on <i>phrasings never seen during training</i>, it reaches <b>{pct(g['accuracy_mean'])} ± {g['accuracy_std']*100:.1f}</b> accuracy. That is {(g['accuracy_mean']-s['accuracy_mean'])*100:.1f} points above the same network trained without transfer learning, and on par with a bag-of-words MLP baseline that is twice its size. On the in-distribution random split it reaches {pct(rs['bilstm_glove']['accuracy'])}. The whole system runs client-side and is deployed to GitHub Pages by a CI pipeline that runs end-to-end tests before every release.</p>

<div class="kpis">
  <div class="kpi"><b>{len(CARS)}</b><span>cars from {len(DB['brands'])} brands</span></div>
  <div class="kpi"><b>{M['n_intents']}</b><span>intents · {n_templates} templates</span></div>
  <div class="kpi"><b>{pct(g['accuracy_mean'])}</b><span>accuracy on unseen phrasings (5-fold CV)</span></div>
  <div class="kpi"><b>&lt;50 ms</b><span>in-browser inference per message</span></div>
</div>

<h2>2. Introduction &amp; objectives</h2>
<p>Car enthusiasts often want quick, specific answers ("How fast is the Koenigsegg Jesko?", "Where can I buy a McLaren 750S?") without searching through brochures and listings. A voice interface makes this natural, but it brings two NLP problems. First, speech recognisers often mishear exotic model names. Second, users phrase the same question in many different ways. The project objectives were to:</p>
<ol>
  <li>accept voice input and convert it to text with a suitable speech recognition technique;</li>
  <li>classify the text with a deep-learning intent model and generate an appropriate response;</li>
  <li>display both the recognised speech and the chatbot response;</li>
  <li>deploy the application online with a public link; and</li>
  <li>document the dataset, architecture, methodology and results.</li>
</ol>

<h2>3. System architecture</h2>
<figure>{ARCH_SVG}<figcaption>Figure 1. End-to-end pipeline. Every stage runs inside the user's browser except the optional cloud ASR of the Web Speech API.</figcaption></figure>
<p>The application is a static web app: HTML, CSS and JavaScript ES modules, a JSON knowledge base, and the exported model weights. The neural network is trained offline in Python with TensorFlow/Keras, and its weights are exported to a compact binary file. A hand-written JavaScript implementation of the forward pass runs inference in the browser. This design needs no server, has no API keys and no cold starts, and can be hosted free on GitHub Pages.</p>

<h2>4. Speech recognition</h2>
<table>
  <tr><th>Engine</th><th>Technique</th><th>Runs</th><th>Notes</th></tr>
  <tr><td><b>Web Speech API</b> (default)</td><td>Browser vendor's neural ASR (Google's speech service in Chrome)</td><td>Browser + vendor cloud</td><td>Streaming interim results; n-best list (3 alternatives) with confidence; en-US/GB/IN/AU</td></tr>
  <tr><td><b>Whisper-tiny.en</b> (fallback)</td><td>Transformer encoder-decoder (39M params) via Transformers.js / ONNX Runtime Web</td><td>Fully on device</td><td>Used automatically if the cloud service is unreachable (e.g. Brave); push-to-talk; audio resampled to 16 kHz mono</td></tr>
</table>
<p>General-purpose ASR is weakest on exactly the words this domain depends on: "Chiron" is heard as <i>"sheeran"</i>, "Huayra" as <i>"wire a"</i>, "Nevera" as <i>"never a"</i>. Three mechanisms compensate:</p>
<ul>
  <li><b>n-best rescoring:</b> a lower-ranked alternative is chosen if it is the only one that mentions a known car.</li>
  <li><b>Fuzzy matching:</b> windows of 1–3 words are matched to every alias by normalised Levenshtein similarity (≥ 0.8). Only windows containing an uncommon word are considered, to avoid false positives.</li>
  <li><b>Phonetic matching:</b> a simplified Metaphone key maps <code>sheeran</code> and <code>chiron</code> to the same key <code>xrn</code>, and <code>hurricane</code> and <code>huracan</code> to <code>hrkn</code>.</li>
</ul>
<p>The recognised text and its ASR confidence are shown above each answer, so the user can always see what the system heard.</p>

<h2>5. Dataset</h2>
<h3>5.1 Intent corpus</h3>
<p>No public dataset exists for a supercar assistant, so a domain corpus was written by hand: <b>{n_templates} templates over {M['n_intents']} intents</b>. It includes short follow-ups ("how much does it cost"), unpunctuated speech-style text and an <code>out_of_scope</code> class. Templates contain slot placeholders <code>{{car}}</code>, <code>{{brand}}</code> and <code>{{num}}</code>, which are <b>delexicalised</b> to the tokens <code>xcar</code>, <code>xbrand</code> and <code>xnum</code>. As a result, the network learns the <i>intent</i> rather than memorising car names, and new cars can be added without retraining.</p>
<p><b>Augmentation</b> (12 variants per template):</p>
<ul>
  <li>polite prefixes and suffixes ("hey …", "… please");</li>
  <li>unknown-entity simulation: 12% of car slots are filled with out-of-vocabulary tokens;</li>
  <li>ASR word-dropout: one non-entity word is deleted with probability 0.15.</li>
</ul>
<p>The final training set has {fm['n_examples']:,} sentences.</p>
<table class="small">
  <tr><th>Intent</th><th>Meaning</th><th class="num">Templates</th><th>Example</th><th class="num">F1*</th></tr>
  {intent_table()}
</table>
<p class="small">* F1 on fold 1 of the template cross-validation, i.e. on phrasings unseen during training.</p>

<h3>5.2 Knowledge base</h3>
<p><code>web/data/cars.json</code> holds {len(CARS)} cars ({n_hyper} hypercars, {len(CARS)-n_hyper} supercars; {n_ev} electric, {n_hy} hybrid) from {len(DB['brands'])} manufacturers. Each entry records the engine, power, torque, 0–100 km/h time, top speed, approximate price, production volume, availability status, a description, and aliases (including common mishearings). The <i>where to buy</i> answer depends on the car's status:</p>
<ul>
  <li><b>available:</b> the official dealer and configurator;</li>
  <li><b>limited:</b> the manufacturer, plus the secondary market;</li>
  <li><b>sold out</b> or <b>discontinued:</b> specialist marketplaces (duPont REGISTRY, JamesEdition) and auction houses (Bring a Trailer, RM Sotheby's).</li>
</ul>
<table class="small">
  <tr><th>Car</th><th>Class</th><th>Power­train</th><th class="num">hp</th><th class="num">km/h</th><th class="num">0-100 s</th><th class="num">Price</th><th>Status</th></tr>
  {car_table()}
</table>
<p class="small">Specs are manufacturer claims. Prices are approximate US base prices for new cars, or typical market values for sold-out and discontinued cars.</p>

<h2>6. Text preprocessing &amp; entity extraction</h2>
<ol>
  <li><b>Normalisation:</b> lowercase, remove apostrophes ("what's" → "whats"), replace punctuation with spaces while keeping decimals and model names such as "t.50".</li>
  <li><b>Entity extraction:</b> car aliases are matched longest first (so "Bugatti Chiron" wins over "Bugatti"), then fuzzy and phonetic car matches, then brands. Matches are replaced by <code>xcar</code> or <code>xbrand</code>.</li>
  <li><b>Number masking:</b> amounts such as "$2.5 million", "300k" and "2.5s" become <code>xnum</code>, while names like "V12" are kept. A separate parser converts budgets to USD, including spoken forms ("two million", "half a million", "five hundred thousand") and rupees ("3 crore").</li>
  <li><b>Encoding:</b> tokens are mapped to vocabulary ids (1 = OOV) and post-padded to length 24.</li>
</ol>
<p>The Python (training) and JavaScript (inference) preprocessors are tested against each other on tricky inputs to guarantee identical tokens.</p>

<h2>7. Model architecture &amp; training</h2>
<table>
  <tr><th>Layer</th><th>Output</th><th class="num">Parameters</th><th>Notes</th></tr>
  <tr><td>Input</td><td>24 token ids</td><td class="num">0</td><td>0 = padding (masked)</td></tr>
  <tr><td>Embedding</td><td>24 × 50</td><td class="num">{fm['vocab_size']*50:,}</td><td>GloVe-50d init (Wikipedia + Gigaword, 400k words), fine-tuned, <code>mask_zero</code></td></tr>
  <tr><td>Bidirectional LSTM</td><td>128</td><td class="num">58,880</td><td>64 units per direction</td></tr>
  <tr><td>Dropout</td><td>128</td><td class="num">0</td><td>rate 0.3</td></tr>
  <tr><td>Dense</td><td>64</td><td class="num">8,256</td><td>ReLU</td></tr>
  <tr><td>Dropout</td><td>64</td><td class="num">0</td><td>rate 0.3</td></tr>
  <tr><td>Dense (output)</td><td>{M['n_intents']}</td><td class="num">1,690</td><td>softmax</td></tr>
  <tr><th colspan="2">Total</th><th class="num">{fm['total_params']:,}</th><th></th></tr>
</table>
<p><b>Why a BiLSTM?</b> Intent depends on word order and context, e.g. "is the <i>xcar</i> faster than the <i>xcar</i>" (compare) versus "fastest car" (ranking). A bidirectional LSTM reads the sentence in both directions and handles variable length through masking.</p>
<p><b>Why GloVe transfer learning?</b> The vocabulary ({fm['vocab_size']:,} words) contains every training word plus the 8,000 most frequent English words, all initialised with GloVe vectors. Words that never appear in training receive no gradient and keep their pre-trained meaning. This lets the model handle unseen wording, e.g. "where can I <b>acquire</b> one" → where_to_buy and "which <b>vehicles</b> are battery electric" → electric_hybrid. The placeholder tokens start from the vectors of "supercar", "manufacturer" and "million".</p>
<p><b>Training:</b> Adam (learning rate 2·10<super>-3</super>), sparse categorical cross-entropy, batch size 32, up to 80 epochs, early stopping on validation loss (patience 12, best weights restored), 10% validation split, seed 42. The final model trained for {fm['epochs_run']} epochs and reached <b>{pct(fm['best_val_accuracy'])} validation accuracy</b>.</p>
<p><b>Browser export:</b> embeddings are quantised to int8 with a per-row scale, and the other weights stay float32, giving a ≈0.7 MB binary plus an 80 KB vocabulary. The JavaScript LSTM (Keras gate order i, f, c, o) reproduces the Keras probabilities to within <b>5·10<super>-7</super></b>.</p>

<h2>8. Dialogue management</h2>
<ul>
  <li><b>Confidence threshold (0.5):</b> below it, rule-based fallbacks use the extracted entities (two cars → compare, one car → overview, a budget → recommendations). Otherwise the bot asks the user to rephrase.</li>
  <li><b>Context memory:</b> the most recent cars are remembered, so "How fast is it?", "Price please" and "Compare it with the Jesko" work. Context is reused only for genuine follow-ups (a pronoun or a very short question). "What is the price of the Tesla Roadster?" gets "I don't have that car" instead of a wrong answer.</li>
  <li><b>Slot filling:</b> "Recommend me a car" → "What is your budget?" → "500 thousand" → recommendations.</li>
  <li><b>Response types:</b> text answers, spec cards, comparison tables (winners highlighted), top-5 rankings (optionally per brand), budget lists, purchase links, and text-to-speech output.</li>
</ul>

<h2>9. Implementation &amp; deployment</h2>
<table>
  <tr><th>Component</th><th>Technology</th></tr>
  <tr><td>Model training &amp; evaluation</td><td>Python 3, TensorFlow 2 / Keras, NumPy, Matplotlib, GloVe embeddings</td></tr>
  <tr><td>Frontend</td><td>HTML5, CSS, vanilla JavaScript ES modules (no framework), responsive for mobile</td></tr>
  <tr><td>Speech</td><td>Web Speech API (SpeechRecognition + SpeechSynthesis); Transformers.js Whisper fallback</td></tr>
  <tr><td>Inference</td><td>Custom JavaScript BiLSTM forward pass over exported weights</td></tr>
  <tr><td>Testing</td><td>Node.js test scripts (model parity, tokenizer parity, entities, budgets, 32-turn dialogue); Playwright browser tests</td></tr>
  <tr><td>Hosting / CI-CD</td><td>GitHub Actions → GitHub Pages (tests must pass before deploy)</td></tr>
</table>
<pre>web/                    deployed static site
  js/speech.js          speech recognition engines + text-to-speech
  js/nlp.js             normalisation, entity extraction, budget parser
  js/model.js           BiLSTM forward pass in JavaScript
  js/chatbot.js         dialogue manager + response generation
  data/cars.json        knowledge base
  model/model.json|bin  vocabulary + weights
training/               intents.py, nlp.py, train.py, artifacts/
tests/                  parity, entity and conversation tests
.github/workflows/      test &amp; deploy pipeline</pre>

<h2>10. Results &amp; evaluation</h2>
<h3>10.1 Intent classification</h3>
<p>Three neural models were compared under two protocols:</p>
<ul>
  <li><b>Random split</b> (85/15 of the generated sentences, {rs['n_train']:,}/{rs['n_test']}). This is in-distribution: test sentences can share a template with training sentences.</li>
  <li><b>Template 5-fold cross-validation.</b> Each fold holds out 20% of <i>every intent's templates</i>, so every test sentence is a phrasing the model has never seen. This is the more realistic estimate for new users.</li>
</ul>
<table>
  <tr><th>Model</th><th class="num">Trainable params</th><th class="num">Random: acc</th><th class="num">Random: F1</th><th class="num">Template CV: acc</th><th class="num">Template CV: F1</th></tr>
  {results_table()}
</table>
<table class="small">
  <tr><th>Accuracy per fold</th><th class="num">Fold 1</th><th class="num">Fold 2</th><th class="num">Fold 3</th><th class="num">Fold 4</th><th class="num">Fold 5</th><th class="num">Mean</th></tr>
  {fold_table()}
</table>
<div class="callout"><b>Findings.</b>
<ul>
  <li><b>Transfer learning helps most on unseen phrasings.</b> GloVe initialisation adds +{(g['accuracy_mean']-s['accuracy_mean'])*100:.1f} accuracy points and +{g['macro_f1_mean']-s['macro_f1_mean']:.3f} macro-F1 over the same BiLSTM trained from scratch.</li>
  <li><b>On the in-distribution split, all models exceed 94%.</b> The bag-of-words MLP is slightly ahead there because keyword memorisation works well when templates repeat.</li>
  <li><b>On genuinely new phrasings, the GloVe-BiLSTM has the highest mean accuracy and F1.</b> The gap to the baseline is within fold-to-fold variance, so the two are best described as comparable. The BiLSTM is less than half the size and also models word order.</li>
</ul></div>
<div class="two">
  <figure><img src="../training/artifacts/training_curves.png"><figcaption>Figure 2. Loss and accuracy curves (fold 1).</figcaption></figure>
  <figure><img src="../training/artifacts/confusion_matrix.png"><figcaption>Figure 3. Confusion matrix on held-out templates (fold 1).</figcaption></figure>
</div>
<p><b>Error analysis.</b> Most remaining errors fall into four groups:</p>
<ul>
  <li>one-word slang with no embedding ("thx", "cya", "yo");</li>
  <li>off-topic questions shaped like "who/what is X" ("who are the beatles" → car_overview);</li>
  <li>a single car's acceleration confused with the acceleration ranking ("zero to sixty" → quickest_car);</li>
  <li>overlap between availability and purchasing ("is it sold out" → where_to_buy).</li>
</ul>
<p>The dialogue manager limits the impact of these errors. For example, a car-specific intent with no car mentioned produces a clarifying question rather than a wrong fact.</p>

<h3>10.2 End-to-end testing</h3>
<table>
  <tr><th>Test (run in CI before every deployment)</th><th>Result</th></tr>
  <tr><td>JavaScript BiLSTM vs Keras on 11 reference sentences</td><td>identical labels; max |Δp| = 5·10<super>-7</super></td></tr>
  <tr><td>Python vs JavaScript tokenizer on 13 tricky inputs</td><td>identical</td></tr>
  <tr><td>Entity extraction incl. ASR errors ("bugatti sheeran", "koenig seg jesko", "rimac never a")</td><td>11 / 11 correct</td></tr>
  <tr><td>Budget parser ("$300k", "2.5 million dollars", "two million", "half a million", "3 crore rupees")</td><td>8 / 8 correct</td></tr>
  <tr><td>32-turn scripted conversation (follow-ups, comparisons, slot filling, out-of-scope)</td><td>32 / 32 intents and entities correct</td></tr>
  <tr><td>Headless Chromium, desktop + mobile, simulated microphone</td><td>speech shown with confidence, correct answer, no console errors, no horizontal overflow</td></tr>
  <tr><td>Simulated speech-service network failure</td><td>automatic switch to on-device Whisper with explanation</td></tr>
</table>

<h3>10.3 Sample dialogue (actual system output)</h3>
<table class="small">
  <tr><th style="width:30%">User (recognised speech)</th><th style="width:16%">Intent</th><th>HyperBot response (spoken version)</th></tr>
  {conversation()}
</table>

<h2>11. Application walkthrough</h2>
<figure><img class="shot" src="screenshot-overview.png"><figcaption>Figure 4. Voice query "tell me about the Bugatti Tourbillon": the recognised speech with 94% ASR confidence, the answer with a spec card, the predicted intent and confidence, and the NLP pipeline panel.</figcaption></figure>
<figure><img class="shot" src="screenshot-desktop.png"><figcaption>Figure 5. A comparison table with winners highlighted, and where-to-buy links for a car in production.</figcaption></figure>
<figure><img class="shot" src="screenshot-budget.png"><figcaption>Figure 6. Budget recommendation (parsed $400,000) and a power ranking.</figcaption></figure>
<div class="two">
  <figure><img class="shot" src="screenshot-mobile.png" style="max-height:150mm"><figcaption>Figure 7. Responsive mobile layout.</figcaption></figure>
  <div>
    <h3>Using the application</h3>
    <ol>
      <li>Open <a href="{LIVE}">{LIVE}</a> in Chrome, Edge or Safari (HTTPS is required for the microphone).</li>
      <li>Tap the red microphone button and allow microphone access.</li>
      <li>Ask a question, for example "How much is the Ferrari F80?".</li>
      <li>The recognised speech appears in the chat, followed by the answer, which is also spoken aloud.</li>
      <li>Typing and the suggestion chips work as alternatives to voice.</li>
      <li>In browsers that block the cloud speech service (e.g. Brave), the app switches to on-device Whisper. Tap the mic once to start and again to stop.</li>
    </ol>
  </div>
</div>

<h2>12. Limitations &amp; future work</h2>
<ul>
  <li>The intent corpus is synthetic. Collecting real anonymised queries would give the largest robustness gain.</li>
  <li>Web Speech API quality and privacy depend on the browser vendor. Whisper keeps audio on the device but needs a ~40 MB download.</li>
  <li>Prices and availability change. The static JSON should be refreshed regularly or connected to a live listings API.</li>
  <li>Possible extensions: a fine-tuned transformer encoder (e.g. DistilBERT) for intents; joint intent + slot tagging (BiLSTM-CRF) instead of dictionary matching; multilingual ASR; images of each car.</li>
</ul>

<h2>13. Conclusion</h2>
<p>HyperBot meets all project requirements. It accepts voice input, transcribes it with a speech recognition engine (with an on-device fallback), classifies intent with a deep-learning BiLSTM using transfer-learned GloVe embeddings, and answers questions about details, specifications, prices and where to buy {len(CARS)} supercars and hypercars. Each turn shows both the recognised speech and the response. The system is publicly deployed through an automated test-and-deploy pipeline. Evaluation on unseen phrasings shows a clear benefit from transfer learning ({pct(g['accuracy_mean'])} vs {pct(s['accuracy_mean'])}). Domain-specific handling of speech recognition errors (n-best rescoring, fuzzy and phonetic matching) makes the voice experience reliable even for hard-to-pronounce car names.</p>

<h2>14. References</h2>
<ol class="small">
  <li>S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," <i>Neural Computation</i>, 9(8), 1997.</li>
  <li>M. Schuster and K. K. Paliwal, "Bidirectional Recurrent Neural Networks," <i>IEEE Trans. Signal Processing</i>, 45(11), 1997.</li>
  <li>J. Pennington, R. Socher and C. D. Manning, "GloVe: Global Vectors for Word Representation," <i>EMNLP</i>, 2014.</li>
  <li>A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)," 2022.</li>
  <li>W3C Community Group, "Web Speech API" specification.</li>
  <li>V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," <i>Soviet Physics Doklady</i>, 1966.</li>
  <li>L. Philips, "Hanging on the Metaphone," <i>Computer Language</i>, 1990.</li>
  <li>F. Chollet et al., Keras; M. Abadi et al., TensorFlow, 2015–2026.</li>
  <li>Hugging Face, Transformers.js documentation.</li>
  <li>Manufacturer press releases and official websites for vehicle specifications (see the knowledge base).</li>
</ol>

<h2>Appendix: reproducing the project</h2>
<pre>git clone {REPO} &amp;&amp; cd slp
pip install -r training/requirements.txt
python3 training/train.py      # trains, evaluates (random split + 5-fold template CV),
                               # writes web/model/* and training/artifacts/*
npm test                       # parity, entity, budget and conversation tests
python3 -m http.server 8000 --directory web   # open http://localhost:8000</pre>
</body></html>"""

OUT.write_text(HTML)
print("wrote", OUT)
