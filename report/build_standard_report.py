"""Builds report/SLP_Standard_Report_HyperBot.html in the course's standard
report format (VIT SLP template); render with
`node report/render_pdf.cjs SLP_Standard_Report_HyperBot`.

Numbers come from training/artifacts/metrics.json, the intent corpus and the
car database, so the report stays in sync with the code.
"""
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from intents import INTENTS  # noqa: E402

M = json.loads((ROOT / "training/artifacts/metrics.json").read_text())
DB = json.loads((ROOT / "web/data/cars.json").read_text())
CARS = DB["cars"]
OUT = ROOT / "report" / "SLP_Standard_Report_HyperBot.html"
LIVE = "https://kanavbh29.github.io/slp/"
REPO = "https://github.com/KanavBh29/slp"
e = html.escape

# Author (single-member project).
TEAM = [
    ("Kanav", "Design, Implementation &amp; Deployment", "23BAI0018"),
]
GUIDE = "Prof. Swetha Vasudevan"
DATE = "11 October 2026"

fm, cv, rs = M["final_model"], M["template_cv"], M["random_split"]
g, b, s = cv["bilstm_glove"], cv["bow_mlp"], cv["bilstm_scratch"]
n_hyper = sum(c["class"] == "Hypercar" for c in CARS)
n_ev = sum(c["powertrain"] == "Electric" for c in CARS)
n_hy = sum(c["powertrain"] == "Hybrid" for c in CARS)


def pct(x):
    return f"{x * 100:.1f}%"


def per_intent_rows():
    per = M["per_intent_fold1"]
    rows, P, R, F, S = [], [], [], [], 0
    for name in INTENTS:
        p = per[name]
        rows.append(f"<tr><td>{name}</td><td>{p['precision']:.2f}</td><td>{p['recall']:.2f}</td>"
                    f"<td>{p['f1']:.2f}</td><td>{p['support']}</td></tr>")
        P.append(p["precision"]); R.append(p["recall"]); F.append(p["f1"]); S += p["support"]
    n = len(P)
    rows.append(f"<tr class='b'><td>Macro Average</td><td>{sum(P)/n:.2f}</td><td>{sum(R)/n:.2f}</td>"
                f"<td>{sum(F)/n:.2f}</td><td>{S}</td></tr>")
    return "\n".join(rows)


def weakest(k=3):
    per = M["per_intent_fold1"]
    return sorted(per.items(), key=lambda kv: kv[1]["f1"])[:k]


def dialogue_rows():
    script = ROOT / "report" / "_dialogue.mjs"
    script.write_text("""
import { readFileSync } from 'node:fs';
import { IntentModel } from '../web/js/model.js';
import { Chatbot } from '../web/js/chatbot.js';
const r = new URL('../', import.meta.url);
const bin = readFileSync(new URL('web/model/model.bin', r));
const bot = new Chatbot(new IntentModel(JSON.parse(readFileSync(new URL('web/model/model.json', r))),
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength)), JSON.parse(readFileSync(new URL('web/data/cars.json', r))));
console.log(JSON.stringify(JSON.parse(process.argv[2]).map(q => { const x = bot.reply(q); return { q, intent: x.intent, conf: x.confidence, a: x.speech }; })));
""")
    qs = ["Tell me about the Bugatti Chiron", "How fast is it?", "How much does it cost?",
          "Where can I buy one?", "Compare it with the Koenigsegg Jesko",
          "What can I buy with 400k?", "bugatti sheeran price", "What is the price of the Tesla Roadster?"]
    try:
        res = json.loads(subprocess.check_output(["node", str(script), json.dumps(qs)], cwd=ROOT))
    finally:
        script.unlink()
    return "\n".join(f"<tr><td>“{e(r['q'])}”</td><td>{r['intent']} ({pct(r['conf'])})</td><td>{e(r['a'])}</td></tr>"
                     for r in res)


team_rows = "\n".join(f"<tr><td class='b'>{n}</td><td>- {role}</td><td>{rid}</td></tr>" for n, role, rid in TEAM)
weak = ", ".join(f"<i>{k}</i> (F1 = {v['f1']:.2f})" for k, v in weakest())

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Speech and Language Processing - Project Report</title>
<style>
@page {{ size: A4; margin: 22mm 20mm 22mm 20mm; }}
html {{ -webkit-print-color-adjust: exact; }}
body {{ font-family: 'Times New Roman', 'Liberation Serif', Times, serif; font-size: 11.5pt; line-height: 1.45; color: #000; margin: 0; }}
h2 {{ font-size: 14pt; margin: 20px 0 8px; padding-bottom: 4px; border-bottom: 1.2px solid #000; break-after: avoid; }}
h3 {{ font-size: 12pt; margin: 14px 0 6px; break-after: avoid; }}
p {{ margin: 6px 0; text-align: justify; }}
ul, ol {{ margin: 6px 0; padding-left: 26px; }}
li {{ margin: 3px 0; text-align: justify; }}
code, pre {{ font-family: 'Courier New', 'Liberation Mono', monospace; font-size: 9.5pt; }}
code {{ background: #f2f2f2; padding: 0 2px; }}
pre {{ background: #f2f2f2; border: 1px solid #bbb; padding: 8px 10px; white-space: pre-wrap; break-inside: avoid; }}
table.t {{ width: 100%; border-collapse: collapse; margin: 8px 0 10px; font-size: 10pt; }}
table.t th, table.t td {{ border: 1px solid #000; padding: 3px 7px; text-align: left; vertical-align: top; }}
table.t th {{ background: #f2f2f2; }}
table.t tr {{ break-inside: avoid; }}
.b, tr.b td {{ font-weight: bold; }}
figure {{ margin: 10px 0; text-align: center; break-inside: avoid; }}
figure img {{ max-width: 100%; border: 1px solid #999; }}
figcaption {{ font-style: italic; font-size: 9.5pt; margin-top: 4px; }}
.ref {{ margin: 8px 0; text-align: justify; }}
.center {{ text-align: center; }}
.cover {{ text-align: center; break-after: page; padding-top: 6mm; }}
.cover img {{ width: 62mm; }}
.cover .inst {{ font-size: 12pt; margin: 4px 0; }}
.cover h1 {{ font-size: 21pt; letter-spacing: .02em; margin: 18px 0 6px; line-height: 1.3; border-bottom: 1.5px solid #000; padding-bottom: 10px; }}
.cover .title {{ font-size: 14.5pt; font-weight: bold; margin: 22px 0 18px; line-height: 1.35; }}
.cover table {{ margin: 6px auto; border-collapse: collapse; font-size: 11pt; }}
.cover table td {{ border: 1px solid #000; padding: 3px 10px; text-align: left; }}
.cover p {{ text-align: center; margin: 4px 0; }}
.link {{ text-align: center; font-weight: bold; margin: 8px 0; }}
</style></head><body>

<section class="cover">
  <img src="vit_logo.png" alt="VIT">
  <p class="inst" style="margin-top:14px">VIT VELLORE</p>
  <p class="inst">SCOPE (School of Computer Science Engineering)</p>
  <h1>SPEECH AND LANGUAGE<br>PROCESSING</h1>
  <p>A Project Report</p>
  <p style="margin-top:16px">Submitted in partial fulfilment of the requirements under the</p>
  <p class="b">Theory of Speech and Language Processing</p>
  <p>BCSE419L</p>
  <div class="title">HyperBot: A Voice-Enabled Chatbot for Supercar<br>and Hypercar Information using Speech<br>Recognition and Deep Learning</div>
  <p>Submitted by</p>
  <table>{team_rows}</table>
  <p style="margin-top:18px">Under the guidance of</p>
  <p class="b">{GUIDE}</p>
  <p style="margin-top:18px">{DATE}</p>
  <p>VIT Vellore</p>
</section>

<h2>1. Introduction</h2>
<h3>1.1 Problem Statement</h3>
<p>Supercar and hypercar information (specifications, prices, production numbers and purchasing channels) is spread across manufacturer press releases, dealer websites and secondary-market listings. A spoken conversational assistant is a natural way to ask such questions, but it combines two language-processing problems. First, a speech recognition front end must turn the user's voice into text, and general-purpose recognisers often mishear exotic model names ("Chiron" → "sheeran", "Huayra" → "wire a"). Second, a natural language understanding component must work out what the user is asking for (the intent) and which car they mean (the entities), however the question is phrased. This project builds an end-to-end, publicly deployed voice chatbot that solves both problems. It uses a speech recognition pathway, entity extraction that is robust to recognition errors, and a deep-learning intent classifier, together with a dialogue manager that generates grounded answers from a curated knowledge base.</p>
<h3>1.2 Objectives</h3>
<ol>
  <li>Accept voice input and convert speech into text using a suitable speech recognition technique, with an on-device fallback when the browser's cloud speech service is unavailable.</li>
  <li>Build a deep-learning intent classifier (Bidirectional LSTM with pre-trained GloVe embeddings) that maps a user utterance to one of {M['n_intents']} chatbot intents.</li>
  <li>Make car-name recognition robust to speech recognition errors through n-best rescoring, fuzzy matching and phonetic matching.</li>
  <li>Generate appropriate responses (details, specifications, price, where to buy, comparisons and recommendations) and display both the recognised speech and the chatbot response.</li>
  <li>Evaluate the classifier honestly on phrasings never seen during training, against non-sequential and non-pretrained baselines.</li>
  <li>Deploy the working application online with a publicly accessible link.</li>
</ol>

<h2>2. Literature Review</h2>
<p>The following works were studied as the foundation for the speech recognition component, the word representations, the intent classification model, the dataset design, and the matching techniques used to correct recognition errors in this project.</p>
<h3>2.1 Speech Recognition</h3>
<p class="ref"><b>[1] Hinton, G., Deng, L., Yu, D., Dahl, G. E., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T. N., &amp; Kingsbury, B. (2012).</b> <i>Deep Neural Networks for Acoustic Modeling in Speech Recognition: The Shared Views of Four Research Groups.</i> IEEE Signal Processing Magazine, 29(6), 82-97. This paper established deep neural acoustic models as the basis of modern commercial recognisers, including the cloud service behind the browser Web Speech API used as this project's default speech recognition engine.</p>
<p class="ref"><b>[2] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., &amp; Sutskever, I. (2022).</b> <i>Robust Speech Recognition via Large-Scale Weak Supervision.</i> arXiv:2212.04356. This paper introduces Whisper, a transformer encoder-decoder trained on 680,000 hours of weakly supervised audio. Its <i>tiny.en</i> variant runs entirely in the browser as this project's fallback recogniser.</p>
<h3>2.2 Word Representations</h3>
<p class="ref"><b>[3] Pennington, J., Socher, R., &amp; Manning, C. D. (2014).</b> <i>GloVe: Global Vectors for Word Representation.</i> Proceedings of EMNLP 2014, 1532-1543. This paper introduces the pre-trained word vectors used to initialise this project's embedding layer. With them, the classifier can interpret synonyms that never appear in the hand-written training data.</p>
<h3>2.3 Recurrent Networks and Intent Classification</h3>
<p class="ref"><b>[4] Hochreiter, S., &amp; Schmidhuber, J. (1997).</b> <i>Long Short-Term Memory.</i> Neural Computation, 9(8), 1735-1780. This paper introduces the gated recurrent unit (the LSTM cell) at the core of the intent classifier, which models word order in variable-length utterances.</p>
<p class="ref"><b>[5] Schuster, M., &amp; Paliwal, K. K. (1997).</b> <i>Bidirectional Recurrent Neural Networks.</i> IEEE Transactions on Signal Processing, 45(11), 2673-2681. This paper motivates reading each utterance in both directions, the bidirectional wrapper used in this project's model.</p>
<p class="ref"><b>[6] Liu, B., &amp; Lane, I. (2016).</b> <i>Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling.</i> Proceedings of Interspeech 2016, 685-689. This paper demonstrates recurrent networks for spoken-language intent detection with slots, which informed this project's split into an intent classifier plus slot (entity) handling.</p>
<h3>2.4 Dataset Design for Conversational Agents</h3>
<p class="ref"><b>[7] Hemphill, C. T., Godfrey, J. J., &amp; Doddington, G. R. (1990).</b> <i>The ATIS Spoken Language Systems Pilot Corpus.</i> Proceedings of the DARPA Speech and Natural Language Workshop, 96-101. This is the classic spoken-language-understanding corpus of domain-specific intents. No equivalent exists for the automotive domain, which motivated building a custom intent corpus.</p>
<p class="ref"><b>[8] Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., &amp; Mars, J. (2019).</b> <i>An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.</i> Proceedings of EMNLP-IJCNLP 2019, 1311-1316. This paper shows that deployed intent classifiers must handle off-topic queries, which motivated the explicit <i>out_of_scope</i> class and the confidence threshold used in this project.</p>
<p class="ref"><b>[9] Wei, J., &amp; Zou, K. (2019).</b> <i>EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks.</i> Proceedings of EMNLP-IJCNLP 2019, 6382-6388. This paper describes simple augmentations such as random word deletion. This project adapts them to simulate speech recognition errors in the training data.</p>
<h3>2.5 Approximate String and Phonetic Matching</h3>
<p class="ref"><b>[10] Levenshtein, V. I. (1966).</b> <i>Binary Codes Capable of Correcting Deletions, Insertions, and Reversals.</i> Soviet Physics Doklady, 10(8), 707-710. This paper defines the edit distance used to match misrecognised car names to the database aliases.</p>
<p class="ref"><b>[11] Philips, L. (1990).</b> <i>Hanging on the Metaphone.</i> Computer Language, 7(12), 39-43. This paper introduces the Metaphone phonetic encoding. A simplified version lets this project match words that sound alike but are spelled differently ("sheeran" and "chiron" both encode to <code>xrn</code>).</p>

<h2>3. Methodology</h2>
<h3>3.1 Dataset Description</h3>
<p>Two datasets were built for this project. The first is an <b>intent corpus</b> of {M['n_templates']} hand-written utterance templates over {M['n_intents']} intents ({min(map(len, INTENTS.values()))} to {max(map(len, INTENTS.values()))} templates per intent). It covers greetings and social turns, car-specific questions (overview, specifications, engine, power, top speed, acceleration, price, where to buy, production), comparisons, rankings, budget recommendations, definitions, and an <i>out_of_scope</i> class. Templates contain the slot placeholders <code>{{car}}</code>, <code>{{brand}}</code> and <code>{{num}}</code>, which are <b>delexicalised</b> into the tokens <code>xcar</code>, <code>xbrand</code> and <code>xnum</code>. The network therefore learns the intent of a sentence instead of memorising car names, and new cars can be added without retraining.</p>
<p>Each template is expanded into 12 variants. The variants add polite prefixes and suffixes, fill 12% of car slots with out-of-vocabulary tokens to simulate an unrecognised name, and apply random deletion of one non-entity word (probability 0.15) to simulate speech recognition errors. The real split sizes used are:</p>
<table class="t">
  <tr><th>Split</th><th>Sentence Count</th></tr>
  <tr><td>Random split: train</td><td>{rs['n_train']:,}</td></tr>
  <tr><td>Random split: test</td><td>{rs['n_test']:,}</td></tr>
  <tr><td>Template 5-fold CV: train (per fold, approx.)</td><td>{cv['fold1_n_train']:,}</td></tr>
  <tr><td>Template 5-fold CV: test (per fold, approx.)</td><td>{cv['fold1_n_test']:,}</td></tr>
  <tr><td>Final deployed model: all templates</td><td>{fm['n_examples']:,}</td></tr>
</table>
<p>In the template cross-validation, the held-out test sentences come from templates that are never used in training, and they are generated without word deletion. This measures generalisation to genuinely new phrasings rather than to re-worded copies of training sentences.</p>
<p>The second dataset is a <b>knowledge base</b> of {len(CARS)} cars from {len(DB['brands'])} manufacturers ({n_hyper} hypercars and {len(CARS) - n_hyper} supercars; {n_ev} fully electric and {n_hy} hybrid). For each car it stores the engine, power, torque, 0-100 km/h time, top speed, approximate price in USD, production volume, availability status, a description, and aliases (including common mishearings). Specifications are manufacturer-claimed figures. Prices are approximate US base prices for new cars, or typical market values for sold-out and discontinued cars.</p>

<h3>3.2 Speech Recognition</h3>
<p>Voice input is captured in the browser. Two recognition engines are supported:</p>
<ul>
  <li><b>Web Speech API (default):</b> the browser vendor's neural recogniser. It streams interim results, which are shown live, and returns an n-best list of three alternatives with confidence scores. Supported accents are en-US, en-GB, en-IN and en-AU.</li>
  <li><b>Whisper-tiny.en (fallback):</b> a 39M-parameter transformer encoder-decoder that runs on the device through Transformers.js and ONNX Runtime Web. Recorded audio is resampled to 16 kHz mono before transcription. The application switches to this engine automatically when the cloud speech service cannot be reached (for example in the Brave browser or on restricted networks).</li>
</ul>
<p>Among the n-best alternatives, the top hypothesis is used unless only a lower-ranked one mentions a known car, in which case that alternative is chosen (n-best rescoring). The recognised text and its confidence are displayed next to every chatbot response.</p>

<h3>3.3 Text Preprocessing and Entity Extraction</h3>
<p>The recognised text is lower-cased and normalised: apostrophes are removed and punctuation is replaced, while decimals and model names such as "T.50" are preserved. Entities are then extracted in a fixed order:</p>
<ol>
  <li>exact car aliases, longest first (so that "Bugatti Chiron" wins over "Bugatti");</li>
  <li>fuzzy car matches, where a window of one to three words is compared with every alias using normalised Levenshtein similarity (≥ 0.8) [10];</li>
  <li>phonetic car matches, using a simplified Metaphone key [11];</li>
  <li>brand matches.</li>
</ol>
<p>Fuzzy and phonetic matching are applied only to windows that contain at least one uncommon word, to avoid false positives on everyday English. Monetary amounts and numbers are masked as <code>xnum</code>, while names such as "V12" are kept. A dedicated parser converts budgets to US dollars, including spoken forms ("two million", "half a million", "five hundred thousand") and Indian units ("3 crore"). Tokens are mapped to vocabulary indices (index 1 = out-of-vocabulary) and post-padded to a length of 24.</p>

<h3>3.4 Intent Classification Model</h3>
<p>The classifier is a Bidirectional LSTM [4, 5] implemented in TensorFlow/Keras:</p>
<ul>
  <li><b>Embedding layer:</b> {fm['vocab_size']:,} × 50, initialised with GloVe-50d vectors [3] (glove-wiki-gigaword-50) and fine-tuned, with padding masked. The vocabulary contains every training word plus the 8,000 most frequent English words. Rows for words absent from training receive no gradient and therefore keep their pre-trained meaning. The placeholder tokens start from the vectors of "supercar", "manufacturer" and "million".</li>
  <li><b>Bidirectional LSTM:</b> 64 units per direction, giving a 128-dimensional sentence vector.</li>
  <li><b>Classification head:</b> dropout 0.3 → dense 64 (ReLU) → dropout 0.3 → dense {M['n_intents']} (softmax).</li>
</ul>
<p>The model has {fm['total_params']:,} parameters in total. It is trained with the Adam optimiser (learning rate 2 × 10<sup>-3</sup>), sparse categorical cross-entropy, batch size 32 and a 10% validation split, for up to 80 epochs with early stopping on validation loss (patience 12, best weights restored). Two baselines are trained under identical conditions:</p>
<ul>
  <li>a bag-of-words multilayer perceptron, which ignores word order;</li>
  <li>the same BiLSTM with embeddings learned from scratch, which uses no transfer learning.</li>
</ul>
<p>For deployment, the trained weights are exported to a compact binary file, with int8-quantised embeddings and float32 elsewhere (≈0.7 MB). A hand-written JavaScript implementation of the masked BiLSTM forward pass runs inference in the browser in a few milliseconds.</p>

<h3>3.5 Dialogue Management and Response Generation</h3>
<p>The dialogue manager combines the classifier output with the extracted entities:</p>
<ul>
  <li><b>Confidence threshold:</b> if the top intent probability is below 0.5, rule-based fallbacks use the entities (two cars → compare, one car → overview, a budget → recommendations). Otherwise the bot asks the user to rephrase.</li>
  <li><b>Context memory:</b> the most recent cars are remembered, so that follow-ups such as "How fast is it?" and "Compare it with the Jesko" work. Context is reused only when a pronoun or a very short question indicates a follow-up. An unknown car ("Tesla Roadster") gets a clear "not in my database" answer instead of facts about the previous car.</li>
  <li><b>Slot filling:</b> "Recommend me a car" is followed by "What is your budget?".</li>
  <li><b>Where-to-buy answers</b> depend on availability:
    <ul>
      <li>in production: the official dealer and configurator;</li>
      <li>limited production: the manufacturer, plus the secondary market;</li>
      <li>sold out or discontinued: specialist marketplaces (duPont REGISTRY, JamesEdition) and auction houses (Bring a Trailer, RM Sotheby's).</li>
    </ul>
  </li>
</ul>
<p>Responses are shown as text with spec cards, comparison tables and ranked lists, and are read aloud with speech synthesis.</p>

<h3>3.6 Deployment</h3>
<p>The application is a static web application (HTML, CSS and JavaScript modules, the JSON knowledge base and the exported model weights). Speech recognition, entity extraction, inference and response generation all run in the browser, so no backend server or API key is needed. A GitHub Actions pipeline runs the automated test suite and then publishes the site to GitHub Pages.</p>

<h2>4. Results</h2>
<h3>4.1 Intent Classification</h3>
<table class="t">
  <tr><th>Model</th><th>Random-split Accuracy</th><th>Random-split Macro-F1</th><th>Template CV Accuracy</th><th>Template CV Macro-F1</th></tr>
  <tr><td>Bag-of-words MLP (baseline)</td><td>{pct(rs['bow_mlp']['accuracy'])}</td><td>{rs['bow_mlp']['macro_f1']:.3f}</td><td>{pct(b['accuracy_mean'])} ± {b['accuracy_std']*100:.1f}</td><td>{b['macro_f1_mean']:.3f} ± {b['macro_f1_std']:.3f}</td></tr>
  <tr><td>BiLSTM, embeddings from scratch</td><td>{pct(rs['bilstm_scratch']['accuracy'])}</td><td>{rs['bilstm_scratch']['macro_f1']:.3f}</td><td>{pct(s['accuracy_mean'])} ± {s['accuracy_std']*100:.1f}</td><td>{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}</td></tr>
  <tr class="b"><td>BiLSTM + GloVe (deployed)</td><td>{pct(rs['bilstm_glove']['accuracy'])}</td><td>{rs['bilstm_glove']['macro_f1']:.3f}</td><td>{pct(g['accuracy_mean'])} ± {g['accuracy_std']*100:.1f}</td><td>{g['macro_f1_mean']:.3f} ± {g['macro_f1_std']:.3f}</td></tr>
</table>
<p>On the random split, test sentences can share a template with training sentences, and all three models exceed 94% accuracy. The bag-of-words model is slightly ahead there because keyword memorisation suffices. On the stricter template cross-validation, every test sentence is a phrasing the model has never seen. There, pre-trained GloVe embeddings improve the BiLSTM by <b>{(g['accuracy_mean'] - s['accuracy_mean']) * 100:.1f} accuracy points</b> over the same network trained from scratch, confirming the value of transfer learning. The GloVe-BiLSTM has the highest mean accuracy and macro-F1 of all three models, although its margin over the bag-of-words baseline lies within the fold-to-fold variation. The deployed model, trained on all templates, reaches <b>{pct(fm['best_val_accuracy'])} validation accuracy</b>.</p>

<h3>4.2 Per-Intent Performance (Template CV, Fold 1)</h3>
<table class="t">
  <tr><th>Intent</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>
  {per_intent_rows()}
</table>
<p>The weakest intents are {weak}. Their errors come from one-word slang the model has no embedding for ("thx", "cya"), off-topic questions shaped like "who/what is X", and questions about the bot ("what technology do you use") that are confused with <i>help</i>. Intents tied to distinctive vocabulary (power, fastest, cheapest, most powerful) reach F1 = 1.00. In the running application, the dialogue manager limits the impact of misclassifications: a car-specific intent without any car mentioned produces a clarifying question rather than a wrong fact.</p>
<figure><img src="../training/artifacts/confusion_matrix.png" style="width:78%"><figcaption>Figure 1: Confusion matrix of the deployed architecture on held-out templates (template CV, fold 1).</figcaption></figure>
<figure><img src="../training/artifacts/training_curves.png" style="width:92%"><figcaption>Figure 2: Training and validation loss and accuracy (fold 1); early stopping restores the best weights.</figcaption></figure>

<div style="break-inside: avoid"><h3>4.3 Speech-Error Robustness and System Tests</h3>
<table class="t">
  <tr><th>Test (run automatically before every deployment)</th><th>Result</th></tr>
  <tr><td>Browser BiLSTM vs Keras reference predictions (11 sentences)</td><td>Identical labels; max probability difference 5 × 10<sup>-7</sup></td></tr>
  <tr><td>Python vs JavaScript tokenizer (13 tricky inputs)</td><td>Identical</td></tr>
  <tr><td>Entity extraction incl. misrecognised names ("bugatti sheeran", "koenig seg jesko", "rimac never a")</td><td>11 / 11 correct</td></tr>
  <tr><td>Budget parser ("$300k", "2.5 million dollars", "two million", "half a million", "3 crore rupees")</td><td>8 / 8 correct</td></tr>
  <tr><td>32-turn scripted conversation (follow-ups, comparisons, slot filling, out-of-scope)</td><td>32 / 32 intents and entities correct</td></tr>
</table></div>
<h3>4.4 Sample Dialogue</h3>
<p>The table below is generated by running the deployed model on a scripted conversation:</p>
<table class="t" style="font-size:9.5pt">
  <tr><th style="width:27%">User (recognised speech)</th><th style="width:21%">Predicted Intent</th><th>HyperBot Response</th></tr>
  {dialogue_rows()}
</table>

<h2>5. Demonstration</h2>
<p>The system is delivered as a public web application. It shows the recognised speech with its recognition confidence, the chatbot response (with spec cards, comparison tables and purchase links), the predicted intent with its confidence, and a live panel that traces every stage of the language-processing pipeline. Answers are also spoken aloud. The screenshots below are captured from the running application.</p>
<figure><img src="screenshot-overview.png"><figcaption>Figure 3: Voice query "tell me about the Bugatti Tourbillon": the recognised speech (94% confidence), the response with a spec card, the predicted intent and the NLP pipeline panel.</figcaption></figure>
<figure><img src="screenshot-desktop.png"><figcaption>Figure 4: A comparison table (winners highlighted) and a where-to-buy answer with official-dealer and marketplace links.</figcaption></figure>
<h3>5.1 Live Deployment, Repository and Local Setup</h3>
<p>The application is deployed and publicly accessible at:</p>
<p class="link">{LIVE}</p>
<p>The complete source code is hosted at:</p>
<p class="link">{REPO}</p>
<pre>git clone {REPO}
cd slp
pip install -r training/requirements.txt
python training/train.py                        # train, evaluate, export model
npm test                                        # run all tests
python -m http.server 8000 --directory web      # open http://localhost:8000</pre>
<p>Speech input requires HTTPS (or localhost) and microphone permission. It works in Chrome, Edge and Safari, and other browsers fall back to the on-device Whisper engine.</p>

<h2>6. Conclusion</h2>
<p>This project implements a complete, publicly deployed voice-enabled chatbot for supercar and hypercar information. Speech is converted to text by a neural speech recognition engine, with an on-device Whisper fallback. A Bidirectional LSTM with transfer-learned GloVe embeddings classifies the user's intent among {M['n_intents']} classes, and a dialogue manager with context memory and slot filling generates grounded answers about the details, specifications, prices and purchasing options of {len(CARS)} cars. On phrasings never seen during training, the deployed model reaches {pct(g['accuracy_mean'])} accuracy, {(g['accuracy_mean'] - s['accuracy_mean']) * 100:.1f} points above the same architecture without pre-trained embeddings. Errors specific to speech recognition, such as misheard car names, are handled with n-best rescoring, fuzzy matching and phonetic matching. The interface displays both the recognised speech and the chatbot response, and the whole pipeline runs in the user's browser.</p>

<h2>7. Future Work</h2>
<ul>
  <li>Replace the BiLSTM with a fine-tuned transformer encoder (e.g. DistilBERT) and compare accuracy against browser inference cost.</li>
  <li>Replace dictionary-based entity extraction with joint intent detection and slot tagging (e.g. a BiLSTM-CRF or the attention model of [6]).</li>
  <li>Collect real, anonymised user queries to extend the synthetic intent corpus, and measure end-to-end word error rate on recorded car-related speech.</li>
  <li>Connect the knowledge base to live listing data so that prices and availability stay current, and add multilingual speech input.</li>
</ul>

<h2>8. References</h2>
<ol style="font-size:10.5pt">
  <li>Hinton, G., Deng, L., Yu, D., Dahl, G. E., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T. N., &amp; Kingsbury, B. (2012). Deep Neural Networks for Acoustic Modeling in Speech Recognition. <i>IEEE Signal Processing Magazine</i>, 29(6), 82-97.</li>
  <li>Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., &amp; Sutskever, I. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. <i>arXiv:2212.04356</i>.</li>
  <li>Pennington, J., Socher, R., &amp; Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. <i>Proceedings of EMNLP 2014</i>, 1532-1543.</li>
  <li>Hochreiter, S., &amp; Schmidhuber, J. (1997). Long Short-Term Memory. <i>Neural Computation</i>, 9(8), 1735-1780.</li>
  <li>Schuster, M., &amp; Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. <i>IEEE Transactions on Signal Processing</i>, 45(11), 2673-2681.</li>
  <li>Liu, B., &amp; Lane, I. (2016). Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling. <i>Proceedings of Interspeech 2016</i>, 685-689.</li>
  <li>Hemphill, C. T., Godfrey, J. J., &amp; Doddington, G. R. (1990). The ATIS Spoken Language Systems Pilot Corpus. <i>Proceedings of the DARPA Speech and Natural Language Workshop</i>, 96-101.</li>
  <li>Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. <i>Proceedings of EMNLP-IJCNLP 2019</i>, 1311-1316.</li>
  <li>Wei, J., &amp; Zou, K. (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. <i>Proceedings of EMNLP-IJCNLP 2019</i>, 6382-6388.</li>
  <li>Levenshtein, V. I. (1966). Binary Codes Capable of Correcting Deletions, Insertions, and Reversals. <i>Soviet Physics Doklady</i>, 10(8), 707-710.</li>
  <li>Philips, L. (1990). Hanging on the Metaphone. <i>Computer Language</i>, 7(12), 39-43.</li>
</ol>
</body></html>"""

OUT.write_text(HTML)
print("wrote", OUT)
