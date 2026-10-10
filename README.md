# 🏎️ HyperBot: Voice-Enabled Supercar & Hypercar Chatbot

**Live demo:** https://kanavbh29.github.io/slp/

HyperBot is a voice chatbot for supercars and hypercars. You can ask it about a car's details, specs, price and where to buy it. Tap the mic and ask, for example, *"How fast is the Koenigsegg Jesko?"*, *"How much is a Ferrari F80?"* or *"Where can I buy a McLaren 750S?"*. The app shows the recognised speech and the chatbot's answer, and it reads the answer aloud.

| Requirement | How it is met |
|---|---|
| Speech recognition for voice input | Web Speech API (the browser's deep-learning ASR). Optional **Whisper-tiny** runs fully in the browser via Transformers.js, for browsers without Web Speech (e.g. Firefox). |
| Deep learning model for intent classification | **Bidirectional LSTM** with pre-trained **GloVe** word embeddings, trained in TensorFlow/Keras on 26 intents (796 utterance templates). Inference runs in the browser in pure JavaScript. |
| Display recognised speech and response | Every turn shows the transcript (with ASR confidence), the answer, the predicted intent with its confidence, and the detected entities. A side panel shows the full NLP pipeline. |
| Deployed online | Static site on GitHub Pages, deployed by `.github/workflows/deploy.yml` after the tests pass. |
| Report | [`REPORT.md`](REPORT.md) covers the dataset, architecture, methodology and results. |

## How it works

```
 🎤 voice ──► Speech recognition ──► text ──► normalise ──► entity masking ──► BiLSTM ──► intent ──► response
              (Web Speech / Whisper)                        "bugatti chiron"→xcar        (26 classes)   (car DB +
                                                             "$300k"→xnum                               dialogue state)
                                                                                                        └─► 🔊 text-to-speech
```

1. **ASR**: the browser turns speech into text. The app picks the best of the recogniser's n-best alternatives (it prefers one that mentions a known car).
2. **Entity extraction**: exact, fuzzy (Levenshtein) and phonetic matching against 35 cars and 18 brands. Phonetic matching catches mishearings like *"Bugatti sheeran"* → Chiron and *"Pagani wire a"* → Huayra. Budgets such as *"$2.5 million"*, *"300k"* and *"two million"* are parsed into numbers.
3. **Intent classification**: the masked sentence goes through the BiLSTM. If confidence is below 0.5, rule-based fallbacks take over.
4. **Response generation**: answers come from [`web/data/cars.json`](web/data/cars.json). The bot remembers the last car for follow-ups ("how fast is it?") and fills in missing details ("what's your budget?").

## Repository layout

```
web/                     ← the deployed site (static, no server needed)
  index.html, css/, js/
    js/speech.js         speech recognition (Web Speech API + Whisper) and text-to-speech
    js/nlp.js            normalisation, entity extraction, budget parser (mirrors training/nlp.py)
    js/model.js          BiLSTM forward pass in plain JavaScript
    js/chatbot.js        dialogue manager and response generation
  data/cars.json         knowledge base: 35 cars, 18 brands, marketplaces
  model/model.json|bin   exported vocabulary and weights (≈0.8 MB)
training/
  intents.py             intent dataset (26 intents, 796 templates)
  nlp.py                 preprocessing shared with the browser
  train.py               training, evaluation, baselines, export
  artifacts/             metrics.json, training curves, confusion matrix, weights
tests/                   JS↔Keras parity, tokenizer parity, entities, 32-turn conversation
REPORT.md                project report
```

## Run locally

```bash
# Serve the app (speech recognition needs http://localhost or https)
python3 -m http.server 8000 --directory web    # open http://localhost:8000

# Retrain the model (downloads GloVe, ~66 MB, on first run)
pip install -r training/requirements.txt
python3 training/train.py

# Tests
npm test
```

## Deploying

The workflow deploys `web/` to GitHub Pages on every push. **One-time setup:** go to *Settings → Pages → Build and deployment → Source* and choose **GitHub Actions**, then re-run the workflow. The site is published at `https://<user>.github.io/<repo>/`.

`web/` is a plain static folder, so it also deploys unchanged to Netlify (drag and drop the folder), Vercel, Cloudflare Pages or Render (set the publish directory to `web`).

> Speech recognition needs HTTPS (or localhost) and microphone permission. The Web Speech API works in Chrome, Edge and Safari. In Firefox, switch the engine to *Whisper* in Settings.

*Specs are manufacturer claims and prices are approximate. This is for education only.*
