// Speech recognition engines.
//  1. WebSpeechEngine: the browser's built-in recogniser (Web Speech API);
//     in Chrome/Edge this is Google's deep-learning ASR service.
//  2. WhisperEngine: OpenAI Whisper (tiny.en, transformer encoder-decoder),
//     run fully in the browser with Transformers.js / ONNX Runtime Web.

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
const WHISPER_MODEL = 'Xenova/whisper-tiny.en';

export class WebSpeechEngine {
  static supported() { return !!(window.SpeechRecognition || window.webkitSpeechRecognition); }

  constructor() { this.rec = null; }

  /** cb: {onInterim(text), onFinal(alternatives[{transcript, confidence}]), onError(msg, code), onEnd()} */
  start(lang, cb) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.lang = lang;
    rec.interimResults = true;
    rec.continuous = false;
    rec.maxAlternatives = 3;
    let done = false;
    rec.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        if (res.isFinal) {
          done = true;
          cb.onFinal(Array.from(res, alt => ({ transcript: alt.transcript.trim(), confidence: alt.confidence })));
        } else {
          interim += res[0].transcript;
        }
      }
      if (interim) cb.onInterim(interim);
    };
    rec.onerror = (e) => {
      const msg = {
        'not-allowed': 'Microphone permission was denied. Allow mic access in the address bar and try again.',
        'no-speech': 'I didn’t hear anything. Tap the mic and try again.',
        'network': 'The browser’s speech service could not be reached (network error).',
        'audio-capture': 'No microphone was found.',
        'aborted': null,
      }[e.error] ?? `Speech recognition error: ${e.error}`;
      if (msg) cb.onError(msg, e.error);
    };
    rec.onend = () => { this.rec = null; cb.onEnd(done); };
    rec.start();
    this.rec = rec;
  }

  stop() { this.rec?.stop(); }
  get active() { return !!this.rec; }
}

export class WhisperEngine {
  static supported() { return !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder); }

  constructor(onStatus) {
    this.onStatus = onStatus;
    this.asr = null;
    this.recorder = null;
  }

  async load() {
    if (this.asr) return this.asr;
    this.onStatus('Downloading Whisper model (first time only)…');
    const { pipeline } = await import(/* webpackIgnore: true */ TRANSFORMERS_URL);
    this.asr = await pipeline('automatic-speech-recognition', WHISPER_MODEL, {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
          this.onStatus(`Downloading Whisper model… ${Math.round(p.progress)}%`);
        }
      },
    });
    return this.asr;
  }

  // Push-to-talk: first call starts recording, stop() transcribes.
  async start(_lang, cb) {
    this.cb = cb;
    try {
      await this.load();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        this.recorder = null;
        try {
          cb.onInterim('Transcribing with Whisper…');
          const audio = await decodeTo16k(new Blob(chunks, { type: rec.mimeType }));
          const out = await this.asr(audio);
          const text = (out.text || '').trim();
          if (text && !/^\[.*\]$/.test(text)) cb.onFinal([{ transcript: text, confidence: null }]);
          else cb.onError('I didn’t hear anything. Try again.');
          cb.onEnd(!!text);
        } catch (err) {
          cb.onError(`Whisper failed: ${err.message}`); cb.onEnd(false);
        }
      };
      rec.start();
      this.recorder = rec;
      cb.onInterim('Recording… tap the mic again when you’re done.');
      // Safety stop after 12 seconds.
      this.timer = setTimeout(() => this.stop(), 12000);
    } catch (err) {
      cb.onError(err.name === 'NotAllowedError' ? 'Microphone permission was denied.' : `Could not start Whisper: ${err.message}`);
      cb.onEnd(false);
    }
  }

  stop() {
    clearTimeout(this.timer);
    if (this.recorder?.state === 'recording') this.recorder.stop();
  }

  get active() { return !!this.recorder; }
}

async function decodeTo16k(blob) {
  const buf = await blob.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(buf);
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

// Text-to-speech for the bot's answers.
export function speak(text, lang) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.replace(/[*_#]/g, ''));
  u.lang = lang;
  const voice = window.speechSynthesis.getVoices().find(v => v.lang === lang) ||
    window.speechSynthesis.getVoices().find(v => v.lang.startsWith('en'));
  if (voice) u.voice = voice;
  u.rate = 1.03;
  window.speechSynthesis.speak(u);
}
