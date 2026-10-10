// Pure-JavaScript inference for the Keras BiLSTM intent classifier.
// Weights are exported by training/train.py (model.json + model.bin).

const sigmoid = x => 1 / (1 + Math.exp(-x));

export class IntentModel {
  static async load(baseUrl) {
    const [meta, buf] = await Promise.all([
      fetch(`${baseUrl}/model.json`).then(r => r.json()),
      fetch(`${baseUrl}/model.bin`).then(r => r.arrayBuffer()),
    ]);
    return new IntentModel(meta, buf);
  }

  constructor(meta, buffer) {
    this.meta = meta;
    this.labels = meta.labels;
    this.maxlen = meta.maxlen;
    this.vocab = meta.vocab;
    this.wordIndex = new Map(meta.vocab.map((w, i) => [w, i]));
    const t = name => {
      const { offset, shape, dtype } = meta.layout[name];
      const n = shape.reduce((a, b) => a * b, 1);
      const arr = dtype === 'int8' ? new Int8Array(buffer, offset, n) : new Float32Array(buffer, offset, n);
      return { data: arr, shape };
    };
    this.embQ = t('embedding.q');
    this.embScale = t('embedding.scale').data;
    this.dim = this.embQ.shape[1];
    this.fw = { k: t('lstm_fw.kernel'), u: t('lstm_fw.recurrent'), b: t('lstm_fw.bias') };
    this.bw = { k: t('lstm_bw.kernel'), u: t('lstm_bw.recurrent'), b: t('lstm_bw.bias') };
    this.dense = { k: t('dense.kernel'), b: t('dense.bias') };
    this.out = { k: t('output.kernel'), b: t('output.bias') };
    this.units = this.fw.u.shape[0];
  }

  embed(id) {
    const v = new Float32Array(this.dim), s = this.embScale[id], q = this.embQ.data;
    for (let j = 0; j < this.dim; j++) v[j] = q[id * this.dim + j] * s;
    return v;
  }

  // Keras LSTM, gate order i, f, c, o; activation tanh, recurrent sigmoid.
  lstm(seq, p) {
    const U = this.units, G = 4 * U;
    let h = new Float32Array(U), c = new Float32Array(U);
    const K = p.k.data, R = p.u.data, B = p.b.data, D = this.dim;
    for (const x of seq) {
      const z = Float32Array.from(B);
      for (let i = 0; i < D; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * G;
        for (let g = 0; g < G; g++) z[g] += xi * K[row + g];
      }
      for (let i = 0; i < U; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = i * G;
        for (let g = 0; g < G; g++) z[g] += hi * R[row + g];
      }
      const hn = new Float32Array(U), cn = new Float32Array(U);
      for (let j = 0; j < U; j++) {
        const ig = sigmoid(z[j]), fg = sigmoid(z[U + j]);
        const cc = Math.tanh(z[2 * U + j]), og = sigmoid(z[3 * U + j]);
        cn[j] = fg * c[j] + ig * cc;
        hn[j] = og * Math.tanh(cn[j]);
      }
      h = hn; c = cn;
    }
    return h;
  }

  dense_(x, p, act) {
    const [nin, nout] = p.k.shape, K = p.k.data;
    const y = Float32Array.from(p.b.data);
    for (let i = 0; i < nin; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      for (let o = 0; o < nout; o++) y[o] += xi * K[i * nout + o];
    }
    if (act === 'relu') for (let o = 0; o < nout; o++) y[o] = Math.max(0, y[o]);
    return y;
  }

  /** ids: padded token ids (0 = padding, masked like Keras mask_zero=True). */
  predictIds(ids) {
    const seq = ids.filter(id => id !== 0).map(id => this.embed(id));
    if (!seq.length) return this.labels.map(() => 1 / this.labels.length);
    const hf = this.lstm(seq, this.fw);
    const hb = this.lstm([...seq].reverse(), this.bw);
    const hcat = new Float32Array(2 * this.units);
    hcat.set(hf, 0); hcat.set(hb, this.units);
    const d = this.dense_(hcat, this.dense, 'relu');
    const logits = this.dense_(d, this.out);
    const mx = Math.max(...logits);
    const ex = Array.from(logits, v => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / sum);
  }
}
