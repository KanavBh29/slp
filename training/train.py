"""Train the BiLSTM intent classifier and export it for the browser.

Usage:  python training/train.py
Outputs:
  web/model/model.json + model.bin   vocabulary + weights used by the web app
  training/artifacts/*               Keras model, metrics, plots used in the report

The embedding layer is initialised with pre-trained GloVe vectors
(glove-wiki-gigaword-50, 400k words, Wikipedia + Gigaword) and fine-tuned, so
the classifier also understands words that never appear in the training
templates (e.g. "farewell" ~ "goodbye", "pricey" ~ "expensive").
"""
import gzip
import urllib.request
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from intents import INTENTS  # noqa: E402
from nlp import MAXLEN, OOV, PAD, encode, tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "training" / "artifacts"
WEB_MODEL = ROOT / "web" / "model" / "model.json"
WEB_BIN = ROOT / "web" / "model" / "model.bin"
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz")
GLOVE = Path(os.environ.get("GLOVE_PATH", ROOT / "training" / ".cache" / "glove-wiki-gigaword-50.gz"))
EMB_DIM = 50
EXTRA_VOCAB = 8000  # most frequent GloVe words shipped to the browser
# Placeholder tokens get the GloVe vector of a representative word.
SPECIAL = {"xcar": "supercar", "xbrand": "manufacturer", "xnum": "million"}
SEED = 42
SOCIAL = {"greeting", "goodbye", "thanks", "out_of_scope", "bot_identity", "help"}

PREFIXES = ["hey ", "ok ", "please ", "can you tell me ", "i want to know ",
            "tell me ", "um ", "so ", "bot ", "quick question "]
SUFFIXES = [" please", " thanks", " now", " bro", " mate", " right"]


def fill(template: str, rng: random.Random) -> str:
    def car():
        r = rng.random()
        if r < 0.80:
            return "xcar"
        if r < 0.88:
            return "xbrand xcar"
        return rng.choice(["xunk", "xunk xunk"])  # unrecognised car name -> OOV

    out = template
    for ph in ("{car}", "{car2}"):
        while ph in out:
            out = out.replace(ph, car(), 1)
    out = out.replace("{brand}", "xbrand" if rng.random() < 0.9 else "xunk")
    out = out.replace("{num}", "xnum")
    return out


def augment(text: str, intent: str, rng: random.Random, noise: bool = True) -> str:
    if intent not in SOCIAL:
        if rng.random() < 0.3:
            text = rng.choice(PREFIXES) + text
        if rng.random() < 0.2:
            text = text + rng.choice(SUFFIXES)
    words = text.split()
    # Simulate speech-recognition errors: randomly drop one word.
    if noise and len(words) > 3 and rng.random() < 0.15:
        i = rng.randrange(len(words))
        if words[i] not in ("xcar", "xbrand", "xnum", "xunk"):
            del words[i]
    return " ".join(words)


def generate(templates_by_intent, n_per_template, rng, noise=True):
    xs, ys = [], []
    for intent, temps in templates_by_intent.items():
        for t in temps:
            seen = set()
            for i in range(n_per_template):
                s = fill(t, rng) if i == 0 else augment(fill(t, rng), intent, rng, noise)
                if s in seen:
                    continue
                seen.add(s)
                xs.append(tokenize(s))
                ys.append(intent)
    order = list(range(len(xs)))
    rng.shuffle(order)  # Keras' validation_split takes the *last* rows
    return [xs[i] for i in order], [ys[i] for i in order]


def load_glove():
    if not GLOVE.exists():
        GLOVE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading GloVe vectors ...")
        urllib.request.urlretrieve(GLOVE_URL, GLOVE)
    words, vecs = [], []
    with gzip.open(GLOVE, "rt", encoding="utf8") as f:
        next(f)  # header: "400000 50"
        for line in f:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype="float32"))
    return {w: i for i, w in enumerate(words)}, np.stack(vecs)


def build_vocab(token_lists, glove_idx, glove_words):
    """Training words + the most frequent plain GloVe words (frequency order)."""
    words = sorted({w for toks in token_lists for w in toks if w != "xunk"})
    extra = [w for w in glove_words[:EXTRA_VOCAB * 2] if w.isalnum() and w.isascii()][:EXTRA_VOCAB]
    vocab = [PAD, OOV] + words + sorted(set(extra) - set(words))
    return vocab, {w: i for i, w in enumerate(vocab)}


def embedding_matrix(vocab, glove_idx, glove_vecs):
    rng = np.random.default_rng(SEED)
    M = np.zeros((len(vocab), EMB_DIM), dtype="float32")
    missing = []
    for i, w in enumerate(vocab[1:], start=1):
        src = SPECIAL.get(w, w)
        if src in glove_idx:
            M[i] = glove_vecs[glove_idx[src]]
        elif w != OOV:
            M[i] = rng.normal(0, 0.4, EMB_DIM)
            missing.append(w)
    return quantize(M)[0], missing


def quantize(M):
    """Per-row int8 quantisation (what the browser receives)."""
    scale = np.abs(M).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(M / scale[:, None]).astype("int8")
    return (q.astype("float32") * scale[:, None]).astype("float32"), (q, scale.astype("float32"))


def bilstm(emb, n_classes, vocab_size=None):
    """emb=None -> embedding learned from scratch (ablation); else GloVe-initialised."""
    inp = keras.Input(shape=(MAXLEN,), dtype="int32", name="tokens")
    if emb is None:
        x = layers.Embedding(vocab_size, EMB_DIM, mask_zero=True, name="embedding")(inp)
    else:
        # Fine-tuned: rows of words that never occur in training receive no
        # gradient, so unseen words keep their pre-trained GloVe vectors.
        x = layers.Embedding(emb.shape[0], EMB_DIM, mask_zero=True, trainable=True,
                             embeddings_initializer=keras.initializers.Constant(emb),
                             name="embedding")(inp)
    x = layers.Bidirectional(layers.LSTM(64), name="bilstm")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    model = keras.Model(inp, out, name="intent_bilstm")
    model.compile(optimizer=keras.optimizers.Adam(2e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def bow_mlp(vocab_size, n_classes):
    """Baseline: bag-of-words multilayer perceptron (ignores word order)."""
    inp = keras.Input(shape=(vocab_size,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name="bow_mlp")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def to_bow(X, vocab_size):
    B = np.zeros((len(X), vocab_size), dtype="float32")
    for i, row in enumerate(X):
        for t in row:
            if t > 0:
                B[i, t] = 1.0
    return B


def macro_f1(y_true, y_pred, n):
    f1s, per = [], {}
    for c in range(n):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f)
        per[c] = (p, r, f, int(tp + fn))
    return float(np.mean(f1s)), per


def early_stop():
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)]


def evaluate(name, xtr, ytr, xte, yte, labels, glove, verbose=0):
    """Train the three compared models on one split and score them on the test set."""
    lab_idx = {l: i for i, l in enumerate(labels)}
    vocab, widx = build_vocab(xtr, glove[0], glove[2])
    emb, _ = embedding_matrix(vocab, glove[0], glove[1])
    Xtr = np.array([encode(t, widx) for t in xtr]); Ytr = np.array([lab_idx[y] for y in ytr])
    Xte = np.array([encode(t, widx) for t in xte]); Yte = np.array([lab_idx[y] for y in yte])
    out, n = {}, len(labels)
    for model_name in ("bow_mlp", "bilstm_scratch", "bilstm_glove"):
        tf.random.set_seed(SEED)
        if model_name == "bow_mlp":
            m = bow_mlp(len(vocab), n)
            h = m.fit(to_bow(Xtr, len(vocab)), Ytr, validation_split=0.1, epochs=80,
                      batch_size=32, callbacks=early_stop(), verbose=0)
            pred = m.predict(to_bow(Xte, len(vocab)), verbose=0).argmax(1)
        else:
            m = bilstm(emb if model_name == "bilstm_glove" else None, n, len(vocab))
            h = m.fit(Xtr, Ytr, validation_split=0.1, epochs=80, batch_size=32,
                      callbacks=early_stop(), verbose=verbose)
            pred = m.predict(Xte, verbose=0).argmax(1)
        f1, per = macro_f1(Yte, pred, n)
        trainable = int(sum(np.prod(w.shape) for w in m.trainable_weights))
        out[model_name] = {"accuracy": round(float((pred == Yte).mean()), 4),
                           "macro_f1": round(f1, 4), "trainable_params": trainable,
                           "epochs": len(h.history["loss"])}
        print(f"[{name}] {model_name:15s} acc={out[model_name]['accuracy']:.4f} "
              f"macroF1={f1:.4f}")
        if model_name == "bilstm_glove":
            details = {"history": h.history, "pred": pred, "y": Yte, "per": per,
                       "errors": [{"text": " ".join(xte[i]), "true": labels[Yte[i]],
                                   "pred": labels[pred[i]]} for i in np.where(pred != Yte)[0]]}
    return out, details


def main():
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    rng = random.Random(SEED)
    labels = list(INTENTS.keys())
    lab_idx = {l: i for i, l in enumerate(labels)}
    g_idx, g_vecs = load_glove()
    g_words = sorted(g_idx, key=g_idx.get)
    glove = (g_idx, g_vecs, g_words)
    ART.mkdir(parents=True, exist_ok=True)

    # ---- 1a. Random split of generated sentences (in-distribution) -------
    xs, ys = generate(INTENTS, 12, random.Random(SEED))
    k = int(len(xs) * 0.85)
    rand_res, _ = evaluate("random split", xs[:k], ys[:k], xs[k:], ys[k:], labels, glove)

    # ---- 1b. 5-fold cross-validation over *templates* ------------------
    # Each fold holds out 20% of every intent's phrasings, so test sentences
    # are worded differently from anything seen in training.
    K = 5
    folds = {}
    for intent, temps in INTENTS.items():
        temps = temps[:]
        rng.shuffle(temps)
        folds[intent] = [temps[i::K] for i in range(K)]
    cv = {m: [] for m in ("bow_mlp", "bilstm_scratch", "bilstm_glove")}
    for f in range(K):
        train_t = {i: [t for j, part in enumerate(fs) if j != f for t in part] for i, fs in folds.items()}
        test_t = {i: fs[f] for i, fs in folds.items()}
        xtr, ytr = generate(train_t, 12, rng)
        xte, yte = generate(test_t, 3, rng, noise=False)  # clean phrasings, no word dropout
        res, d = evaluate(f"template fold {f + 1}/{K}", xtr, ytr, xte, yte, labels, glove)
        for m, r in res.items():
            cv[m].append(r)
        if f == 0:
            plot(d["history"], d["y"], d["pred"], labels)
            fold1 = {"n_train": len(xtr), "n_test": len(xte), "details": d}
    summary = {m: {"accuracy_mean": round(float(np.mean([r["accuracy"] for r in rs])), 4),
                   "accuracy_std": round(float(np.std([r["accuracy"] for r in rs])), 4),
                   "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in rs])), 4),
                   "macro_f1_std": round(float(np.std([r["macro_f1"] for r in rs])), 4),
                   "folds": rs} for m, rs in cv.items()}
    for m, r in summary.items():
        print(f"[template 5-fold CV] {m:15s} acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} "
              f"macroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}")
    d = fold1["details"]

    metrics = {
        "n_intents": len(labels), "n_templates": sum(map(len, INTENTS.values())),
        "random_split": {"n_train": k_rand(xs), "n_test": len(xs) - k_rand(xs), **rand_res},
        "template_cv": {"folds": K, "fold1_n_train": fold1["n_train"], "fold1_n_test": fold1["n_test"], **summary},
        "per_intent_fold1": {labels[c]: {"precision": round(p, 3), "recall": round(r, 3),
                                         "f1": round(f, 3), "support": s}
                             for c, (p, r, f, s) in d["per"].items()},
        "errors_fold1": d["errors"][:40],
    }

    # ---- 2. Final model: all templates, exported to the browser ----------
    xall, yall = generate(INTENTS, 12, random.Random(SEED + 1))
    vocab, widx = build_vocab(xall, g_idx, g_words)
    emb, missing = embedding_matrix(vocab, g_idx, g_vecs)
    Xa = np.array([encode(t, widx) for t in xall]); Ya = np.array([lab_idx[y] for y in yall])
    tf.random.set_seed(SEED)
    final = bilstm(emb, len(labels))
    final.summary()
    fh = final.fit(Xa, Ya, validation_split=0.1, epochs=80, batch_size=32,
                   callbacks=early_stop(), verbose=0)
    metrics["final_model"] = {
        "n_examples": int(len(Xa)), "vocab_size": len(vocab),
        "words_without_glove_vector": missing,
        "total_params": int(final.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in final.trainable_weights)),
        "epochs_run": len(fh.history["loss"]),
        "best_val_accuracy": round(float(max(fh.history["val_accuracy"])), 4)}
    final.save_weights(ART / "intent_bilstm.weights.h5")
    export(final, vocab, labels)
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Reference predictions for the JS parity test (tests/parity.test.mjs).
    probe = ["how fast is the xcar", "where can i buy a xcar", "xcar vs xcar",
             "hello there", "what can i buy with xnum", "tell me a joke",
             "how much horsepower does it have", "xbrand", "farewell my friend",
             "is it pricey", "show me some electric hypercars please"]
    ref = final.predict(np.array([encode(tokenize(s), widx) for s in probe]), verbose=0)
    (ROOT / "tests" / "reference_predictions.json").write_text(json.dumps(
        [{"text": s, "label": labels[int(r.argmax())], "probs": [round(float(v), 6) for v in r]}
         for s, r in zip(probe, ref)], indent=1))
    print("Exported", WEB_MODEL, WEB_BIN)


def k_rand(xs):
    return int(len(xs) * 0.85)


def export(model, vocab, labels):
    """model.json (metadata + layout) and model.bin (float32 weights, int8 embedding)."""
    bi = model.get_layer("bilstm")
    tensors = {}
    for prefix, layer in (("lstm_fw", bi.forward_layer), ("lstm_bw", bi.backward_layer)):
        for name, w in zip(("kernel", "recurrent", "bias"), layer.get_weights()):
            tensors[f"{prefix}.{name}"] = w
    for lname in ("dense", "output"):
        k, b = model.get_layer(lname).get_weights()
        tensors[f"{lname}.kernel"], tensors[f"{lname}.bias"] = k, b
    deq, (q, scale) = quantize(model.get_layer("embedding").get_weights()[0])
    model.get_layer("embedding").set_weights([deq])  # keep Keras == browser for parity tests
    tensors["embedding.scale"] = scale

    layout, chunks, offset = {}, [], 0
    for name, w in tensors.items():
        w = np.ascontiguousarray(w, dtype="<f4")
        layout[name] = {"offset": offset, "shape": list(w.shape), "dtype": "float32"}
        chunks.append(w.tobytes()); offset += w.nbytes
    layout["embedding.q"] = {"offset": offset, "shape": list(q.shape), "dtype": "int8"}
    chunks.append(q.tobytes())
    WEB_MODEL.parent.mkdir(parents=True, exist_ok=True)
    WEB_BIN.write_bytes(b"".join(chunks))
    WEB_MODEL.write_text(json.dumps({
        "architecture": "GloVe-50d Embedding (fine-tuned) -> BiLSTM(64) -> Dense(64, relu) -> Dense(softmax)",
        "maxlen": MAXLEN, "labels": labels, "layout": layout, "vocab": vocab,
    }, separators=(",", ":")))


def plot(hist, y_true, y_pred, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("loss", "val_loss"):
        ax[0].plot(hist[k], label=k)
    for k in ("accuracy", "val_accuracy"):
        ax[1].plot(hist[k], label=k)
    ax[0].set_title("Loss"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ART / "training_curves.png", dpi=120); plt.close(fig)

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (template CV, fold 1)")
    fig.tight_layout(); fig.savefig(ART / "confusion_matrix.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
