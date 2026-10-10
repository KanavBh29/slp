"""Text preprocessing shared by training (Python) and inference (web/js/nlp.js).

Both implementations MUST stay in sync; tests/parity.test.mjs checks that the
browser model reproduces the Keras predictions.
"""
import re

PAD, OOV = "<pad>", "<oov>"
MAXLEN = 24

# Numbers / money ("300k", "$2.5 million", "2.5s", "100") -> "xnum"; "v8" is kept
NUM_RE = re.compile(
    r"(?<![a-z0-9])\$?\d+(?:[.,]\d+)*"
    r"(?:\s*(?:k|m|mn|mil|million|millions|thousand|grand|lakh|lakhs|crore|crores|usd|dollars)\b|[a-z]+)?"
)


def normalize(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"['’]", "", text)            # what's -> whats
    text = re.sub(r"[^a-z0-9$.]+", " ", text)         # keep letters, digits, $ and .
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", text)  # sentence dots
    return re.sub(r"\s+", " ", text).strip()


def mask_numbers(text: str) -> str:
    text = NUM_RE.sub("xnum", text)
    text = text.replace("$", " ").replace(".", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return mask_numbers(normalize(text)).split()


def encode(tokens: list[str], word_index: dict, maxlen: int = MAXLEN) -> list[int]:
    ids = [word_index.get(t, 1) for t in tokens][:maxlen]
    return ids + [0] * (maxlen - len(ids))
