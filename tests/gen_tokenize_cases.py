"""Writes tests/tokenize_cases.json: Python tokenizer outputs the JS tokenizer must match."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from nlp import tokenize  # noqa: E402

CASES = [
    "What's the top speed of the xcar?", "0-100 in 2.5s for $2.5 million",
    "is it a V8 or V12", "I have 300k dollars", "Under $1,500,000 please!",
    "compare xcar & xcar", "t.50 price", "how much... is it", "Hello, bot.",
    "xbrand's cars", "reach 60 mph", "budget: 3 crore", "what's   up",
]
out = Path(__file__).with_name("tokenize_cases.json")
out.write_text(json.dumps([{"text": c, "tokens": tokenize(c)} for c in CASES], indent=1))
print(f"wrote {len(CASES)} cases to {out.name}")
