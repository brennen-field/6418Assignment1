"""Sentiment classifier for Amazon Gift_Cards reviews via an OpenAI-compatible LLM endpoint.

Classifies each review (using ONLY the title and text) as positive / negative / neutral,
printing each label and ending with a per-class tally. No files are written.

Config via environment variables (with safe local defaults):
    ENDPOINT_URL   - base URL, e.g. http://host:9000/v1
    ENDPOINT_KEY   - API key / bearer token
    ENDPOINT_MODEL - model id as listed by the server
"""

import gzip
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://dobolyi.com:9000/v1")
ENDPOINT_KEY = os.environ.get("ENDPOINT_KEY", "")
ENDPOINT_MODEL = os.environ.get("ENDPOINT_MODEL", "DeepSeek-V4-Flash-0731")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gift_Cards.jsonl.gz")
N_REVIEWS = 100
CONCURRENCY = 6
MAX_TOKENS = 300

VALID_LABELS = {"positive", "negative", "neutral"}

SYSTEM_PROMPT = (
    "You are a sentiment-analysis classifier. Read the product review's title and body "
    "and classify its sentiment into exactly one of: positive, negative, or neutral. "
    "Respond with a single word only: 'positive', 'negative', or 'neutral'. "
    "Do not add any other text, explanation, or punctuation."
)


def classify(title: str, text: str) -> str:
    """Return the LLM's label for one review (lowercased, validated)."""
    user_prompt = (
        f"Classify the sentiment of the following product review.\n\n"
        f"TITLE: {title}\nTEXT: {text}\n\nLabel:"
    )
    payload = {
        "model": ENDPOINT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{ENDPOINT_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ENDPOINT_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # vLLM reasoning models may spend tokens reasoning before answering in content.
    content = data["choices"][0]["message"].get("content") or ""
    cleaned = re.sub(r"[^A-Za-z]", "", content).strip().lower()

    # Robust match: find any valid label inside the answer.
    for label in VALID_LABELS:
        if label in cleaned:
            return label
    return "unknown"


def load_reviews(n: int) -> list[dict]:
    """Read the first n review records, keeping only title and text."""
    reviews = []
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rec = json.loads(line)
            reviews.append({
                "title": str(rec.get("title") or ""),
                "text": str(rec.get("text") or ""),
            })
    return reviews


def short(val: str, length: int = 60) -> str:
    val = " ".join(val.split())
    return val if len(val) <= length else val[: length - 1] + "…"


def main() -> None:
    if not ENDPOINT_KEY:
        sys.exit("ENDPOINT_KEY is not set.")

    reviews = load_reviews(N_REVIEWS)
    print(f"Loaded {len(reviews)} reviews from {DATA_FILE}\n")

    results = [None] * len(reviews)

    def run(i: int):
        return i, classify(reviews[i]["title"], reviews[i]["text"])

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(run, i) for i in range(len(reviews))]
        for fut in as_completed(futures):
            i, label = fut.result()
            results[i] = label
            rev = reviews[i]
            print(f"[{i:>3}] {label:<8} | T: {short(rev['title']):<60} | {short(rev['text'], 80)}")

    # Tally.
    from collections import Counter
    counts = Counter(results)
    total = len(results)
    print("\n=== RESULTS ===")
    for label in sorted(VALID_LABELS | {"unknown"}):
        n = counts.get(label, 0)
        print(f"  {label:<10} {n:>3}  ({n / total * 100:5.1f}%)")
    print(f"  {'total':<10} {total:>3}")


if __name__ == "__main__":
    main()
