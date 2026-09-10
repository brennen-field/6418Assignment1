"""Re-run LLM classification on the reproducible random sample (seed 42, n=100)
using ONLY title+text, capture the star-rating ground truth, and save the full
evaluated dataset to random100_eval.json for the dashboard.

Config via env: ENDPOINT_URL, ENDPOINT_KEY, ENDPOINT_MODEL.
"""

import gzip
import json
import os
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://dobolyi.com:9000/v1")
ENDPOINT_KEY = os.environ.get("ENDPOINT_KEY", "")
ENDPOINT_MODEL = os.environ.get("ENDPOINT_MODEL", "DeepSeek-V4-Flash-0731")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gift_Cards.jsonl.gz")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random100_eval.json")

SEED = 42
N = 100
CONCURRENCY = 6
MAX_TOKENS = 300

VALID_LABELS = {"positive", "negative", "neutral"}
CLASSES = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = (
    "You are a sentiment-analysis classifier. Read the product review's title and body "
    "and classify its sentiment into exactly one of: positive, negative, or neutral. "
    "Respond with a single word only: 'positive', 'negative', or 'neutral'. "
    "Do not add any other text, explanation, or punctuation."
)


def star_to_label(rating: float) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def classify(title: str, text: str) -> str:
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
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {ENDPOINT_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"].get("content") or ""
    cleaned = re.sub(r"[^A-Za-z]", "", content).strip().lower()
    for label in VALID_LABELS:
        if label in cleaned:
            return label
    return "unknown"


def main() -> None:
    if not ENDPOINT_KEY:
        sys.exit("ENDPOINT_KEY is not set.")

    # Load the reproducible sample.
    all_recs = []
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            all_recs.append({
                "title": str(rec.get("title") or ""),
                "text": str(rec.get("text") or ""),
                "rating": float(rec.get("rating")),
            })
    chosen = sorted(random.Random(SEED).sample(range(len(all_recs)), N))
    print(f"Seed {SEED}: sampled {len(chosen)} of {len(all_recs)} records")

    records = []
    for lineno in chosen:
        r = all_recs[lineno]
        records.append({
            "index": lineno,
            "title": r["title"],
            "text": r["text"],
            "rating": r["rating"],
            "star_label": star_to_label(r["rating"]),
        })

    # Classify each with title+text ONLY.
    def run(i):
        return i, classify(records[i]["title"], records[i]["text"])

    preds = [None] * len(records)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(run, i) for i in range(len(records))]
        for fut in as_completed(futures):
            i, label = fut.result()
            preds[i] = label

    for rec, p in zip(records, preds):
        rec["llm_label"] = p
        rec["correct"] = rec["llm_label"] == rec["star_label"]

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    # Quick console summary.
    from collections import Counter
    acc = sum(1 for r in records if r["correct"])
    print(f"Saved {len(records)} evaluated reviews -> {OUT_FILE}")
    print(f"Accuracy: {acc}/{len(records)} = {acc / len(records) * 100:.1f}%")
    print("Star-truth dist:", dict(Counter(r["star_label"] for r in records)))
    print("LLM dist:       ", dict(Counter(r["llm_label"] for r in records)))


if __name__ == "__main__":
    main()
