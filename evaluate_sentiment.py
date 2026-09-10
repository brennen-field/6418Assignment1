"""Evaluate the LLM sentiment classifier against star-rating ground truth.

The LLM classifies each review using ONLY title+text. The star rating is used
ONLY as ground truth for evaluation:
    rating 4-5 -> positive
    rating 1-2 -> negative
    rating 3   -> neutral
Prints accuracy, a confusion matrix, and per-class precision/recall/F1, plus
all mismatches. No files are written.

Config via environment variables (with safe local defaults):
    ENDPOINT_URL, ENDPOINT_KEY, ENDPOINT_MODEL
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://dobolyi.com:9000/v1")
ENDPOINT_KEY = os.environ.get("ENDPOINT_KEY", "")
ENDPOINT_MODEL = os.environ.get("ENDPOINT_MODEL", "DeepSeek-V4-Flash-0731")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gift_Cards.jsonl.gz")
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
    return "neutral"  # rating == 3


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
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ENDPOINT_KEY}",
        },
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


def load_reviews(n: int, seed: int | None = None) -> tuple[list[dict], list[int]]:
    """Load review records. With seed, randomly sample n records from the whole
    dataset (reproducible); otherwise take the first n. Returns (reviews, indices)."""
    all_recs = []
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            all_recs.append({
                "title": str(rec.get("title") or ""),
                "text": str(rec.get("text") or ""),
                "rating": float(rec.get("rating")),
            })

    total = len(all_recs)
    if seed is None:
        chosen = list(range(n))
    else:
        chosen = random.Random(seed).sample(range(total), n)
        print(f"Seed {seed}: sampled {n} of {total} records")
    # Enforce same order as the original file for readability of indices.
    chosen.sort()
    return [all_recs[i] for i in chosen], chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM sentiment classifier vs star-rating ground truth.")
    parser.add_argument("--n", type=int, default=100, help="number of reviews to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for a reproducible random sample; defaults to first N")
    args = parser.parse_args()

    if not ENDPOINT_KEY:
        sys.exit("ENDPOINT_KEY is not set.")

    reviews, indices = load_reviews(args.n, args.seed)

    # ---- LLM classification (title+text only) ----
    preds = [None] * len(reviews)
    def run(i):
        return i, classify(reviews[i]["title"], reviews[i]["text"])
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(run, i) for i in range(len(reviews))]
        for fut in as_completed(futures):
            i, label = fut.result()
            preds[i] = label

    # ---- Ground truth from star rating ----
    truths = [star_to_label(r["rating"]) for r in reviews]

    # ---- Metrics ----
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    total = len(reviews)
    acc = correct / total
    print(f"Reviews evaluated: {total}\n")
    print(f"ACCURACY: {correct}/{total} = {acc * 100:.1f}%\n")

    # Confusion matrix.
    print("CONFUSION MATRIX (rows=truth/star, cols=prediction/LLM):")
    header = "          " + "".join(f"{c:>12}" for c in CLASSES + ["unknown"])
    print(header)
    for t in CLASSES:
        row = f"{t:<10}"
        for c in CLASSES + ["unknown"]:
            row += f"{sum(1 for p, tt in zip(preds, truths) if tt == t and p == c):>12}"
        print(row)

    # Per-class precision, recall, F1.
    print("\nPER-CLASS METRICS (LLM prediction vs star-truth):")
    print(f"{'class':<10}{'prec':>8}{'rec':>8}{'f1':>8}")
    for cls in CLASSES:
        tp = sum(1 for p, t in zip(preds, truths) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, truths) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, truths) if p != cls and t == cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{cls:<10}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}")

    # Class distribution sanity check.
    print("\nCLASS DISTRIBUTION (star-truth):")
    for cls in CLASSES:
        print(f"  {cls:<10} {truths.count(cls)}")
    print(f"  {'unknown':<10} {preds.count('unknown')} (LLM outputs)")

    # List mismatches.
    print("\nMISMATCHES (LLM prediction differs from star-truth):")
    n_mis = 0
    for local_i, (p, t, r) in enumerate(zip(preds, truths, reviews)):
        if p != t:
            n_mis += 1
            snippet = " ".join(r["text"].split())[:90]
            print(f"  [line {indices[local_i]}] star={r['rating']}:0 (star->{t}), LLM->{p} | {snippet}")
    if n_mis == 0:
        print("  none")
    print(f"\nTotal mismatches: {n_mis}/{total}")


if __name__ == "__main__":
    main()
