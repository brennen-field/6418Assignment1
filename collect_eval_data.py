"""Classify a STRATIFIED random sample of Amazon Gift Cards reviews.

Sample: an even number from each star-truth sentiment class — 50 positive
(4-5 star), 50 neutral (3 star), 50 negative (1-2 star) = 150 total — chosen
randomly WITHIN each class using a reproducible seed.

Each review (title+text only) is labeled by the LLM with a sentiment
(positive/negative/neutral) and a free-form primary emotion. The emotion is
mapped to the 8 NRC categories where possible, and the review text is also
scored against the NRC Emotion Lexicon to derive its dominant emotion.

Output: stratified150_eval.json as {"meta": {...}, "records": [...]}
Per-record fields:
  index, title, text, rating, star_label, llm_label (sentiment),
  llm_emotion_raw, llm_emotion (mapped to NRC or None), nrc_emotion,
  emotion_match (True/False/None), correct

Config via env: ENDPOINT_URL, ENDPOINT_KEY, ENDPOINT_MODEL.
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

import nrc_emotions as NRC

ENDPOINT_URL = os.environ.get("ENDPOINT_URL", "http://dobolyi.com:9000/v1")
ENDPOINT_KEY = os.environ.get("ENDPOINT_KEY", "")
ENDPOINT_MODEL = os.environ.get("ENDPOINT_MODEL", "DeepSeek-V4-Flash-0731")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gift_Cards.jsonl.gz")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stratified150_eval.json")

CLASSES = ["positive", "neutral", "negative"]
VALID_LABELS = set(CLASSES)
CONCURRENCY = 6
MAX_TOKENS = 300

SYSTEM_PROMPT = (
    "You are a sentiment and emotion classifier for product reviews. "
    "Given a review's title and body, respond with a JSON object with exactly two keys: "
    "\"sentiment\" (one of: positive, negative, neutral) and \"emotion\" (the single "
    "primary emotion the reviewer expresses, e.g. joy, excitement, disappointment, "
    "anger, gratitude, relief, surprise — use whatever word fits best). "
    "Return ONLY the JSON object, no other text, e.g. {\"sentiment\":\"positive\",\"emotion\":\"joy\"}."
)


def star_to_label(rating: float) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def classify(title: str, text: str):
    user_prompt = (
        f"Classify the sentiment and primary emotion of this product review.\n\n"
        f"TITLE: {title}\nTEXT: {text}\n\n"
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
    return parse_llm_reply(data["choices"][0]["message"].get("content") or "")


def parse_llm_reply(content: str):
    content = content or ""
    m = re.search(r'\{\s*"sentiment"\s*:\s*"([^"]+)"\s*,\s*"emotion"\s*:\s*"([^"]+)"\s*\}', content)
    if not m:
        m = re.search(r'\{\s*"emotion"\s*:\s*"([^"]+)"\s*,\s*"sentiment"\s*:\s*"([^"]+)"\s*\}', content)
    if m:
        if m.group(1).lower() in VALID_LABELS:
            sentiment, emotion = m.group(1), m.group(2)
        else:
            sentiment, emotion = m.group(2), m.group(1)
    else:
        sents = sorted(VALID_LABELS, key=len, reverse=True)
        sentiment = next((s for s in sents if re.search(rf"\b{s}\b", content)), "unknown")
        emotion_m = re.search(r'"emotion"\s*:\s*"([^"]+)"', content)
        emotion = emotion_m.group(1) if emotion_m else content.strip()
    return sentiment.lower(), emotion.strip().lower()


def stratified_sample(seed: int, per_class: int) -> list[int]:
    """Return a reproducible list of line indices: `per_class` randomly chosen
    from each star-truth class."""
    buckets = {c: [] for c in CLASSES}
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rating = float(json.loads(line).get("rating"))
            buckets[star_to_label(rating)].append(i)

    chosen = []
    rng = random.Random(seed)
    for c in CLASSES:
        pool = buckets[c]
        k = min(per_class, len(pool))
        chosen.extend(rng.sample(pool, k))
        print(f"  {c:<9}: {len(pool):,} available, sampled {k}")
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect stratified sentiment+emotion eval data")
    parser.add_argument("--per-class", type=int, default=50, help="reviews per class")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    if not ENDPOINT_KEY:
        sys.exit("ENDPOINT_KEY is not set.")

    chosen = stratified_sample(args.seed, args.per_class)
    n = len(chosen)

    # Load just the chosen records' fields.
    rec_by_line = {}
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in chosen:
                rec = json.loads(line)
                rec_by_line[i] = {
                    "title": str(rec.get("title") or ""),
                    "text": str(rec.get("text") or ""),
                    "rating": float(rec.get("rating")),
                }

    records = []
    for lineno in chosen:
        r = rec_by_line[lineno]
        records.append({
            "index": lineno,
            "title": r["title"],
            "text": r["text"],
            "rating": r["rating"],
            "star_label": star_to_label(r["rating"]),
        })

    def run(i):
        return i, classify(records[i]["title"], records[i]["text"])

    preds = [None] * len(records)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(run, i) for i in range(len(records))]
        for fut in as_completed(futures):
            i, (sent, emo) = fut.result()
            preds[i] = (sent, emo)

    for rec, (sent, emo_raw) in zip(records, preds):
        rec["llm_label"] = sent
        rec["llm_emotion_raw"] = emo_raw
        rec["llm_emotion"] = NRC.map_to_nrc(emo_raw)
        rec["nrc_emotion"] = NRC.dominant_emotion(rec["title"] + " " + rec["text"])
        l, nrc = rec["llm_emotion"], rec["nrc_emotion"]
        rec["emotion_match"] = (l == nrc) if (l and nrc) else None
        rec["correct"] = rec["llm_label"] == rec["star_label"]

    meta = {
        "sample": "stratified",
        "description": f"Stratified random sample of {n} reviews "
                       f"({args.per_class} positive / {args.per_class} neutral / "
                       f"{args.per_class} negative), seed {args.seed}",
        "n": n,
        "per_class": args.per_class,
        "seed": args.seed,
        "model": ENDPOINT_MODEL,
    }

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "records": records}, fh, ensure_ascii=False, indent=2)

    acc = sum(1 for r in records if r["correct"])
    emo_matches = sum(1 for r in records if r["emotion_match"] is True)
    emo_compare = sum(1 for r in records if r["emotion_match"] is not None)
    print(f"\nSaved {len(records)} evaluated reviews -> {OUT_FILE}")
    print(f"Sentiment accuracy (vs star-truth): {acc}/{len(records)} = {acc / len(records) * 100:.1f}%")
    print(f"Star-truth dist: {dict(Counter(r['star_label'] for r in records))}")
    print(f"LLM dist:        {dict(Counter(r['llm_label'] for r in records))}")
    if emo_compare:
        print(f"Emotion match (LLM vs NRC): {emo_matches}/{emo_compare} = {emo_matches / emo_compare * 100:.1f}%"
              f"  (comparable; {len(records) - emo_compare} non-comparable)")
    unmapped = sorted({r["llm_emotion_raw"] for r in records if r["llm_emotion"] is None})
    print(f"Unmapped LLM emotions ({len(unmapped)}): {unmapped}")


if __name__ == "__main__":
    main()
