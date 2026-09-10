"""Re-classify records whose LLM reply was empty/invalid (llm_label == unknown),
then recompute mapped/NRC emotions and save. Uses retry for robustness."""

import json
import os
import time

import nrc_emotions as NRC
from collect_eval_data import VALID_LABELS, OUT_FILE, classify


def classify_with_retry(title, text, attempts=3):
    for _ in range(attempts):
        s, e = classify(title, text)
        if s in VALID_LABELS and e.strip():
            return s, e
        time.sleep(0.5)
    return classify(title, text)  # last attempt, accept whatever


def main():
    with open(OUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    recs = data["records"]

    fixed = 0
    for r in recs:
        if r["llm_label"] in VALID_LABELS and r["llm_emotion_raw"].strip():
            continue
        sent, emo = classify_with_retry(r["title"], r["text"])
        r["llm_label"] = sent
        r["llm_emotion_raw"] = emo
        r["llm_emotion"] = NRC.map_to_nrc(emo)
        r["nrc_emotion"] = NRC.dominant_emotion(r["title"] + " " + r["text"])
        l, nrc = r["llm_emotion"], r["nrc_emotion"]
        r["emotion_match"] = (l == nrc) if (l and nrc) else None
        r["correct"] = r["llm_label"] == r["star_label"]
        fixed += 1
        print(f"  line {r['index']}: -> {sent}/{emo!r} (nrc={r['nrc_emotion']})")

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

    n = len(recs)
    acc = sum(1 for r in recs if r["correct"])
    from collections import Counter
    print(f"\nRe-classified {fixed} records -> {OUT_FILE}")
    print(f"Remaining unknown labels: {sum(1 for r in recs if r['llm_label']=='unknown')}")
    print(f"Sentiment accuracy: {acc}/{n} = {acc/n*100:.1f}%")
    print(f"LLM dist: {dict(Counter(r['llm_label'] for r in recs))}")


if __name__ == "__main__":
    main()
