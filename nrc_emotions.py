"""Utilities for the NRC Emotion Lexicon (EmoLex) v0.92.

- Load the lexicon into a {word: set-of-emotions} mapping.
- score_emotions(text) -> Counter of NRC emotions triggered by the text's words.
- dominant_emotion(text) -> the single strongest NRC emotion, or None.
- map_to_nrc(emotion_word) -> map a free-form emotion label to one of the 8
  NRC basic emotions, or None if it cannot be mapped.

The 8 NRC basic emotions: anger, fear, anticipation, trust, surprise,
sadness, joy, disgust.
"""

import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(HERE, "nrc", "NRC-lexicon-v0.92.txt")

EMOTIONS = ["anger", "fear", "anticipation", "trust", "surprise", "sadness", "joy", "disgust"]

_TOKEN_RE = re.compile(r"[a-z']+")

# Tie-break order when two emotions tie for the dominant count.
_ORDER = {"joy": 0, "trust": 1, "anticipation": 2, "surprise": 3,
          "sadness": 4, "fear": 5, "anger": 6, "disgust": 7}


def load_lexicon(path: str = LEXICON_PATH) -> dict:
    """Return {word: set(emotions)} taking only true (1) associations for the 8
    basic emotions (ignoring the positive/negative sentiment columns)."""
    lex = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion not in EMOTIONS or flag != "1":
                continue
            lex.setdefault(word, set()).add(emotion)
    return lex


_LEXICON = load_lexicon()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip HTML entities, and pull out word tokens."""
    if not text:
        return []
    text = html.unescape(text).lower()
    return _TOKEN_RE.findall(text)


def score_emotions(text: str) -> dict:
    """Count how many words in `text` associate with each NRC emotion."""
    counts = {e: 0 for e in EMOTIONS}
    for tok in tokenize(text):
        for e in _LEXICON.get(tok, ()):
            counts[e] += 1
    return counts


def dominant_emotion(text: str):
    """Return the NRC emotion with the highest word count in `text`, or None.
    Ties are broken by a fixed order."""
    counts = score_emotions(text)
    best_emotion = None
    best_count = 0
    for e in EMOTIONS:
        if counts[e] > best_count:
            best_emotion, best_count = e, counts[e]
        elif counts[e] == best_count and best_emotion is not None:
            # stable tie-break: pick the earlier in _ORDER
            if _ORDER[e] < _ORDER[best_emotion]:
                best_emotion = e
    return best_emotion if best_count > 0 else None


# Synonym map from common free-form emotion labels to the 8 NRC categories.
_SYNONYMS = {
    "joy": ["happy", "happiness", "pleased", "pleasure", "glad", "delighted", "delight",
            "great", "good", "wonderful", "excellent", "amazing", "awesome", "fantastic",
            "love", "loved", "thrilled", "thrill", "ecstatic", "joyful", "joyous",
            "overjoyed", "cheerful", "cheer", "content", "contentment", "satisfied",
 "satisfaction", "grateful", "gratitude", "thankful", "amused", "positive",
 "amusement", "relief", "relieved", "joyous"],
    "sadness": ["sad", "unhappy", "disappointed", "disappointment", "sorrow", "grief",
                "down", "upset", "hopeless", "depressed", "depression", "miserable",
                "misery", "regret", "regretful", "lonely", "loneliness", "heartbroken",
                "sadness", "crying", "cry", "gloomy", "melancholy", "blue", "glum",
                "devastated", "hurt"],
    "fear": ["fear", "afraid", "scared", "frightened", "fright", "terrified", "terror",
             "anxious", "anxiety", "nervous", "nervousness", "worry", "worried", "concern",
             "dread", "panic", "uneasy", "alarmed", "alarm", "fearful", "scary", "worrisome"],
    "anger": ["anger", "angry", "mad", "furious", "fury", "irritated", "irritation",
              "annoyed", "annoyance", "frustrated", "frustration", "rage", "resent",
              "resentment", "aggravated", "agitation", "angered", "irked", "infuriated"],
    "disgust": ["disgust", "disgusted", "repulsed", "repulsive", "gross", "grossed",
                "distaste", "distasteful", "contempt", "revulsion", "aversion", "nauseated",
                "sickened", "hate", "hatred", "revulsion"],
    "trust": ["trust", "trusted", "trusting", "confident", "confidence", "reliable",
              "reliability", "safety", "safe", "comfort", "comfortable", "comforted",
              "assured", "assurance", "faithful", "faith", "loyal", "loyalty", "secure",
              "calm", "reassured", "content", "approval", "approving"],
    "anticipation": ["anticipation", "expect", "expected", "expectant", "excited",
                     "excitement", "eager", "eagerness", "curiosity", "curious",
                     "interest", "interested", "hopeful", "hope", "hopes", "enthusiastic",
                     "enthusiasm", "lookingforward", "lookforward", "awaiting", "impatient",
                     "optimistic", "optimism"],
    "surprise": ["surprise", "surprised", "shocked", "shock", "amazed", "amazement",
                 "astonished", "astonishment", "unexpected", "wonder", "awed", "startled",
                 "stunned", "surprising"],
}


def map_to_nrc(emotion_word: str):
    """Map a free-form emotion label (e.g. 'excited', 'disappointed') to one of
    the 8 NRC basic emotions, or return None if unmappable."""
    if not emotion_word:
        return None
    w = emotion_word.strip().lower().replace(" ", "")
    if w in EMOTIONS:
        return w
    for emo, syns in _SYNONYMS.items():
        if w in syns:
            return emo
    # Substring fallback: does the token contain a known emotion stem?
    for emo in EMOTIONS:
        if emo in w or w in emo:
            return emo
    return None


if __name__ == "__main__":
    print(f"Lexicon size: {len(_LEXICON):,} words")
    samples = [
        "I love this gift card, it makes me so happy and excited!",
        "Terrible product, I am so angry and frustrated.",
        "The gift card was fine, nothing special.",
    ]
    for s in samples:
        print(f"\nTEXT: {s}")
        print("  scores:", score_emotions(s))
        print("  dominant:", dominant_emotion(s))
