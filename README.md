# 6418Assignment1 — LLM Sentiment Classifier for Amazon Gift Card Reviews

Classify Amazon **Gift Cards** review sentiment using a large language model, then
evaluate the results against the reviewers' 1–5 star ratings.

## Overview

This project uses the **Amazon 2023 Reviews dataset** (McAuley Lab), restricted to the
**Gift Cards** category (`Gift_Cards.jsonl.gz`, ~152,410 reviews). A local
vLLM–DeepSeek model — accessed through an **OpenAI-compatible endpoint** — labels each
review's sentiment from its **title + text only**. The star rating is never used for
classification; it is held out strictly to *test* the classifier.

Class mapping used for ground-truth evaluation:

| Star rating | Class |
|---:|---|
| 4 – 5 | `positive` |
| 3 | `neutral` |
| 1 – 2 | `negative` |

## Results

On a reproducible random sample of **100 reviews** (seed `42`):

- **Accuracy ≈ 90–91%**
- Confusion matrix, per-class precision / recall / F1, and every per-review
  prediction vs. star rating are presented in the interactive dashboard.

## Repository contents

| File | Purpose |
|---|---|
| `sentiment_classifier.py` | Classify the first *N* reviews (title+text) via the LLM as positive/negative/neutral |
| `evaluate_sentiment.py` | Classify a sample and compare LLM labels against star-rating ground truth (accuracy, confusion matrix, precision/recall/F1) |
| `collect_eval_data.py` | Re-classify the reproducible random sample (seed 42) and save the evaluated data to `random100_eval.json` |
| `render_dashboard.py` | Build the self-contained `dashboard.html` from `random100_eval.json` |
| `dashboard.html` | Interactive, self-contained dashboard of the evaluation (no external dependencies) |
| `random100_eval.json` | Evaluated data for the random-100 sample (line #, title, text, rating, star-truth label, LLM label, correct?) |
| `Gift_Cards.jsonl.gz` | Amazon 2023 review dataset — Gift Cards category |

## Setup

Requires **Python 3.10+** (standard library only — no `pip install`s needed).

Set the endpoint details as environment variables:

```bash
export ENDPOINT_URL=http://<host>:9000/v1
export ENDPOINT_KEY=<your-api-key>
export ENDPOINT_MODEL=DeepSeek-V4-Flash-0731
```

On Windows (PowerShell):

```powershell
$env:ENDPOINT_URL="http://<host>:9000/v1"
$env:ENDPOINT_KEY="<your-api-key>"
$env:ENDPOINT_MODEL="DeepSeek-V4-Flash-0731"
```

## Usage

**Classify the first 100 reviews:**

```bash
python sentiment_classifier.py
```

**Evaluate a random sample against star ratings (reproducible with `--seed`):**

```bash
python evaluate_sentiment.py --n 100 --seed 42
```

**Rebuild the dashboard data + page:**

```bash
python collect_eval_data.py     # writes random100_eval.json
python render_dashboard.py      # writes dashboard.html
```

Then open `dashboard.html` in any browser.
