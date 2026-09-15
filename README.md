# Spam SMS Detection

**CodSoft Machine Learning Internship — Task 4**

An SMS spam classifier built on the SMS Spam Collection Dataset. The script cleans
the raw messages, converts them to TF-IDF features, trains and compares three
classifiers, then saves the best performer to disk and demonstrates it on unseen
messages.

---

## Dataset

The dataset is **not** included in this repository (it's listed in `.gitignore`).
Download it yourself:

**Kaggle:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

1. Open the link above and sign in to Kaggle (a free account is required).
2. Click **Download** — you'll get `archive.zip`.
3. Unzip it. Inside you'll find `spam.csv`.
4. Place `spam.csv` in the **same folder as `spam_sms_detection.py`**.

Your folder should look like this:

```
spam-sms-detection/
├── spam_sms_detection.py
├── requirements.txt
├── README.md
├── .gitignore
└── spam.csv          <- the file you just downloaded
```

**About the file:** 5,572 SMS messages labelled `ham` (legitimate) or `spam`.
It is encoded in **latin-1**, not UTF-8, and ships with three mostly-empty
`Unnamed:` columns — the script handles both automatically.

> Alternative source (no Kaggle account needed):
> https://archive.ics.uci.edu/dataset/228/sms+spam+collection
> — note this version is a tab-separated `.txt` file, so you'd need to convert it
> to a two-column `spam.csv` first.

---

## Installation

Requires Python 3.8 or newer.

```bash
pip install -r requirements.txt
```

Optionally, use a virtual environment first:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## Running

```bash
python spam_sms_detection.py
```

The script runs end to end in a few seconds — no GPU, no internet needed.

---

## What it does

| Step | Detail |
|------|--------|
| **Load** | Reads `spam.csv` with latin-1 encoding, keeps the first two columns as `label` / `message`, drops NA rows and duplicates |
| **Clean** | Lowercases, strips URLs, emails, digits and punctuation, collapses whitespace |
| **Encode** | Maps `ham` → 0, `spam` → 1 |
| **Split** | 80 / 20 train-test, stratified, `random_state=42` |
| **Vectorize** | `TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_features=5000)` — fitted on the training set only |
| **Train** | Multinomial Naive Bayes, Logistic Regression (`class_weight="balanced"`), Linear SVC (`class_weight="balanced"`) |
| **Evaluate** | Accuracy, precision, recall, F1 and a full `classification_report` per model |
| **Select** | Picks the model with the highest **F1 on the spam class** |
| **Save** | Writes the confusion matrix image and the model + vectorizer bundle |
| **Demo** | Reloads the saved bundle from disk and classifies 4 fresh sample messages |

### Why F1 and not accuracy?

The dataset is roughly 87% ham. A model that blindly predicts "ham" every single
time would score **87% accuracy** while catching zero spam. F1 balances
*precision* (how many flagged messages really were spam) against *recall* (how
much of the real spam we caught), which is the metric that actually reflects a
useful spam filter.

---

## Output files

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix of the best-performing model |
| `spam_model.joblib` | The trained model **and** its fitted TF-IDF vectorizer, saved together |

Both are gitignored — rerun the script to regenerate them.

### Using the saved model elsewhere

```python
import joblib
from spam_sms_detection import clean_text

bundle = joblib.load("spam_model.joblib")
model, vectorizer = bundle["model"], bundle["vectorizer"]

msg = "Congratulations! You've won a free cruise. Call now!"
prediction = model.predict(vectorizer.transform([clean_text(msg)]))[0]
print("SPAM" if prediction == 1 else "HAM")
```

The model and vectorizer must always be loaded as a pair — the model's weights
are indexed by that exact vocabulary.

---

## Troubleshooting

**`Could not find 'spam.csv'`** — the file isn't in the same folder as the script,
or it's still inside `archive.zip`. The error message prints the exact folder the
script searched.

**`UnicodeDecodeError`** — you edited the encoding. The dataset must be read with
`encoding="latin-1"`.

**`ModuleNotFoundError`** — run `pip install -r requirements.txt`. If you're using
a virtual environment, make sure it's activated.
