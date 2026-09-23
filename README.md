# CodSoft Machine Learning Internship

Task submissions for the CodSoft Internship Program (CIP), September batch **C22**.

## Tasks

| # | Task | Folder | Status |
|---|------|--------|--------|
| 4 | Spam SMS Detection | [`task4_spam_sms/`](task4_spam_sms/) | ✅ Complete |

## Task 4 — Spam SMS Detection

Classifies an SMS message as spam or legitimate (ham).

- **Features:** TF-IDF over unigrams + bigrams, with URLs, emails, phone numbers and currency amounts replaced by placeholder tokens rather than deleted
- **Models compared:** Multinomial Naive Bayes, Logistic Regression, Linear SVC
- **Selection:** 5-fold cross-validated F1 on the training set, so the test set is used exactly once
- **Winner:** Linear SVC — 98.3% accuracy, 94.5% precision, 91.6% recall, 93.0% F1
- **App:** Streamlit UI with single-message scanning, bulk CSV scanning, an adjustable decision threshold, and a view of the learned term weights
- **Tests:** 61 pytest tests, including headless UI tests via `st.testing.v1.AppTest`

```bash
cd task4_spam_sms
pip install -r requirements.txt
python spam_sms_detection.py   # train -> writes spam_model.joblib
streamlit run app.py           # open the web UI
pytest                         # run the test suite
```

See [`task4_spam_sms/README.md`](task4_spam_sms/README.md) for the full write-up.

## Repository layout

Each task lives in its own folder with its own `README.md` and `requirements.txt`,
so they can be run independently. A single virtual environment at the repository
root is enough for all of them.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
```

---

**Author:** ADjayantan
