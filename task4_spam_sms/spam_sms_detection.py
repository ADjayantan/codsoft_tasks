"""
Spam SMS Detection - CodSoft Machine Learning Internship (Task 4)
=================================================================

Builds a classifier that labels an SMS message as SPAM or HAM (legitimate).

Pipeline
--------
    load -> clean -> split -> TF-IDF vectorize -> train 3 models -> compare -> save best

Models compared: Multinomial Naive Bayes, Logistic Regression, Linear SVC.
The winner is chosen by 5-fold cross-validated F1 on the TRAINING set (the test
set is only used once, for the final report) and persisted to spam_model.joblib.

Usage
-----
    python spam_sms_detection.py

Requires spam.csv (SMS Spam Collection Dataset) in the same folder.
"""

import os
import re
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # render straight to a file, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_FILE = "spam.csv"
MODEL_FILE = "spam_model.joblib"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

LABEL_MAP = {"ham": 0, "spam": 1}
CLASS_NAMES = ["HAM", "SPAM"]


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #

def load_dataset(path=DATA_FILE):
    """Read spam.csv, keep the two useful columns, drop NA rows and duplicates.

    The Kaggle CSV ships with three trailing 'Unnamed: N' columns that are almost
    entirely empty, so we slice the first two positionally rather than by name --
    header names differ between mirrors of this dataset (v1/v2 vs Category/Message).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "\nCould not find '{}' in {}\n\n"
            "Download the SMS Spam Collection Dataset from:\n"
            "    https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n\n"
            "Unzip it and place spam.csv next to this script, then run again.\n".format(
                path, os.getcwd()
            )
        )

    # latin-1: the file contains bytes that are not valid UTF-8 (pound signs, etc.)
    df = pd.read_csv(path, encoding="latin-1")

    df = df.iloc[:, :2]                          # discard the empty trailing columns
    df.columns = ["label", "message"]

    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()

    print("[data] loaded {} rows -> {} after dropping NA + duplicates".format(before, len(df)))
    print("[data] class balance:")
    print(df["label"].value_counts().to_string())
    print()
    return df


# --------------------------------------------------------------------------- #
# 2. Text cleaning
# --------------------------------------------------------------------------- #

def clean_text(text):
    """Normalise a single SMS message into lowercase words + placeholder tokens.

    Spam signals like URLs, phone numbers, prize amounts and currency symbols are
    REPLACED with placeholder words instead of deleted -- "call 09061701461 to claim
    £900" becomes "call longnumtoken to claim currencytoken numtoken". Deleting them
    throws away some of the strongest spam evidence in the dataset.

    Order matters: URLs/emails go first, otherwise 'http://x.com' would decay into
    the meaningless tokens 'http' and 'x com'.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)     # URLs
    text = re.sub(r"\S+@\S+", " emailtoken ", text)             # email addresses
    text = re.sub(r"[£$€]", " currencytoken ", text)            # £ $ €
    text = re.sub(r"\d{5,}", " longnumtoken ", text)            # phone / short codes
    text = re.sub(r"\d+", " numtoken ", text)                   # any other number
    text = re.sub(r"[^a-z\s]", " ", text)                       # punctuation, symbols
    text = re.sub(r"\s+", " ", text)                            # collapse whitespace
    return text.strip()


def prepare_features(df):
    """Add a cleaned-text column and a numeric target column (ham=0, spam=1)."""
    df = df.copy()
    df["clean_message"] = df["message"].apply(clean_text)
    df["target"] = df["label"].str.strip().str.lower().map(LABEL_MAP)

    # A label outside {ham, spam} becomes NaN above; drop those rows rather than crash.
    unmapped = int(df["target"].isna().sum())
    if unmapped:
        print("[warn] dropping {} row(s) with an unrecognised label".format(unmapped))
        df = df.dropna(subset=["target"])

    df["target"] = df["target"].astype(int)

    # Cleaning can empty a message entirely (e.g. one that was only digits/emoji).
    df = df[df["clean_message"].str.len() > 0]
    return df


# --------------------------------------------------------------------------- #
# 3. Split and vectorize
# --------------------------------------------------------------------------- #

def split_data(df):
    """80/20 split, stratified so both sides keep the same spam ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )
    print("[split] train={}  test={}\n".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test


def build_vectorizer():
    """TF-IDF over unigrams + bigrams.

    No stop-word removal on purpose: scikit-learn's English stop list contains
    "call", "now", "please", "get" -- exactly the words spam is written in. With
    stop words removed, the bigram "call now" can never even exist.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams ("call now", "claim prize")
        min_df=2,             # ignore terms that appear in only one message
        sublinear_tf=True,    # log-scale term counts so "free free free" != 3x
    )


def vectorize(X_train, X_test):
    """Turn text into TF-IDF vectors.

    Fitted on the TRAINING set only -- fitting on everything would leak test-set
    vocabulary and IDF statistics into training and inflate the scores.
    """
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[tfidf] vocabulary size: {}\n".format(len(vectorizer.vocabulary_)))
    return X_train_vec, X_test_vec, vectorizer


# --------------------------------------------------------------------------- #
# 4. Training and evaluation
# --------------------------------------------------------------------------- #

def build_models():
    """The three classifiers we are comparing."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Linear SVC": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    }


def evaluate_model(name, model, X_train_vec, y_train, X_test_vec, y_test):
    """Fit one model and report its test-set scores. Returns a result dict."""
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # pos_label=1 -> every score below describes the SPAM class specifically,
    # which is what actually matters on an imbalanced dataset.
    scores = {
        "name": name,
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "y_pred": y_pred,
    }

    print("=" * 66)
    print("  " + name)
    print("=" * 66)
    print("  Accuracy : {:.4f}".format(scores["accuracy"]))
    print("  Precision: {:.4f}   (of messages flagged spam, how many really were)".format(
        scores["precision"]))
    print("  Recall   : {:.4f}   (of all real spam, how much we caught)".format(
        scores["recall"]))
    print("  F1 score : {:.4f}".format(scores["f1"]))
    print()
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return scores


def cross_validate_models(X_train, y_train):
    """5-fold stratified CV F1 on the training set, one score per model.

    Each fold re-fits its own vectorizer inside a pipeline, so no fold ever sees
    the vocabulary of the fold it's being scored on. This -- not the test set --
    is what picks the winner; picking by test score would quietly overfit to it.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, model in build_models().items():
        pipeline = make_pipeline(build_vectorizer(), clone(model))
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        cv_scores[name] = (scores.mean(), scores.std())
    return cv_scores


def compare_models(X_train, y_train, X_train_vec, X_test_vec, y_test):
    """CV every model, evaluate each once on test, print a summary table."""
    cv_scores = cross_validate_models(X_train, y_train)

    results = []
    for name, model in build_models().items():
        r = evaluate_model(name, model, X_train_vec, y_train, X_test_vec, y_test)
        r["cv_f1"], r["cv_std"] = cv_scores[name]
        results.append(r)

    print("=" * 74)
    print("  SUMMARY (metrics for the SPAM class)")
    print("=" * 74)
    print("  {:<26}{:>14}{:>8}{:>8}{:>8}{:>8}".format(
        "Model", "CV F1 (5-fold)", "Acc", "Prec", "Recall", "F1"))
    for r in results:
        print("  {:<26}{:>8.4f}±{:.3f}{:>8.4f}{:>8.4f}{:>8.4f}{:>8.4f}".format(
            r["name"], r["cv_f1"], r["cv_std"],
            r["accuracy"], r["precision"], r["recall"], r["f1"]))
    print()
    return results


# --------------------------------------------------------------------------- #
# 5. Reporting and persistence
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(y_test, y_pred, model_name, path=CONFUSION_MATRIX_FILE):
    """Save a labelled confusion matrix image for the winning model."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.75))
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - {}".format(model_name))

    # Annotate each cell; flip text colour on dark squares so it stays readable.
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("[save] confusion matrix -> {}".format(path))


def save_best_model(best, vectorizer, path=MODEL_FILE):
    """Persist the winning model together with its vectorizer.

    They must travel as a pair: the model's coefficients are indexed by this exact
    vectorizer's vocabulary, so a re-fitted vectorizer would scramble the inputs.
    """
    joblib.dump(
        {"model": best["model"], "vectorizer": vectorizer, "model_name": best["name"]},
        path,
    )
    print("[save] best model '{}' -> {}".format(best["name"], path))


# --------------------------------------------------------------------------- #
# 6. Demo - reload the saved model and classify fresh messages
# --------------------------------------------------------------------------- #

SAMPLE_MESSAGES = [
    "WINNER!! You have been selected to receive a 900 prize reward! "
    "Call 09061701461 now to claim. Offer valid 12 hours only.",
    "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "URGENT! Your mobile number has won 2000 pounds. Text WIN to 80086 to collect your cash.",
    "Sorry I missed your call, I was in a lecture. Will ring you back this evening.",
]


def run_demo(path=MODEL_FILE):
    """Reload the persisted model from disk and predict on sample messages.

    Loading from disk (rather than reusing the in-memory object) proves the saved
    artifact is genuinely self-sufficient -- exactly what a real app would do.
    """
    bundle = joblib.load(path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    print()
    print("=" * 66)
    print("  DEMO - predictions from the saved model ({})".format(bundle["model_name"]))
    print("=" * 66)

    cleaned = [clean_text(m) for m in SAMPLE_MESSAGES]
    predictions = model.predict(vectorizer.transform(cleaned))

    for message, prediction in zip(SAMPLE_MESSAGES, predictions):
        verdict = CLASS_NAMES[prediction]
        preview = message if len(message) <= 70 else message[:67] + "..."
        print("  [{:>4}]  {}".format(verdict, preview))
    print()


# --------------------------------------------------------------------------- #
# 7. Feature inspection (which words gave the spam away?)
# --------------------------------------------------------------------------- #

def top_spam_indicators(model, vectorizer, n=15):
    """Return the n terms that push a message hardest toward SPAM.

    Every model here is linear over the TF-IDF vocabulary, so each term has one
    number attached to it -- but that number lives in a different attribute and
    means a different thing depending on the model:

      * LogisticRegression / LinearSVC -> model.coef_[0][i]
            a signed weight; large positive means "this term argues for spam".
      * MultinomialNB -> model.feature_log_prob_[1][i] and [0][i]
            log P(term | spam) and log P(term | ham). A raw spam log-prob is NOT
            a good ranking on its own, because common words score high in both
            classes -- the difference between the two rows is what discriminates.

    Term names come from vectorizer.get_feature_names_out(), aligned by index.

    Returns: list of (term, score) tuples, highest score first. Return [] if the
    model type isn't recognised.
    """
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        # LogisticRegression / LinearSVC: one signed weight per term, already on a
        # scale where "more positive" literally means "pushes the decision to spam".
        scores = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        # MultinomialNB: the log-ratio log P(term|spam) - log P(term|ham).
        # Ranking by the spam row alone would just surface words that are common
        # everywhere; subtracting the ham row leaves only what discriminates.
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        return []

    # argsort is ascending, so the strongest terms are the last n -- take that tail
    # and reverse it to get highest-first.
    top_indices = np.argsort(scores)[-n:][::-1]
    return [(str(feature_names[i]), float(scores[i])) for i in top_indices]


def print_top_indicators(model, vectorizer, n=15):
    """Pretty-print the output of top_spam_indicators (skips quietly if empty)."""
    indicators = top_spam_indicators(model, vectorizer, n=n)
    if not indicators:
        return

    print("=" * 66)
    print("  TOP {} SPAM INDICATOR TERMS".format(len(indicators)))
    print("=" * 66)
    for rank, (term, score) in enumerate(indicators, start=1):
        print("  {:>2}. {:<28} {:+.4f}".format(rank, term, score))
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print("[error]{}".format(exc), file=sys.stderr)
        sys.exit(1)

    df = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_vec, X_test_vec, vectorizer = vectorize(X_train, X_test)

    results = compare_models(X_train, y_train, X_train_vec, X_test_vec, y_test)

    best = max(results, key=lambda r: r["cv_f1"])
    print("[best] {}  (CV F1 = {:.4f}, test F1 = {:.4f})\n".format(
        best["name"], best["cv_f1"], best["f1"]))

    plot_confusion_matrix(y_test, best["y_pred"], best["name"])
    save_best_model(best, vectorizer)

    print()
    print_top_indicators(best["model"], vectorizer)

    run_demo()


if __name__ == "__main__":
    main()
