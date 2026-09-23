"""
Streamlit front-end for the Spam SMS classifier.

    python spam_sms_detection.py     # train once -> writes spam_model.joblib
    streamlit run app.py             # open the web UI

Three views: scan one message, scan a batch (paste or CSV upload), and inspect
what the trained model actually learned.
"""

import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from spam_sms_detection import (
    CONFUSION_MATRIX_FILE,
    MODEL_FILE,
    clean_text,
    top_spam_indicators,
)

st.set_page_config(
    page_title="Spam SMS detector",
    page_icon=":material/shield:",
    layout="wide",
)

EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! "
                  "Call 09061701461 now to claim. Valid 12 hours only.",
    "Friend": "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "Promo": "FREE entry to win a new iPhone! Text WIN to 80086. T&Cs apply www.win-now.com",
    "Work": "Sorry I missed your call, I was in a lecture. Will ring you back this evening.",
    "Bank phish": "ALERT: your account has been suspended. Verify at www.secure-bank-login.com "
                  "within 24 hrs or it will be closed permanently.",
    "Reminder": "Don't forget the dentist appointment on the 24th at 4pm. Reply OK to confirm.",
}


# --------------------------------------------------------------------------- #
# Loading and scoring
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading model…")
def load_bundle(path=MODEL_FILE):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def term_weights(_model, _vectorizer):
    """Every vocabulary term ranked by how hard it pushes toward SPAM.

    Sorting ~30k terms is not free, so it runs once per session rather than on
    every rerun. The leading underscores tell Streamlit not to try hashing the
    model and vectorizer -- there is only ever one of each, so a constant cache
    key is correct here.
    """
    ranked = top_spam_indicators(_model, _vectorizer, n=len(_vectorizer.vocabulary_))
    return pd.DataFrame(ranked, columns=["term", "weight"])


def spam_scores(model, X):
    """Map the model's raw scores to 0-1 spam scores, one per row of X.

    Naive Bayes / Logistic Regression have predict_proba. LinearSVC only has a
    signed distance from the margin, so squash it with a sigmoid -- good for a
    confidence bar, not a calibrated probability. Either way a score of 0.5 is
    exactly the model's own decision boundary, so the threshold slider means
    the same thing for all three models.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X), dtype=float)))


def spam_confidence(model, X):
    """Spam score for a single-row matrix."""
    return float(spam_scores(model, X)[0])


def words_that_triggered(model, vectorizer, cleaned, n=6):
    """The message's own terms that pushed it hardest toward SPAM."""
    X = vectorizer.transform([cleaned])
    present = set(vectorizer.get_feature_names_out()[X.nonzero()[1]])
    ranked = top_spam_indicators(model, vectorizer, n=len(vectorizer.vocabulary_))
    return [(t, s) for t, s in ranked if t in present and s > 0][:n]


def classify(model, vectorizer, messages, threshold):
    """Score a list of raw messages. Returns a tidy results frame."""
    cleaned = [clean_text(m) for m in messages]
    scores = spam_scores(model, vectorizer.transform(cleaned))
    return pd.DataFrame({
        "message": messages,
        "cleaned": cleaned,
        "score": scores,
        "verdict": np.where(scores >= threshold, "SPAM", "HAM"),
    })


def review_queue(results, threshold):
    """Pick the rows a human should double-check before acting on the verdict.

    `results` is the frame classify() returns, with a float `score` column in
    [0, 1] and a `verdict` of "SPAM" or "HAM". Return the subset worth a second
    look, most-urgent first.
    """
    # TODO(human): decide which rows earn a manual review.
    return results.iloc[0:0]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

if not os.path.exists(MODEL_FILE):
    st.error(
        "`{}` not found. Train the model first:\n\n```\npython spam_sms_detection.py\n```".format(
            MODEL_FILE
        ),
        icon=":material/error:",
    )
    st.stop()

bundle = load_bundle()
model, vectorizer = bundle["model"], bundle["vectorizer"]
weights = term_weights(model, vectorizer)

with st.sidebar:
    st.subheader("Model", divider=False)
    st.markdown(":blue-badge[{}]".format(bundle["model_name"]))
    st.caption("{:,} terms in vocabulary · unigrams + bigrams".format(
        len(vectorizer.vocabulary_)))

    st.subheader("Decision threshold")
    threshold = st.slider(
        "Flag as spam at or above",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="0.50 is the model's own boundary. Raise it to cut false alarms, "
             "lower it to catch more spam.",
    )
    if threshold > 0.5:
        st.caption(":material/verified_user: Cautious — fewer real messages misfiled.")
    elif threshold < 0.5:
        st.caption(":material/radar: Aggressive — more spam caught, more false alarms.")
    else:
        st.caption(":material/balance: The model's default boundary.")

    st.subheader("About")
    st.caption(
        "CodSoft ML internship · Task 4. TF-IDF features over unigrams and "
        "bigrams, with the best of Naive Bayes / Logistic Regression / Linear SVC "
        "picked by 5-fold cross-validated F1."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("Spam SMS detector")
st.caption("Paste a message, scan a batch, or look inside the model.")

scan_tab, batch_tab, model_tab = st.tabs([
    ":material/sms: Scan a message",
    ":material/dataset: Bulk scan",
    ":material/insights: Model insights",
])


# --------------------------------------------------------------------------- #
# Tab 1 - single message
# --------------------------------------------------------------------------- #

def use_example():
    picked = st.session_state.get("example_pick")
    if picked:
        st.session_state["msg"] = EXAMPLES[picked]


with scan_tab:
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.pills(
            "Load an example",
            list(EXAMPLES),
            key="example_pick",
            on_change=use_example,
            label_visibility="collapsed",
        )
        message = st.text_area(
            "Message",
            key="msg",
            height=170,
            placeholder="e.g. URGENT! You have won a £1000 cash prize…",
            label_visibility="collapsed",
        )
        with st.container(horizontal=True):
            analyse = st.button("Check message", type="primary", icon=":material/search:")
            if st.button("Clear", icon=":material/backspace:"):
                st.session_state["msg"] = ""
                st.session_state.pop("analysed", None)
                st.rerun()

        if analyse and message.strip():
            st.session_state["analysed"] = message
            st.session_state.setdefault("history", []).insert(0, message)
            del st.session_state["history"][25:]

    analysed = st.session_state.get("analysed")

    with right:
        if not analysed:
            st.container(height=40, border=False)
            st.caption("Results will appear here.")
        else:
            cleaned = clean_text(analysed)
            X = vectorizer.transform([cleaned])
            score = spam_confidence(model, X)
            is_spam = score >= threshold

            with st.container(border=True):
                if is_spam:
                    st.markdown("### :material/block: :red[Spam]")
                else:
                    st.markdown("### :material/check_circle: :green[Legitimate]")
                st.progress(score)
                with st.container(horizontal=True):
                    st.metric("Spam score", "{:.0%}".format(score))
                    st.metric("Threshold", "{:.0%}".format(threshold))
                    st.metric(
                        "Margin",
                        "{:+.0%}".format(score - threshold),
                        help="How far past the threshold this message sits. "
                             "A margin near zero is a borderline call.",
                    )
                if abs(score - threshold) < 0.10:
                    st.warning(
                        "Borderline — this one sits close to the threshold.",
                        icon=":material/help:",
                    )

    if analysed:
        cleaned = clean_text(analysed)
        triggers = words_that_triggered(model, vectorizer, cleaned, n=8)

        detail_left, detail_right = st.columns([3, 2], gap="medium")
        with detail_left:
            with st.container(border=True):
                st.markdown("**Terms pushing this message toward spam**")
                if triggers:
                    trigger_df = pd.DataFrame(triggers, columns=["term", "weight"])
                    st.altair_chart(
                        alt.Chart(trigger_df)
                        .mark_bar(cornerRadiusEnd=4)
                        .encode(
                            x=alt.X("weight:Q", title="Weight toward spam"),
                            y=alt.Y("term:N", sort="-x", title=None),
                            tooltip=["term", alt.Tooltip("weight:Q", format=".3f")],
                        )
                        .properties(height=min(30 * len(trigger_df) + 30, 280))
                    )
                else:
                    st.caption("No term in this message argues for spam.")

        with detail_right:
            with st.container(border=True):
                st.markdown("**What the model actually saw**")
                st.caption(
                    "URLs, emails, currency symbols and long numbers become "
                    "placeholder tokens rather than being deleted."
                )
                st.code(cleaned or "(empty after cleaning)", language=None, wrap_lines=True)

    history = st.session_state.get("history", [])
    if len(history) > 1:
        with st.expander("Recent messages this session", icon=":material/history:"):
            recent = classify(model, vectorizer, history, threshold)
            st.dataframe(
                recent[["message", "score", "verdict"]],
                hide_index=True,
                column_config={
                    "message": st.column_config.TextColumn("Message", width="large"),
                    "score": st.column_config.ProgressColumn(
                        "Spam score", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "verdict": st.column_config.TextColumn("Verdict", width="small"),
                },
            )


# --------------------------------------------------------------------------- #
# Tab 2 - bulk scan
# --------------------------------------------------------------------------- #

with batch_tab:
    source = st.segmented_control(
        "Input", ["Paste messages", "Upload CSV"], default="Paste messages"
    )

    messages = []
    if source == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if upload is not None:
            try:
                raw = pd.read_csv(upload, encoding="latin-1")
            except Exception as exc:  # noqa: BLE001 - surface any parse failure
                st.error("Could not read that CSV: {}".format(exc), icon=":material/error:")
                raw = None
            if raw is not None and len(raw.columns):
                column = st.selectbox("Which column holds the message text?", list(raw.columns))
                messages = raw[column].dropna().astype(str).tolist()
    else:
        pasted = st.text_area(
            "One message per line",
            height=170,
            placeholder="Paste one message per line…",
        )
        messages = [line.strip() for line in pasted.splitlines() if line.strip()]

    if not messages:
        st.caption("Add some messages to scan.")
    else:
        results = classify(model, vectorizer, messages, threshold)
        flagged = int((results["verdict"] == "SPAM").sum())

        with st.container(horizontal=True):
            st.metric("Messages", "{:,}".format(len(results)), border=True)
            st.metric(
                "Flagged as spam", "{:,}".format(flagged), border=True,
                chart_data=results["score"].round(2).tolist(), chart_type="bar",
            )
            st.metric("Spam rate", "{:.0%}".format(flagged / len(results)), border=True)
            st.metric("Median score", "{:.2f}".format(results["score"].median()), border=True)

        only_spam = st.toggle("Show only flagged messages")
        table = results[results["verdict"] == "SPAM"] if only_spam else results

        st.dataframe(
            table[["message", "score", "verdict"]].sort_values("score", ascending=False),
            hide_index=True,
            column_config={
                "message": st.column_config.TextColumn("Message", width="large"),
                "score": st.column_config.ProgressColumn(
                    "Spam score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "verdict": st.column_config.TextColumn("Verdict", width="small"),
            },
        )

        queue = review_queue(results, threshold)
        with st.container(border=True):
            st.markdown("**Needs a human look**")
            if len(queue):
                st.dataframe(
                    queue[["message", "score", "verdict"]],
                    hide_index=True,
                    column_config={
                        "message": st.column_config.TextColumn("Message", width="large"),
                        "score": st.column_config.ProgressColumn(
                            "Spam score", min_value=0.0, max_value=1.0, format="%.2f"
                        ),
                        "verdict": st.column_config.TextColumn("Verdict", width="small"),
                    },
                )
            else:
                st.caption("Nothing queued for review.")

        st.download_button(
            "Download results",
            data=results[["message", "score", "verdict"]].to_csv(index=False).encode("utf-8"),
            file_name="spam_scan_results.csv",
            mime="text/csv",
            icon=":material/download:",
        )


# --------------------------------------------------------------------------- #
# Tab 3 - model insights
# --------------------------------------------------------------------------- #

with model_tab:
    with st.container(horizontal=True):
        st.metric("Winning model", bundle["model_name"], border=True)
        st.metric("Vocabulary", "{:,} terms".format(len(vectorizer.vocabulary_)), border=True)
        st.metric("N-grams", "1–2", border=True)

    top_n = st.slider("Terms to show", min_value=5, max_value=30, value=15, step=5)

    spam_side = weights.head(top_n).assign(pull="Spam")
    ham_side = weights.tail(top_n).assign(pull="Ham")

    chart_left, chart_right = st.columns(2, gap="medium")
    for column, frame, title, order in (
        (chart_left, spam_side, "Strongest spam signals", "-x"),
        (chart_right, ham_side, "Strongest legitimate signals", "x"),
    ):
        with column:
            with st.container(border=True):
                st.markdown("**{}**".format(title))
                st.altair_chart(
                    alt.Chart(frame)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X("weight:Q", title="Weight toward spam"),
                        y=alt.Y("term:N", sort=order, title=None),
                        color=alt.Color("pull:N", legend=None),
                        tooltip=["term", alt.Tooltip("weight:Q", format=".3f")],
                    )
                    .properties(height=min(24 * top_n + 30, 760))
                )

    st.caption(
        "Weights come straight from the trained model — signed coefficients for "
        "Logistic Regression and Linear SVC, or log P(term|spam) − log P(term|ham) "
        "for Naive Bayes. Tokens like `longnumtoken` and `currencytoken` are the "
        "placeholders the cleaner substitutes for phone numbers and prices."
    )

    if os.path.exists(CONFUSION_MATRIX_FILE):
        with st.expander("Confusion matrix from training", icon=":material/grid_on:"):
            st.image(CONFUSION_MATRIX_FILE, width=520)
