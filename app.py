"""
Streamlit front-end for the Spam SMS classifier.

    python spam_sms_detection.py     # train once -> writes spam_model.joblib
    streamlit run app.py             # open the web UI
"""

import os

import joblib
import numpy as np
import streamlit as st

from spam_sms_detection import CLASS_NAMES, MODEL_FILE, clean_text, top_spam_indicators

EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! "
                  "Call 09061701461 now to claim. Valid 12 hours only.",
    "Friend": "Hey, are we still meeting for lunch at 1pm tomorrow?",
    "Promo": "FREE entry to win a new iPhone! Text WIN to 80086. T&Cs apply www.win-now.com",
    "Work": "Sorry I missed your call, I was in a lecture. Will ring you back this evening.",
}

st.set_page_config(page_title="Spam SMS Detector", page_icon="📩", layout="centered")


@st.cache_resource
def load_bundle(path=MODEL_FILE):
    return joblib.load(path)


def spam_confidence(model, X):
    """Map the model's raw score to a 0-1 spam confidence.

    Naive Bayes / Logistic Regression have predict_proba. LinearSVC only has a
    signed distance from the margin, so squash it with a sigmoid -- good for a
    confidence bar, not a calibrated probability.
    """
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    return float(1.0 / (1.0 + np.exp(-model.decision_function(X)[0])))


def words_that_triggered(model, vectorizer, cleaned, n=6):
    """The message's own terms that pushed it hardest toward SPAM."""
    X = vectorizer.transform([cleaned])
    present = set(vectorizer.get_feature_names_out()[X.nonzero()[1]])
    ranked = top_spam_indicators(model, vectorizer, n=len(vectorizer.vocabulary_))
    return [(t, s) for t, s in ranked if t in present and s > 0][:n]


st.title("📩 Spam SMS Detector")
st.caption("CodSoft ML Internship · Task 4 · TF-IDF + Naive Bayes / Logistic Regression / Linear SVC")

if not os.path.exists(MODEL_FILE):
    st.error(
        f"`{MODEL_FILE}` not found. Train the model first:\n\n"
        "```\npython spam_sms_detection.py\n```"
    )
    st.stop()

bundle = load_bundle()
model, vectorizer = bundle["model"], bundle["vectorizer"]
st.sidebar.markdown(f"**Model:** {bundle['model_name']}")
st.sidebar.markdown("**Try an example:**")
for label, text in EXAMPLES.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state["msg"] = text

message = st.text_area("Paste an SMS message", key="msg", height=130,
                       placeholder="e.g. URGENT! You have won a £1000 cash prize...")

if st.button("Check message", type="primary") and message.strip():
    cleaned = clean_text(message)
    X = vectorizer.transform([cleaned])
    label = CLASS_NAMES[int(model.predict(X)[0])]
    conf = spam_confidence(model, X)

    if label == "SPAM":
        st.error(f"🚫 **SPAM** — spam score {conf:.0%}")
    else:
        st.success(f"✅ **HAM (legitimate)** — spam score {conf:.0%}")
    st.progress(conf)

    triggers = words_that_triggered(model, vectorizer, cleaned)
    if triggers:
        st.markdown("**Terms pushing toward spam:** " +
                    ", ".join(f"`{t}`" for t, _ in triggers))
    with st.expander("What the model actually saw (cleaned text)"):
        st.code(cleaned or "(empty after cleaning)")
