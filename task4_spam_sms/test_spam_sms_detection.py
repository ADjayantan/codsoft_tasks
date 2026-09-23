"""
Test suite for the Spam SMS Detection project.

    pytest -v

Covers the pure logic (cleaning, feature prep, model introspection) with fast
synthetic data, plus a few slower checks against the real spam.csv and the
persisted spam_model.joblib artifact.
"""

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import spam_sms_detection as sd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, sd.DATA_FILE)
MODEL_PATH = os.path.join(HERE, sd.MODEL_FILE)


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #

class TestCleanText:

    def test_lowercases_and_strips_punctuation(self):
        assert sd.clean_text("Hello, WORLD!!") == "hello world"

    def test_collapses_whitespace_and_strips_edges(self):
        assert sd.clean_text("  too    many   spaces  ") == "too many spaces"

    @pytest.mark.parametrize("raw", [
        "check http://bit.ly/abc now",
        "check www.win-prize.com now",
    ])
    def test_urls_become_a_single_token(self, raw):
        out = sd.clean_text(raw)
        assert "urltoken" in out
        # the URL must not decay into its own fragments
        assert "http" not in out and "bit" not in out and "com" not in out

    def test_emails_become_a_token(self):
        assert "emailtoken" in sd.clean_text("mail me at foo.bar@example.org")

    @pytest.mark.parametrize("symbol", ["£", "$", "€"])
    def test_currency_symbols_become_a_token(self, symbol):
        assert "currencytoken" in sd.clean_text("win {}900 today".format(symbol))

    def test_long_numbers_become_longnumtoken(self):
        assert "longnumtoken" in sd.clean_text("call 09061701461 to claim")

    def test_short_numbers_become_numtoken_not_longnumtoken(self):
        out = sd.clean_text("meet at 1pm on the 24th")
        assert "numtoken" in out
        assert "longnumtoken" not in out

    def test_url_is_replaced_before_digits_are(self):
        """Ordering guard: if digits ran first, the URL would be shredded."""
        assert sd.clean_text("go to http://deal123456.com") == "go to urltoken"

    def test_digits_are_never_left_behind(self):
        assert not any(ch.isdigit() for ch in sd.clean_text("order 12345678 and 42"))

    def test_punctuation_only_message_cleans_to_empty(self):
        assert sd.clean_text("!!! ??? ...") == ""

    def test_accepts_non_string_input(self):
        # prepare_features maps over a column that may hold NaN / numbers
        assert sd.clean_text(12345678) == "longnumtoken"
        assert sd.clean_text(None) == "none"


# --------------------------------------------------------------------------- #
# prepare_features
# --------------------------------------------------------------------------- #

class TestPrepareFeatures:

    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["label", "message"])

    def test_maps_labels_to_0_and_1(self):
        out = sd.prepare_features(self._frame([
            ["ham", "see you soon"],
            ["spam", "WIN a free prize now"],
        ]))
        assert out["target"].tolist() == [0, 1]
        assert out["target"].dtype.kind == "i"

    def test_label_matching_is_case_and_whitespace_insensitive(self):
        out = sd.prepare_features(self._frame([
            [" Ham ", "hello there"],
            ["SPAM", "free prize"],
        ]))
        assert out["target"].tolist() == [0, 1]

    def test_drops_rows_with_an_unrecognised_label(self):
        out = sd.prepare_features(self._frame([
            ["ham", "hello there"],
            ["unknown", "mystery message"],
            ["spam", "free prize"],
        ]))
        assert len(out) == 2
        assert set(out["target"]) == {0, 1}

    def test_drops_rows_that_clean_to_nothing(self):
        out = sd.prepare_features(self._frame([
            ["ham", "hello there"],
            ["ham", "!!!"],
        ]))
        assert out["clean_message"].tolist() == ["hello there"]

    def test_does_not_mutate_the_input_frame(self):
        df = self._frame([["ham", "hello there"], ["spam", "free prize"]])
        sd.prepare_features(df)
        assert list(df.columns) == ["label", "message"]

    def test_adds_a_cleaned_column(self):
        out = sd.prepare_features(self._frame([["spam", "WIN £900 NOW!"]]))
        assert out["clean_message"].iloc[0] == "win currencytoken numtoken now"


# --------------------------------------------------------------------------- #
# load_dataset
# --------------------------------------------------------------------------- #

class TestLoadDataset:

    def test_missing_file_raises_with_a_helpful_message(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            sd.load_dataset(str(tmp_path / "nope.csv"))
        assert "kaggle.com" in str(exc.value)

    def test_keeps_first_two_columns_and_renames_them(self, tmp_path):
        csv = tmp_path / "spam.csv"
        csv.write_text(
            "v1,v2,Unnamed: 2,Unnamed: 3\n"
            "ham,hello there,,\n"
            "spam,free prize,,\n",
            encoding="latin-1",
        )
        assert list(sd.load_dataset(str(csv)).columns) == ["label", "message"]

    def test_drops_duplicate_rows(self, tmp_path):
        csv = tmp_path / "spam.csv"
        csv.write_text(
            "v1,v2\nham,hello there\nham,hello there\nspam,free prize\n",
            encoding="latin-1",
        )
        assert len(sd.load_dataset(str(csv))) == 2

    def test_drops_na_rows(self, tmp_path):
        csv = tmp_path / "spam.csv"
        csv.write_text("v1,v2\nham,hello there\nspam,\n", encoding="latin-1")
        assert len(sd.load_dataset(str(csv))) == 1

    @pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="spam.csv not present")
    def test_reads_the_real_dataset(self):
        df = sd.load_dataset(DATA_PATH)
        assert list(df.columns) == ["label", "message"]
        assert len(df) > 5000
        assert set(df["label"].str.strip().str.lower()) == {"ham", "spam"}


# --------------------------------------------------------------------------- #
# split_data / vectorize
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_df():
    """80 ham + 20 spam, cleaned and targeted -- the shape prepare_features emits."""
    rows = [["ham", "hey are we still on for lunch tomorrow"] for _ in range(40)]
    rows += [["ham", "sorry i missed your call i was in a lecture"] for _ in range(40)]
    rows += [["spam", "WINNER claim your free prize call 09061701461 now"] for _ in range(10)]
    rows += [["spam", "URGENT you have won £2000 text WIN to 80086"] for _ in range(10)]
    return sd.prepare_features(pd.DataFrame(rows, columns=["label", "message"]))


class TestSplitData:

    def test_split_sizes_follow_test_size(self, synthetic_df):
        X_train, X_test, _, _ = sd.split_data(synthetic_df)
        assert len(X_test) == pytest.approx(len(synthetic_df) * sd.TEST_SIZE, abs=1)
        assert len(X_train) + len(X_test) == len(synthetic_df)

    def test_split_is_stratified(self, synthetic_df):
        _, _, y_train, y_test = sd.split_data(synthetic_df)
        assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)

    def test_split_is_deterministic(self, synthetic_df):
        first = sd.split_data(synthetic_df)[0]
        second = sd.split_data(synthetic_df)[0]
        assert first.index.tolist() == second.index.tolist()


class TestVectorize:

    def test_produces_aligned_matrices(self, synthetic_df):
        X_train, X_test, _, _ = sd.split_data(synthetic_df)
        X_train_vec, X_test_vec, vec = sd.vectorize(X_train, X_test)
        assert X_train_vec.shape[0] == len(X_train)
        assert X_test_vec.shape[0] == len(X_test)
        # same feature space on both sides, or the model would read garbage
        assert X_train_vec.shape[1] == X_test_vec.shape[1] == len(vec.vocabulary_)

    def test_vectorizer_never_sees_test_vocabulary(self):
        """The leakage guard: a word only the test set contains must not exist."""
        X_train = pd.Series(["free prize now", "free prize now", "lunch tomorrow"])
        X_test = pd.Series(["zzzunseenword appears only here"])
        _, _, vec = sd.vectorize(X_train, X_test)
        assert "zzzunseenword" not in vec.vocabulary_

    def test_builds_bigrams(self):
        X_train = pd.Series(["call now to claim", "call now to claim", "call now please"])
        _, _, vec = sd.vectorize(X_train, pd.Series(["call now"]))
        assert "call now" in vec.vocabulary_

    def test_keeps_stopword_like_spam_terms(self):
        """build_vectorizer deliberately does NOT strip English stop words."""
        X_train = pd.Series(["call now free", "call now free", "please get this"])
        _, _, vec = sd.vectorize(X_train, pd.Series(["call now"]))
        assert "call" in vec.vocabulary_ and "now" in vec.vocabulary_


# --------------------------------------------------------------------------- #
# top_spam_indicators
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_fitted():
    """A fitted vectorizer plus the matrices, ready to fit any of the three models."""
    texts = pd.Series([
        "free prize winner claim now",
        "free cash prize claim now",
        "winner free prize call now",
        "lunch tomorrow at one",
        "see you at lunch tomorrow",
        "call you after lunch tomorrow",
    ])
    y = np.array([1, 1, 1, 0, 0, 0])
    vec = sd.build_vectorizer()
    X = vec.fit_transform(texts)
    return X, y, vec


class TestTopSpamIndicators:

    @pytest.mark.parametrize("model_cls", [MultinomialNB, LogisticRegression, LinearSVC])
    def test_returns_n_terms_sorted_by_descending_score(self, tiny_fitted, model_cls):
        X, y, vec = tiny_fitted
        model = model_cls().fit(X, y)
        out = sd.top_spam_indicators(model, vec, n=5)

        assert len(out) == 5
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(t, str) for t, _ in out)
        assert all(isinstance(s, float) for _, s in out)

    @pytest.mark.parametrize("model_cls", [MultinomialNB, LogisticRegression, LinearSVC])
    def test_surfaces_actual_spam_words(self, tiny_fitted, model_cls):
        X, y, vec = tiny_fitted
        model = model_cls().fit(X, y)
        terms = {t for t, _ in sd.top_spam_indicators(model, vec, n=6)}
        assert terms & {"free", "prize", "winner", "claim"}
        assert "lunch" not in terms

    def test_naive_bayes_uses_the_log_ratio_not_the_raw_spam_row(self, tiny_fitted):
        X, y, vec = tiny_fitted
        model = MultinomialNB().fit(X, y)
        expected = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        names = vec.get_feature_names_out()
        top_term, top_score = sd.top_spam_indicators(model, vec, n=1)[0]
        assert top_score == pytest.approx(float(expected.max()))
        # several terms can tie on the max ratio; any of them is a correct answer
        tied = {str(names[i]) for i in np.flatnonzero(expected == expected.max())}
        assert top_term in tied

    def test_coef_models_report_their_raw_weights(self, tiny_fitted):
        X, y, vec = tiny_fitted
        model = LogisticRegression().fit(X, y)
        _, top_score = sd.top_spam_indicators(model, vec, n=1)[0]
        assert top_score == pytest.approx(float(model.coef_[0].max()))

    def test_unknown_model_type_returns_empty_list(self, tiny_fitted):
        _, _, vec = tiny_fitted
        assert sd.top_spam_indicators(object(), vec, n=5) == []

    def test_n_larger_than_vocabulary_is_clamped(self, tiny_fitted):
        X, y, vec = tiny_fitted
        model = LogisticRegression().fit(X, y)
        out = sd.top_spam_indicators(model, vec, n=10_000)
        assert len(out) == len(vec.vocabulary_)


# --------------------------------------------------------------------------- #
# persistence round-trip
# --------------------------------------------------------------------------- #

class TestPersistence:

    def test_save_and_reload_preserves_predictions(self, tiny_fitted, tmp_path):
        X, y, vec = tiny_fitted
        model = LogisticRegression().fit(X, y)
        path = str(tmp_path / "model.joblib")

        sd.save_best_model({"model": model, "name": "Logistic Regression"}, vec, path)

        bundle = joblib.load(path)
        assert set(bundle) == {"model", "vectorizer", "model_name"}
        assert bundle["model_name"] == "Logistic Regression"

        probe = ["free prize winner", "lunch tomorrow"]
        before = model.predict(vec.transform(probe))
        after = bundle["model"].predict(bundle["vectorizer"].transform(probe))
        assert (before == after).all()

    def test_plot_confusion_matrix_writes_a_file(self, tmp_path):
        path = str(tmp_path / "cm.png")
        sd.plot_confusion_matrix(
            np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), "Test Model", path
        )
        assert os.path.exists(path) and os.path.getsize(path) > 0


# --------------------------------------------------------------------------- #
# the real saved artifact
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def saved_bundle():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("spam_model.joblib not present -- run spam_sms_detection.py first")
    return joblib.load(MODEL_PATH)


def predict_label(bundle, message):
    """Full inference path, exactly as app.py does it."""
    X = bundle["vectorizer"].transform([sd.clean_text(message)])
    return sd.CLASS_NAMES[int(bundle["model"].predict(X)[0])]


class TestSavedArtifact:

    def test_bundle_has_the_expected_shape(self, saved_bundle):
        assert set(saved_bundle) == {"model", "vectorizer", "model_name"}
        assert saved_bundle["model_name"] in sd.build_models()

    def test_model_and_vectorizer_agree_on_dimensions(self, saved_bundle):
        model, vec = saved_bundle["model"], saved_bundle["vectorizer"]
        n_features = len(vec.vocabulary_)
        if hasattr(model, "coef_"):
            assert model.coef_.shape[1] == n_features
        else:
            assert model.feature_log_prob_.shape[1] == n_features

    @pytest.mark.parametrize("message", [sd.SAMPLE_MESSAGES[0], sd.SAMPLE_MESSAGES[2]])
    def test_classifies_obvious_spam(self, saved_bundle, message):
        assert predict_label(saved_bundle, message) == "SPAM"

    @pytest.mark.parametrize("message", [sd.SAMPLE_MESSAGES[1], sd.SAMPLE_MESSAGES[3]])
    def test_classifies_obvious_ham(self, saved_bundle, message):
        assert predict_label(saved_bundle, message) == "HAM"

    def test_top_indicators_are_available_on_the_saved_model(self, saved_bundle):
        out = sd.top_spam_indicators(saved_bundle["model"], saved_bundle["vectorizer"], n=15)
        assert len(out) == 15

    @pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="spam.csv not present")
    def test_holdout_quality_floor(self, saved_bundle):
        """Guard against a retrain silently shipping a worse model.

        Rebuilds the exact train/test split the training script uses (same
        RANDOM_STATE, same stratification) and scores the persisted model on the
        held-out fifth it never trained on.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        df = sd.prepare_features(sd.load_dataset(DATA_PATH))
        _, X_test, _, y_test = sd.split_data(df)
        y_pred = saved_bundle["model"].predict(saved_bundle["vectorizer"].transform(X_test))

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Floors sit a few points under the shipped Linear SVC (acc .983, prec .945,
        # rec .916, f1 .930) -- tight enough to catch a real regression, loose enough
        # that an ordinary retrain does not turn the suite red. Precision is the
        # strictest: a real message landing in the spam folder is the costly error.
        assert precision >= 0.90, "spam precision dropped to {:.4f}".format(precision)
        assert recall >= 0.85, "spam recall dropped to {:.4f}".format(recall)
        assert f1 >= 0.88, "spam F1 dropped to {:.4f}".format(f1)
        # Accuracy alone proves little here -- 87% of the data is ham -- so it is
        # only a backstop against a model that collapsed to a single class.
        assert accuracy >= 0.97, "accuracy dropped to {:.4f}".format(accuracy)


# --------------------------------------------------------------------------- #
# app.py helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def app_module():
    """app.py runs Streamlit calls at import; skip if that fails outside a runtime."""
    try:
        import app
    except Exception as exc:  # noqa: BLE001 - any Streamlit bare-mode failure
        pytest.skip("app.py not importable outside a Streamlit runtime: {}".format(exc))
    return app


class TestAppHelpers:

    def test_spam_confidence_is_a_probability_for_proba_models(self, app_module, tiny_fitted):
        X, y, vec = tiny_fitted
        model = MultinomialNB().fit(X, y)
        probe = vec.transform(["free prize winner"])
        assert app_module.spam_confidence(model, probe) == pytest.approx(
            float(model.predict_proba(probe)[0, 1])
        )

    def test_spam_confidence_squashes_svc_margins_into_0_1(self, app_module, tiny_fitted):
        X, y, vec = tiny_fitted
        model = LinearSVC().fit(X, y)
        assert not hasattr(model, "predict_proba")
        spam_conf = app_module.spam_confidence(model, vec.transform(["free prize winner claim"]))
        ham_conf = app_module.spam_confidence(model, vec.transform(["lunch tomorrow"]))
        assert 0.0 <= ham_conf < 0.5 < spam_conf <= 1.0

    def test_words_that_triggered_only_returns_terms_in_the_message(self, app_module, tiny_fitted):
        X, y, vec = tiny_fitted
        model = LogisticRegression().fit(X, y)
        cleaned = sd.clean_text("free prize winner")
        out = app_module.words_that_triggered(model, vec, cleaned)
        assert out, "expected at least one triggering term"
        for term, score in out:
            assert score > 0
            assert all(word in cleaned for word in term.split())


# --------------------------------------------------------------------------- #
# the Streamlit UI (headless, via AppTest -- no browser or server)
# --------------------------------------------------------------------------- #

APP_PATH = os.path.join(HERE, "app.py")
SPAM_PROBE = "WINNER!! You have won a 900 prize! Call 09061701461 now to claim."
HAM_PROBE = "Hey, are we still meeting for lunch at 1pm tomorrow?"


@pytest.fixture
def running_app():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("spam_model.joblib not present -- run spam_sms_detection.py first")
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def scan_message(app, message):
    """Type into the single-message box and press the primary button."""
    app.text_area[0].set_value(message).run()
    app.button[0].click().run()
    return app


def verdict_heading(app):
    return next(m.value for m in app.markdown if str(m.value).startswith("###"))


class TestAppSmoke:

    def test_app_renders_without_exceptions(self, running_app):
        assert not running_app.exception
        assert running_app.title[0].value == "Spam SMS detector"
        assert len(running_app.tabs) == 3

    def test_spam_message_gets_a_spam_verdict(self, running_app):
        app = scan_message(running_app, SPAM_PROBE)
        assert not app.exception
        assert "Spam" in verdict_heading(app)

    def test_ham_message_gets_a_legitimate_verdict(self, running_app):
        app = scan_message(running_app, HAM_PROBE)
        assert not app.exception
        assert "Legitimate" in verdict_heading(app)

    def test_threshold_slider_moves_the_decision_boundary(self, running_app):
        """Same message, same score -- only the threshold changes the verdict."""
        borderline = ("Sorry I missed your call, will ring you back this evening "
                      "about the free tickets")
        slider = next(s for s in running_app.slider if s.label.startswith("Flag"))
        slider.set_value(0.20).run()
        aggressive = verdict_heading(scan_message(running_app, borderline))

        app = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(
            APP_PATH, default_timeout=120
        ).run()
        next(s for s in app.slider if s.label.startswith("Flag")).set_value(0.80).run()
        cautious = verdict_heading(scan_message(app, borderline))

        assert "Spam" in aggressive
        assert "Legitimate" in cautious

    def test_bulk_scan_counts_and_offers_a_download(self, running_app):
        running_app.text_area[1].set_value(
            "WINNER!! call 09061701461 to claim your 900 prize now\n"
            "Hey are we still on for lunch tomorrow\n"
            "FREE entry to win an iPhone text WIN to 80086\n"
            "Sorry I missed your call, will ring back tonight"
        ).run()
        assert not running_app.exception

        metrics = {m.label: m.value for m in running_app.metric}
        assert metrics["Messages"] == "4"
        assert int(metrics["Flagged as spam"]) >= 1
        assert [b.label for b in running_app.get("download_button")] == ["Download results"]

    def test_model_insights_tab_reports_the_saved_model(self, running_app):
        metrics = {m.label: m.value for m in running_app.metric}
        assert metrics["Winning model"] in sd.build_models()
        assert "terms" in metrics["Vocabulary"]
