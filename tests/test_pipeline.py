"""
Unit tests for the Language Detection pipeline.

Tests cover:
  1. Pipeline output shape and type
  2. Known-input predictions
  3. Probability calibration (sums to 1)
  4. Preprocessing function

Lives at: tests/test_pipeline.py
Run with: pytest tests/ -v
"""
import pickle
import numpy as np
import pytest
import re


# ── Preprocessing function (mirrors the notebook) ──
def clean_text(text: str) -> str:
    text = re.sub(r'[!@#$(),\\n"%^*?\\:;~`0-9]', ' ', text)
    text = re.sub(r'[\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


@pytest.fixture(scope="module")
def model():
    """Load the trained pipeline once for all tests."""
    # The pipeline's CountVectorizer references clean_text from __main__
    # (since it was defined in a Jupyter notebook cell). We must inject
    # clean_text into __main__ before unpickling.
    import __main__
    __main__.clean_text = clean_text
    with open("trained_pipeline-0.1.0.pkl", "rb") as f:
        return pickle.load(f)


class TestPreprocessing:
    """Tests for the text cleaning function."""

    def test_removes_digits(self):
        assert "123" not in clean_text("hello123world")

    def test_removes_special_chars(self):
        result = clean_text("hello!@#world")
        assert "!" not in result
        assert "@" not in result

    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self):
        result = clean_text("hello    world")
        assert "  " not in result


class TestPipeline:
    """Tests for the trained ML pipeline."""

    def test_predict_returns_array(self, model):
        pred = model.predict(["Hello world"])
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (1,)

    def test_predict_known_english(self, model):
        pred = model.predict(["Hello, how are you doing today?"])
        # Class 3 = English (alphabetically sorted)
        assert pred[0] == 3, f"Expected English (3), got {pred[0]}"

    def test_predict_known_french(self, model):
        pred = model.predict(["Bonjour, comment allez-vous?"])
        # Class 4 = French
        assert pred[0] == 4, f"Expected French (4), got {pred[0]}"

    def test_probabilities_sum_to_one(self, model):
        probas = model.predict_proba(["Some test text"])[0]
        assert abs(probas.sum() - 1.0) < 1e-6, f"Probas sum to {probas.sum()}"

    def test_batch_prediction(self, model):
        texts = ["Hello", "Bonjour", "Hola"]
        preds = model.predict(texts)
        assert preds.shape == (3,)
