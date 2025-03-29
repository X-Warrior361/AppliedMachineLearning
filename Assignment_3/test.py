import joblib
import numpy as np
from score import score

# Load the trained model and vectorizer
model = joblib.load('best_model.pkl')

def test_score_smoke_test():
    """Smoke test to ensure the function runs without crashing"""
    result = score("Test message", model, 0.5)
    assert result is not None

def test_score_output_format():
    """Test the output format and types"""
    result = score("Test message", model, 0.5)
    assert len(result) == 2
    assert isinstance(result[0], bool)  # prediction
    assert isinstance(result[1], float)  # propensity

def test_score_prediction_values():
    """Test prediction is either 0 or 1"""
    result = score("Test message", model, 0.5)
    prediction, _ = result
    assert prediction in [True, False]

def test_score_propensity_range():
    """Test propensity score is between 0 and 1"""
    _, propensity = score("Test message", model, 0.5)
    assert 0 <= propensity <= 1

def test_threshold_zero():
    """Test prediction is always 1 when threshold is 0"""
    prediction, _ = score("Test message", model, 0)
    assert prediction is True 

def test_threshold_one():
    """Test prediction is always 0 when threshold is 1"""
    prediction, _ = score("Test message", model, 1)
    assert prediction is False

def test_obvious_spam_text():
    """Test an obvious spam text"""
    spam_text = "well done england get official poly ringtone colour flag yer mobile text tone flag optout txt eng stop box wwx £"
    prediction, propensity = score(spam_text, model, 0.5)
    assert prediction is True
    assert propensity > 0.5

def test_obvious_non_spam_text():
    """Test an obvious non-spam text"""
    non_spam_text = "Hi Mom, how are you doing today?"
    prediction, propensity = score(non_spam_text, model, 0.5)
    assert prediction is False
    assert propensity <= 0.5

if __name__ == "__main__":
    test_score_smoke_test()
    test_score_output_format()
    test_score_prediction_values()
    test_score_propensity_range()
    test_threshold_zero()
    test_threshold_one()
    test_obvious_spam_text()
    test_obvious_non_spam_text()
