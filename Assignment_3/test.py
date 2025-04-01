import os
import sys
import time
import requests
import pytest
import subprocess
import signal
import joblib
import platform
from score import score

# Load the trained model
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

def test_flask():
    # Prepare the command to run the Flask app
    flask_command = [sys.executable, 'app.py']
    
    try:
        # Launch the Flask app
        process = subprocess.Popen(
            flask_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        time.sleep(15)
        
        # Test endpoint with spam text
        spam_response = requests.post(
            'http://127.0.0.1:5000/score', 
            json={'text': 'well done england get official poly ringtone colour flag yer mobile text tone flag optout txt eng stop box wwx £'},
            timeout=15
        )
        
        # Validate response
        assert spam_response.status_code == 200
        response_data = spam_response.json()
        
        # Check response structure
        assert 'prediction' in response_data
        assert 'propensity' in response_data
        
        # Check prediction is True for spam
        assert response_data['prediction'] is True
        assert 0 <= response_data['propensity'] <= 1
        
        # Test non-spam text
        non_spam_response = requests.post(
            'http://127.0.0.1:5000/score', 
            json={'text': 'Hi Mom, how are you today?'},
            timeout=15
        )
        
        # Validate response
        assert non_spam_response.status_code == 200
        non_spam_data = non_spam_response.json()
        
        # Check response structure
        assert 'prediction' in non_spam_data
        assert 'propensity' in non_spam_data
        
        # Check prediction is False for non-spam
        assert non_spam_data['prediction'] is False
        assert 0 <= non_spam_data['propensity'] <= 1
        
    finally:
        if process:
            os.kill(process.pid, signal.SIGTERM)
