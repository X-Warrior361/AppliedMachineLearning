import os
import sys
import time
import requests
import pytest
import subprocess
import signal

def test_flask_endpoint():
    """
    Integration test for Flask scoring endpoint
    - Launches Flask app
    - Tests endpoint
    - Closes Flask app
    """
    # Prepare the command to run the Flask app
    flask_command = [sys.executable, 'flask-app.py']
    
    # Start the Flask app as a subprocess
    try:
        # Launch the Flask app
        process = subprocess.Popen(
            flask_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Give the app a moment to start
        time.sleep(2)
        
        # Test endpoint with spam text
        try:
            spam_response = requests.post(
                'http://127.0.0.1:5000/score', 
                json={'text': 'well done england get official poly ringtone colour flag yer mobile text tone flag optout txt eng stop box wwx £'},
                timeout=5
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
                timeout=5
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
        
        except requests.RequestException as e:
            pytest.fail(f"Request to Flask endpoint failed: {e}")
        
    finally:
        # Terminate the Flask process
        if process:
            # Send SIGTERM to gracefully stop the process
            os.kill(process.pid, signal.SIGTERM)
            
            # Wait for the process to terminate
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                os.kill(process.pid, signal.SIGKILL)


if __name__ == "__main__":
    test_flask_endpoint()