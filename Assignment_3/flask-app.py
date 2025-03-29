from flask import Flask, request, jsonify
import joblib
import sklearn
import os
import sys

# Add the directory containing score.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from score import score

app = Flask(__name__)

# Load the model globally
try:
    model = joblib.load('best_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/score', methods=['POST'])
def score_text():
    """
    Endpoint to score a text for spam prediction
    
    Expected JSON input: {'text': 'sample text to score'}
    Returns JSON with prediction and propensity
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'prediction': None,
            'propensity': None
        }), 500
    
    # Get text from request
    data = request.get_json()
    
    # Validate input
    if not data or 'text' not in data:
        return jsonify({
            'error': 'No text provided',
            'prediction': None,
            'propensity': None
        }), 400
    
    # Default threshold to 0.5 if not provided
    threshold = data.get('threshold', 0.5)
    
    # Score the text
    try:
        prediction, propensity = score(data['text'], model, threshold)
        
        return jsonify({
            'prediction': bool(prediction),
            'propensity': float(propensity)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'prediction': None,
            'propensity': None
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
