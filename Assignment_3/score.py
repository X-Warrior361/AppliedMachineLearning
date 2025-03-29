import sklearn
from typing import Tuple
import joblib   
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    tokens = text.split()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)

def score(text: str, 
          model: sklearn.base.BaseEstimator, 
          threshold: float) -> Tuple[bool, float]:

    vectorizer = joblib.load('vectorizer.joblib')
    text_vectorized = vectorizer.transform([preprocess_text(text)])
    
    # Get probability prediction
    propensity = model.predict_proba(text_vectorized)[0][1]
    
    # Determine prediction based on threshold
    prediction = propensity >= threshold
    
    return bool(prediction), propensity
