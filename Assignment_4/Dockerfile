FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

# Copy application files
COPY app.py .
COPY score.py .
COPY best_model.pkl .
COPY vectorizer.joblib .

# Expose port for the Flask app
EXPOSE 5000

# Run the app when the container launches
CMD ["python", "app.py"]
