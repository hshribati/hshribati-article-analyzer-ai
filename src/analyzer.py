from transformers import pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Load AI Models
# ----------------------------
def get_models():
    models = {}
    try:
        models["summarizer"] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",  # robust for longer text
            device_map="auto"
        )
        models["sentiment"] = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# ----------------------------
# Summarization
# ----------------------------
def summarize_text(models, text, max_length=150, min_length=30):
    if not text.strip():
        return ""
    try:
        result = models["summarizer"](text, max_length=max_length, min_length=min_length, truncation=True)
        return result[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing: {e}]"

# ----------------------------
# Sentiment Analysis
# ----------------------------
def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0}
    try:
        return models["sentiment"](text[:512])[0]
    except Exception as e:
        return {"label": "ERROR", "score": 0}

# ----------------------------
# Global Summary
# ----------------------------
def global_summary(models, all_texts):
    combined_text = " ".join(all_texts)
    return summarize_text(models, combined_text)

# ----------------------------
# Simple Q&A (keyword search)
# ----------------------------
def simple_qa(question, all_texts):
    combined_text = " ".join(all_texts)
    keywords = question.lower().split()
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    answers = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return " ".join(answers[:3]) if answers else "No relevant information found."

# ----------------------------
# Extract Main Terms
# ----------------------------
def extract_main_terms(text, top_n=10):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    sums = X.sum(axis=0)
    terms_freq = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    terms_freq.sort(key=lambda x: x[1], reverse=True)
    return [word for word, freq in terms_freq[:top_n]]
