from transformers import pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Load models
# ----------------------------
def get_models():
    """Load and return AI models."""
    models = {}
    try:
        models["summarizer"] = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
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
def summarize_text(models, text, max_length=130, min_length=30):
    if not text.strip():
        return ""
    try:
        summary = models["summarizer"](text, max_length=max_length, min_length=min_length, truncation=True)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing text: {e}]"

# ----------------------------
# Sentiment analysis
# ----------------------------
def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0}
    try:
        result = models["sentiment"](text[:512])[0]
        return result
    except Exception as e:
        return {"label": "ERROR", "score": 0}

# ----------------------------
# Global summary (single input)
# ----------------------------
def global_summary(models, all_texts):
    combined_text = " ".join(all_texts)
    try:
        return summarize_text(models, combined_text)
    except Exception as e:
        return f"[Error summarizing text: {e}]"

# ----------------------------
# Simple Q&A
# ----------------------------
def simple_qa(question, all_texts):
    combined_text = " ".join(all_texts)
    keywords = question.lower().split()
    sentences = re.split(r'(?<=[.!?]) +', combined_text)
    answers = [s for s in sentences if any(k in s.lower() for k in keywords)]
    return " ".join(answers[:3]) if answers else "No relevant information found."

# ----------------------------
# Extract main terms
# ----------------------------
def extract_main_terms(text, top_n=10):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, freq in words_freq[:top_n]]
