from transformers import pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Load high-quality AI models
# ----------------------------
def get_models():
    """Load summarization and sentiment models (no 200MB restriction)."""
    models = {}
    try:
        # More accurate summarization
        models["summarizer"] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="auto"
        )
        # More nuanced sentiment analysis
        models["sentiment"] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device="auto"
        )
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# ----------------------------
# Summarization with chunking
# ----------------------------
def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_text(models, text, max_length=200, min_length=50):
    if not text.strip():
        return ""
    summaries = []
    for chunk in chunk_text(text):
        try:
            summary = models["summarizer"](chunk, max_length=max_length, min_length=min_length, truncation=True)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {e}]")
    return " ".join(summaries)

# ----------------------------
# Sentiment analysis
# ----------------------------
def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0}
    try:
        result = models["sentiment"](text[:512])[0]  # limit input for performance
        return result
    except Exception as e:
        return {"label": "ERROR", "score": 0}

# ----------------------------
# Global summary for multiple articles
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
# Extract main terms
# ----------------------------
def extract_main_terms(text, top_n=10):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, freq in words_freq[:top_n]]
