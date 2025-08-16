from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Load models
# Load AI Models
# ----------------------------
def get_models():
    """Load and return AI models."""
    models = {}
    try:
        models["summarizer"] = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            model="facebook/bart-large-cnn",  # robust for longer text
            device_map="auto"
        )
        models["sentiment"] = pipeline(
@@ -26,39 +25,35 @@ def get_models():
# ----------------------------
# Summarization
# ----------------------------
def summarize_text(models, text, max_length=130, min_length=30):
def summarize_text(models, text, max_length=150, min_length=30):
    if not text.strip():
        return ""
    try:
        summary = models["summarizer"](text, max_length=max_length, min_length=min_length, truncation=True)
        return summary[0]["summary_text"]
        result = models["summarizer"](text, max_length=max_length, min_length=min_length, truncation=True)
        return result[0]["summary_text"]
    except Exception as e:
        return f"[Error summarizing text: {e}]"
        return f"[Error summarizing: {e}]"

# ----------------------------
# Sentiment analysis
# Sentiment Analysis
# ----------------------------
def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0}
    try:
        result = models["sentiment"](text[:512])[0]
        return result
        return models["sentiment"](text[:512])[0]
    except Exception as e:
        return {"label": "ERROR", "score": 0}

# ----------------------------
# Global summary (single input)
# Global Summary
# ----------------------------
def global_summary(models, all_texts):
    combined_text = " ".join(all_texts)
    try:
        return summarize_text(models, combined_text)
    except Exception as e:
        return f"[Error summarizing text: {e}]"
    return summarize_text(models, combined_text)

# ----------------------------
# Simple Q&A
# Simple Q&A (keyword search)
# ----------------------------
def simple_qa(question, all_texts):
    combined_text = " ".join(all_texts)
@@ -68,12 +63,12 @@ def simple_qa(question, all_texts):
    return " ".join(answers[:3]) if answers else "No relevant information found."

# ----------------------------
# Extract main terms
# Extract Main Terms
# ----------------------------
def extract_main_terms(text, top_n=10):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, freq in words_freq[:top_n]]
    sums = X.sum(axis=0)
    terms_freq = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    terms_freq.sort(key=lambda x: x[1], reverse=True)
    return [word for word, freq in terms_freq[:top_n]]
