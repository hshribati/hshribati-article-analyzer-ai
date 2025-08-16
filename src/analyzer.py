from transformers import pipeline
import re
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Load AI models (Cloud-friendly)
# ----------------------------
def get_models():
    models = {}
    try:
        # Summarization
        models["summarizer"] = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device="cpu"
        )
        # Sentiment analysis
        models["sentiment"] = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device="cpu"
        )
        # Question Answering
        models["qa"] = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device="cpu"
        )
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# ----------------------------
# Q&A (context-aware)
# ----------------------------
def simple_qa(question, all_texts, models=None):
    if not all_texts or not question.strip():
        return "No context or question provided."

    # Combine all texts
    context = " ".join(all_texts)
    # Remove formatting codes (RTF, DOCX leftovers)
    context = re.sub(r'{\\.*?}|\\[a-z]+\d*', '', context)

    try:
        result = models["qa"](question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"No relevant information found: {e}"

# ----------------------------
# Keep existing functions
# ----------------------------
def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_text(models, text, max_length=130, min_length=30):
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

def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0}
    try:
        result = models["sentiment"](text[:512])[0]
        return result
    except Exception as e:
        return {"label": "ERROR", "score": 0}

def global_summary(models, all_texts):
    combined_text = " ".join(all_texts)
    return summarize_text(models, combined_text)

def extract_main_terms(text, top_n=10):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, freq in words_freq[:top_n]]
