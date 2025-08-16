from transformers import pipeline
import re

# ----------------------------
# Load Hugging Face pipelines
# ----------------------------
def get_models():
    models = {}
    # Summarizer (good balance of size/quality, runs on CPU)
    models["summarizer"] = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # force CPU for Streamlit Cloud
    )

    # Sentiment Analysis
    models["sentiment"] = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

    # Named Entity Recognition
    models["ner"] = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1
    )

    return models


# ----------------------------
# Summarization
# ----------------------------
def summarize_text(models, text, max_length=130, min_length=30):
    if not text.strip():
        return "No content to summarize."
    try:
        summary = models["summarizer"](text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error summarizing text: {e}"


# ----------------------------
# Sentiment Analysis
# ----------------------------
def analyze_sentiment(models, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    try:
        result = models["sentiment"](text[:512])[0]  # limit length for performance
        return {"label": result["label"], "score": result["score"]}
    except Exception as e:
        return {"label": "ERROR", "score": 0.0, "error": str(e)}


# ----------------------------
# Named Entity Recognition
# ----------------------------
def extract_entities(models, text):
    if not text.strip():
        return []
    try:
        entities = models["ner"](text[:1000])  # limit for performance
        # clean up entities
        return [{"entity": ent["entity_group"], "word": ent["word"]} for ent in entities]
    except Exception as e:
        return [{"entity": "ERROR", "word": str(e)}]


# ----------------------------
# Global Summary
# ----------------------------
def global_summary(models, texts):
    combined = " ".join(texts)
    return summarize_text(models, combined, max_length=200, min_length=60)


# ----------------------------
# Simple Q&A (heuristic based)
# ----------------------------
def simple_qa(question, texts):
    """
    Very simple Q&A: just searches for sentences containing keywords from the question.
    (If you want, we can later upgrade this to a proper Hugging Face Q&A model.)
    """
    keywords = re.findall(r"\w+", question.lower())
    results = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?]) +', text)
        for s in sentences:
            if any(k in s.lower() for k in keywords):
                results.append(s)
    if results:
        return " ".join(results[:5])  # return up to 5 matching sentences
    else:
        return "No relevant information found."

from collections import Counter
import re

def extract_main_terms(text, top_k=15):
    """
    Extract main terms/keywords from text by counting word frequency.
    Ignores common stopwords.
    """
    STOPWORDS = {
        "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
        "as", "by", "an", "are", "at", "from", "that", "this", "be",
        "has", "have", "it", "its", "or", "but", "not", "was", "which"
    }
    
    # clean and split words
    words = re.findall(r"\b\w+\b", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    counts = Counter(words)
    most_common = counts.most_common(top_k)
    
    return [word for word, count in most_common]

