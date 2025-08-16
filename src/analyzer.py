from typing import List, Dict, Tuple, Any
import re
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------- Model loading ----------
def get_models():
    """
    Load and cache models used across the app.
    Chosen for decent quality vs. small footprint on Streamlit Cloud.
    """
    models = {}

    # Summarization (smallish)
    models["summarizer"] = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device_map="auto",
    )

    # Sentiment (SST-2)
    models["sentiment"] = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device_map="auto",
    )

    # NER
    models["ner"] = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device_map="auto",
    )

    # QA (extractive)
    models["qa"] = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device_map="auto",
    )

    # Embeddings for retrieval
    models["embedder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return models


# ---------- Utilities ----------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 150) -> List[str]:
    """
    Naive chunking by characters with overlap (keeps it tokenizer-agnostic).
    """
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        # extend to sentence end if possible
        if end < len(text):
            dot = text.rfind(".", start, end)
            if dot > 0 and dot - start > 300:
                chunk = text[start:dot + 1]
                end = dot + 1
        chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


# ---------- Per-article analysis ----------
def summarize_text(text: str, models: Dict[str, Any]) -> str:
    text = clean_text(text)
    chunks = chunk_text(text, max_chars=2000, overlap=200)
    summaries = []
    for ch in chunks:
        out = models["summarizer"](ch, max_length=160, min_length=50, do_sample=False)
        summaries.append(out[0]["summary_text"])
    merged = " ".join(summaries)
    # optional second-pass compression
    if len(merged) > 2500:
        out = models["summarizer"](merged[:8000], max_length=220, min_length=80, do_sample=False)
        merged = out[0]["summary_text"]
    return merged


def analyze_sentiment(text: str, models: Dict[str, Any]) -> Tuple[str, float]:
    text = clean_text(text)
    sample = text[:5000]  # keep it fast
    result = models["sentiment"](sample)[0]
    label = result["label"].lower()
    score = float(result["score"])
    if label == "negative" and score < 0.6:
        label = "neutral"  # soften borderline negativity for long docs
    return label, score


def extract_entities(text: str, models: Dict[str, Any]) -> List[Dict[str, str]]:
    text = clean_text(text)
    sample = text[:8000]
    ents = models["ner"](sample)
    cleaned = []
    seen = set()
    for e in ents:
        key = (e["word"].strip(), e["entity_group"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"text": e["word"].strip(), "label": e["entity_group"]})
    return cleaned


# ---------- Global summary ----------
def global_summary_from_summaries(summaries: List[str], models: Dict[str, Any]) -> str:
    joined = " ".join(summaries)
    joined = joined[:12000]  # safety cap
    out = models["summarizer"](joined, max_length=260, min_length=90, do_sample=False)
    return out[0]["summary_text"]


# ---------- RAG (retrieval-augmented Q&A) ----------
def build_corpus_index(docs: List[Dict[str, str]], models: Dict[str, Any], chunk_chars: int = 900) -> Dict[str, Any]:
    """
    Build a simple in-memory index:
      - chunk each doc
      - embed chunks
      - store vectors & metadata
    """
    passages = []
    for d in docs:
        for ch in chunk_text(d["text"], max_chars=chunk_chars, overlap=120):
            passages.append(
                {"id": d["id"], "name": d["name"], "text": ch, "snippet": ch[:360] + ("..." if len(ch) > 360 else "")}
            )
    embeddings = models["embedder"].encode([p["text"] for p in passages], normalize_embeddings=True)
    index = {
        "passages": passages,
        "embeddings": embeddings.astype("float32"),
    }
    return index


def rag_qa(question: str, index: Dict[str, Any], models: Dict[str, Any], top_k: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    q_vec = models["embedder"].encode([question], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, index["embeddings"])[0]
    top_idx = np.argsort(-sims)[:top_k]
    ctx = "\n\n".join(index["passages"][i]["text"] for i in top_idx)
    ctx = ctx[:4500]  # keep QA fast
    qa_out = models["qa"]({"question": question, "context": ctx})
    answer = qa_out.get("answer", "").strip() or "(no exact answer found; try rephrasing)"
    support = [index["passages"][i] for i in top_idx]
    return answer, support
