import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import extract_text_from_file
from src.analyzer import (
    get_models,
    summarize_text,
    analyze_sentiment,
    extract_entities,
    global_summary,
    simple_qa,
)

# ----------------------------
# Load models (cached for performance)
# ----------------------------
@st.cache_resource
def _load_models():
    return get_models()

models = _load_models()

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Article Analyzer AI", layout="wide")
st.title("📑 Article Analyzer AI")

st.set_page_config(page_title="FIKRA Analyzer AI", layout="wide")
st.title("📑 FIKRA Analyzer AI")
st.write(
    "Upload multiple articles (.pdf, .docx, .txt, .html) and get AI-powered analysis: "
    "summaries, sentiment, named entities, global insights, and Q&A."
)

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload Articles",
    type=["pdf", "docx", "txt", "html"],
    accept_multiple_files=True,
)

all_texts = []
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")

        text = extract_text_from_file(uploaded_file)
        all_texts.append(text)

        # Summarize
        summary = summarize_text(models, text)

        # Sentiment
        sentiment = analyze_sentiment(models, text)

        # Entities
        entities = extract_entities(models, text)

        # Save results
        results.append({
            "filename": uploaded_file.name,
            "summary": summary,
            "sentiment": sentiment,
            "entities": entities
        })

        # Show outputs
        st.write("**Summary:**", summary)
        st.write("**Sentiment:**", sentiment)

        if entities:
            st.write("**Named Entities:**")
            df_entities = pd.DataFrame(entities)
            st.dataframe(df_entities)

# ----------------------------
# Global Summary
# ----------------------------
if all_texts:
    st.subheader("🌍 Global Summary Across All Articles")
    st.write(global_summary(models, all_texts))

# ----------------------------
# Q&A Section
# ----------------------------
if all_texts:
    st.subheader("❓ Ask Questions About Articles")
    question = st.text_input("Enter your question:")
    if question:
        answer = simple_qa(question, all_texts)
        st.write("**Answer:**", answer)

# ----------------------------
# Visualizations
# ----------------------------
if results:
    st.subheader("📊 Analysis Visualizations")

    # Sentiment Pie Chart
    labels = [r["sentiment"]["label"] for r in results]
    sentiment_counts = pd.Series(labels).value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
    ax1.set_title("Sentiment Distribution")
    st.pyplot(fig1)

    # Entity Bar Chart
    all_entities = []
    for r in results:
        for ent in r["entities"]:
            all_entities.append(ent["entity"])
    if all_entities:
        entity_counts = pd.Series(all_entities).value_counts()
        fig2, ax2 = plt.subplots()
        entity_counts.plot(kind="bar", ax=ax2)
        ax2.set_title("Entity Distribution")
        st.pyplot(fig2)
