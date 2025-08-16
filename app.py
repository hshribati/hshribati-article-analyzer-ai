import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import extract_text_from_file
from src.analyzer import (
    get_models,
    summarize_text,
    analyze_sentiment,
    global_summary,
    simple_qa,
    extract_main_terms
)

# ----------------------------
# Logo, Title & Tagline
# ----------------------------
st.markdown(
    """
    <div style="text-align: center;">
        <img src="assets/logo.png" width="150">
        <h1>📑 FIKRA Simplify</h1>
        <p><b>Simplifying complex information.</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Load models (cached for performance)
# ----------------------------
@st.cache_resource
def _load_models():
    return get_models()

models = _load_models()

# ----------------------------
# App Description
# ----------------------------
st.write(
    "Upload multiple articles (.pdf, .docx, .txt, .html) and get AI-powered analysis: "
    "summaries, sentiment, main terms, global insights, and Q&A."
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

        # Main Terms
        main_terms = extract_main_terms(text)

        # Save results
        results.append({
            "filename": uploaded_file.name,
            "summary": summary,
            "sentiment": sentiment,
            "main_terms": main_terms
        })

        # Show outputs
        st.write("**Summary:**", summary)
        st.write("**Sentiment:**", sentiment)
        st.write("**Main Terms:**", ", ".join(main_terms))

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

    # Main Terms Table
    all_terms = []
    for r in results:
        all_terms.extend(r["main_terms"])

    if all_terms:
        term_counts = pd.Series(all_terms).value_counts().reset_index()
        term_counts.columns = ["Term", "Frequency"]
        st.subheader("📋 Main Terms Table")
        st.dataframe(term_counts)
