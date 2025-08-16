import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

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
# Page Config
# ----------------------------
st.set_page_config(
    page_title="FIKRA Simplify",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# Header: Logo + Title
# ----------------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=100)
    except:
        st.warning("Logo not found!")

with col_title:
    st.markdown("<h1>📑 FIKRA Simplify</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray; font-size:16px;'>Simplifying complex information.</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    return get_models()

models = load_models()

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
    accept_multiple_files=True
)

all_texts = []
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")

        # Extract text
        text = extract_text_from_file(uploaded_file)
        all_texts.append(text)

        # Summarize
        summary = summarize_text(models, text)

        # Sentiment
        sentiment = analyze_sentiment(models, text)

        # Main terms
        main_terms = extract_main_terms(text)

        # Save results
        results.append({
            "filename": uploaded_file.name,
            "summary": summary,
            "sentiment": sentiment,
            "main_terms": main_terms
        })

        # Display results
        st.markdown(f"**Summary:** {summary}")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Main Terms:** {', '.join(main_terms)}")

# ----------------------------
# Global Summary
# ----------------------------
if all_texts:
    st.markdown("---")
    st.subheader("🌍 Global Summary Across All Articles")
    st.write(global_summary(models, all_texts))

# ----------------------------
# Q&A Section
# ----------------------------
if all_texts:
    st.markdown("---")
    st.subheader("❓ Ask Questions About Articles")
    question = st.text_input("Enter your question:")
    if question:
        st.markdown(f"**Answer:** {simple_qa(question, all_texts)}")

# ----------------------------
# Visualizations
# ----------------------------
if results:
    st.markdown("---")
    st.subheader("📊 Analysis Visualizations")

    # Sentiment Pie Chart
    labels = [r["sentiment"]["label"] for r in results]
    sentiment_counts = pd.Series(labels).value_counts()

    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=['#4CAF50','#F44336','#FFC107']
    )
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    # Main Terms Table
    all_terms = []
    for r in results:
        all_terms.extend(r["main_terms"])

    if all_terms:
        term_counts = pd.Series(all_terms).value_counts().reset_index()
        term_counts.columns = ["Term", "Frequency"]
        st.subheader("📋 Main Terms Table")
        st.dataframe(term_counts.style.background_gradient(cmap='Blues'))
