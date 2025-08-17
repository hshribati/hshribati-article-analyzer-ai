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
# Page config
# ----------------------------
st.set_page_config(page_title="FIKRA Simplify", page_icon="🧠", layout="wide")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("FIKRA Simplify")
st.sidebar.write("Upload articles, see analysis, ask questions.")
# ----------------------------
# Sidebar Topics Section
# ----------------------------
st.sidebar.markdown("## 🎯 Explore Topics")
st.sidebar.markdown("---")

# Political Science
with st.sidebar.expander("🌍 Political Science", expanded=False):
    st.markdown("""
    - 🏛️ Comparative Politics  
    - 🌐 International Relations  
    - 📜 Political Theory  
    - 🏢 Public Policy  
    - ⚖️ Governance & Administration
    """)

# History
with st.sidebar.expander("📖 History", expanded=False):
    st.markdown("""
    - 🏺 Ancient History  
    - 🏰 Medieval History  
    - 🕌 Islamic History  
    - 🌍 World History  
    - 📅 Modern History
    """)

# Economics
with st.sidebar.expander("💰 Economics", expanded=False):
    st.markdown("""
    - 📉 Microeconomics  
    - 📈 Macroeconomics  
    - 🌱 Development Economics  
    - 🧠 Behavioral Economics  
    - 💹 International Trade
    """)

# Other topics
with st.sidebar.expander("✨ Other Topics", expanded=False):
    st.markdown("🚧 More sections coming soon...")

# ----------------------------
# Header: Logo left, title right
# ----------------------------
col_logo, col_title = st.columns([1, 4])

with col_logo:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.warning("Logo not found! Place it in 'assets/logo.png'")

with col_title:
    st.markdown("<h1 style='margin-bottom:0;'>📑 FIKRA Simplify</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:0; color:gray; font-size:16px;'>Simplifying complex information.</p>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Load models
# ----------------------------
@st.cache_resource
def load_models():
    return get_models()

models = load_models()

# ----------------------------
# App description
# ----------------------------
st.write(
    "Upload multiple articles (.pdf, .docx, .txt, .html) and get AI-powered analysis: "
    "summaries, sentiment, main terms, global insights, and Q&A."
)

# ----------------------------
# File upload
# ----------------------------
upload_col, result_col = st.columns([1, 2])

with upload_col:
    uploaded_files = st.file_uploader(
        "Upload Articles",
        type=["pdf", "docx", "txt", "html"],
        accept_multiple_files=True
    )

all_texts = []
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with result_col:
            st.subheader(f"📄 {uploaded_file.name}")

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
    try:
        summary = global_summary(models, all_texts)
        st.write(summary)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")

# ----------------------------
# Q&A Section
# ----------------------------
if all_texts:
    st.markdown("---")
    st.subheader("❓ Ask Questions About Articles")
    question = st.text_input("Enter your question:")
    if question:
        answer = simple_qa(question, all_texts, models=models)
        st.markdown(f"**Answer:** {answer}")


# ----------------------------
# Visualizations
# ----------------------------
if results:
    st.markdown("---")
    st.subheader("📊 Analysis Visualizations")

    # Sentiment Pie Chart
    labels = [r["sentiment"]["label"] for r in results]
    sentiment_counts = pd.Series(labels).value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=['#4CAF50', '#F44336', '#FFC107']
    )
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
        st.dataframe(term_counts.style.background_gradient(cmap='Blues').set_properties(**{'font-size':'14px'}))

# ----------------------------
# Modern CSS Styling
# ----------------------------
def local_css():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
        }

        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #ecf0f1;
        }

        /* Sidebar links */
        section[data-testid="stSidebar"] a {
            color: #1abc9c !important;
            text-decoration: none;
        }

        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
        }

        /* Cards (like summaries) */
        .reportview-container .markdown-text-container {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #1abc9c;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-size: 16px;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #16a085;
            transform: scale(1.05);
        }

        /* Input boxes */
        .stTextInput>div>div>input, textarea {
            border-radius: 8px;
            border: 1px solid #bdc3c7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call it once
local_css()
✨ What this will do:

Gradient background for the whole site.

Sidebar in dark mode with modern colors.

Rounded white cards for summaries & answers.

Modern buttons with hover effects.

Clean inputs with rounded corners.

Would you like me to merge this styling into your existing app code so you get the final full script ready-to-run?








You said:
yes


ChatGPT said:
Perfect 👍 Let’s merge it into your app. I’ll give you the full app.py with the modern design applied (keeping the features you already have like summaries, Q&A, sidebar topics, etc.).

Here’s the upgraded version:

python
Copy
Edit
import streamlit as st
from transformers import pipeline
from keybert import KeyBERT

# ----------------------------
# Modern CSS Styling
# ----------------------------
def local_css():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
        }

        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #ecf0f1;
        }

        /* Sidebar links */
        section[data-testid="stSidebar"] a {
            color: #1abc9c !important;
            text-decoration: none;
        }

        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
        }

        /* Cards (like summaries & answers) */
        .reportview-container .markdown-text-container {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #1abc9c;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6em 1.2em;
            font-size: 16px;
            transition: 0.3s;
        }

        div.stButton > button:hover {
            background-color: #16a085;
            transform: scale(1.05);
        }

        /* Input boxes */
        .stTextInput>div>div>input, textarea {
            border-radius: 8px;
            border: 1px solid #bdc3c7;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call CSS function
local_css()
