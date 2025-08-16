import io
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_file_texts
from src.analyzer import (
    get_models,
    summarize_text,
    analyze_sentiment,
    extract_entities,
    build_corpus_index,
    rag_qa,
    global_summary_from_summaries,
)

st.set_page_config(page_title="Article Analyzer AI", layout="wide")

st.title("📰 Article Analyzer AI")
st.caption("Upload PDFs, Word docs, HTML or TXT — get summaries, sentiment, entities, a global roll‑up, and ask questions.")

# --- Sidebar: Upload ---
st.sidebar.header("Upload articles")
uploaded_files = st.sidebar.file_uploader(
    "Choose files (.pdf, .docx, .txt, .html)",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "html", "htm"],
)

# --- Cache models once per session ---
@st.cache_resource(show_spinner=False)
def _load_models():
    return get_models()

models = _load_models()

# --- Process files ---
if uploaded_files:
    with st.spinner("Extracting text..."):
        docs = load_file_texts(uploaded_files)  # List[dict]: {id, name, text}

    # Per‑article analysis
    results: List[Dict[str, Any]] = []
    with st.spinner("Analyzing articles with open‑source models…"):
        for d in docs:
            text = d["text"]
            summary = summarize_text(text, models)
            sent_label, sent_score = analyze_sentiment(text, models)
            ents = extract_entities(text, models)

            results.append(
                {
                    "id": d["id"],
                    "name": d["name"],
                    "summary": summary,
                    "sentiment": sent_label,
                    "sentiment_score": round(float(sent_score), 3),
                    "entities": ents,  # [{'text':..,'label':..}, ...]
                    "text": text,
                }
            )

    df = pd.DataFrame(
        [
            {
                "File": r["name"],
                "Summary": r["summary"],
                "Sentiment": f"{r['sentiment']} ({r['sentiment_score']})",
                "Top Entities": ", ".join(sorted({e['text'] for e in r["entities"]})[:12]),
                "Chars": len(r["text"]),
            }
            for r in results
        ]
    )

    st.subheader("Per‑Article Results")
    st.dataframe(df, use_container_width=True, height=300)

    # --- Charts ---
    st.subheader("Visualizations")
    col1, col2 = st.columns(2, vertical_alignment="top")

    # Sentiment pie
    with col1:
        st.markdown("**Sentiment Distribution**")
        pie_data = (
            pd.Series([r["sentiment"] for r in results])
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )
        fig1, ax1 = plt.subplots()
        ax1.pie(pie_data["Count"], labels=pie_data["Sentiment"], autopct="%1.0f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1, use_container_width=True)

    # Entity bar
    with col2:
        st.markdown("**Top Named Entities**")
        all_ents = []
        for r in results:
            all_ents.extend([(e["text"], e["label"]) for e in r["entities"]])
        ent_df = pd.DataFrame(all_ents, columns=["Entity", "Type"])
        if not ent_df.empty:
            top_entities = (
                ent_df.groupby(["Entity", "Type"])
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
                .head(15)
            )
            fig2, ax2 = plt.subplots()
            ax2.barh(top_entities["Entity"], top_entities["Count"])
            ax2.invert_yaxis()
            ax2.set_xlabel("Mentions")
            ax2.set_ylabel("Entity")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No entities detected.")

    # --- Global summary ---
    st.subheader("Global Summary (across all articles)")
    global_summary = global_summary_from_summaries([r["summary"] for r in results], models)
    st.write(global_summary)

    # --- Q&A over corpus (RAG-lite) ---
    st.subheader("Ask Questions about these Articles")
    with st.spinner("Indexing the corpus for Q&A…"):
        index = build_corpus_index(
            [{"id": r["id"], "name": r["name"], "text": r["text"]} for r in results],
            models,
        )

    question = st.text_input("Your question")
    if question:
        with st.spinner("Searching & answering…"):
            answer, passages = rag_qa(question, index, models, top_k=5)
        st.markdown("**Answer:**")
        st.write(answer)
        with st.expander("Show supporting passages"):
            for i, p in enumerate(passages, 1):
                st.markdown(f"**{i}. {p['name']}**")
                st.write(p["snippet"])
else:
    st.info("👈 Upload a few articles to begin.")
