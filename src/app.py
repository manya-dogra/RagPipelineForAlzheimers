# app.py
import streamlit as st
from retriever import query_faiss
from pipeline import summarize_with_model


st.set_page_config(page_title="Alzheimer’s RAG System")
st.title("Alzheimer’s Clinical Trial Explorer")

# Get query from user
user_query = st.text_input("Ask a question about Alzheimer’s clinical trials:")

if user_query:
    with st.spinner("Retrieving relevant trials..."):
        docs = query_faiss(user_query, top_k=5)

    st.subheader("Retrieved Trials")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**Document {i}**")
        st.write(doc[:500] + "...")

    # Prepare prompt for query-focused summary
    combined_text = " ".join(docs)
    prompt_text = f"Question: {user_query}\nContext: {combined_text}\nAnswer:"

    models = [
        "sshleifer/distilbart-cnn-12-6",
        "t5-small",
        "facebook/bart-large-cnn",
        "google/flan-t5-base",
        "google/pegasus-xsum",
        "facebook/bart-base"]

    st.subheader("Generated Summaries (Query-Focused)")
    for model in models:
        with st.spinner(f"Generating answer with {model}..."):
            summary, time_taken = summarize_with_model(prompt_text, model)
            st.markdown(f"### {model}")
            st.write(summary)
            st.caption(f"Time taken: {time_taken} seconds")
