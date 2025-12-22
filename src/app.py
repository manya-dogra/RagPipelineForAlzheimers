# app.py
import streamlit as st
from retriever import query_faiss, load_index
from pipeline import summarize_with_model
from embeddings import build_faiss_index
from io import StringIO
import docx2txt
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Alzheimer’s RAG Dashboard",
    layout="wide"
)

st.title("🧠 Alzheimer’s Clinical Trial Explorer")
st.caption("Retrieval-Augmented Generation over clinical trial data")

# ---------------- SIDEBAR: Top-k selection ----------------
top_k = st.sidebar.slider(
    "Select number of top documents to retrieve",
    min_value=1,
    max_value=10,
    value=5,  # default
    step=1
)

# ---------------- SIDEBAR: DOCUMENT UPLOAD ----------------
st.sidebar.header("📂 Upload Reference Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

uploaded_texts = []

for file in uploaded_files:
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
        uploaded_texts.append(text)

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        uploaded_texts.append(docx2txt.process(file))

    elif file.type == "text/plain":
        uploaded_texts.append(StringIO(file.getvalue().decode("utf-8")).read())

if uploaded_texts:
    st.sidebar.success(f"✅ {len(uploaded_texts)} document(s) uploaded")
    st.sidebar.caption("Uploaded files will be indexed and used for RAG summaries.")

# ---------------- QUERY INPUT ----------------
query = st.text_input(
    "🔎 Ask a question about Alzheimer’s clinical trials:"
)

# ---------------- RAG EXECUTION ----------------
if query:
    with st.spinner("🔍 Running Retrieval-Augmented Generation..."):

        # -------- BUILD TEMPORARY FAISS INDEX FROM UPLOADED DOCS --------
        if uploaded_texts:
            st.info("Indexing uploaded documents...")
            # Encode uploaded documents
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(uploaded_texts, convert_to_numpy=True)

            dim = embeddings.shape[1]
            temp_index = faiss.IndexFlatL2(dim)
            temp_index.add(embeddings)

            # Simple FAISS query
            query_vec = model.encode([query], convert_to_numpy=True)
            D, I = temp_index.search(query_vec, k=min(top_k, len(uploaded_texts)))
            retrieved_docs = [uploaded_texts[i] for i in I[0] if i < len(uploaded_texts)]

        else:
            # -------- FALLBACK TO PREBUILT FAISS INDEX --------
            st.info("No uploaded documents. Using prebuilt clinical trial index...")
            retrieved_docs = query_faiss(query, top_k)

        # -------- SUMMARIZATION VIA RAG PIPELINE --------
        combined_text = " ".join(retrieved_docs)
        prompt_text = f"Question: {query}\nContext: {combined_text}\nAnswer:"

        models = [
            "sshleifer/distilbart-cnn-12-6",
            "t5-small",
            "facebook/bart-large-cnn",
            "google/flan-t5-base",
            "google/pegasus-xsum",
            "facebook/bart-base"
        ]

        summaries = {}
        for model_name in models:
            with st.spinner(f"Generating answer with {model_name}..."):
                summary, time_taken = summarize_with_model(prompt_text, model_name)
                summaries[model_name] = {"summary": summary, "time": time_taken}

# ---------------- RETRIEVED SECTIONS ----------------
if query:
    st.subheader("📄 Retrieved Sections")
    for i, doc in enumerate(retrieved_docs, 1):
        with st.expander(f"Retrieved Section {i}"):
            st.write(doc[:1200] + ("..." if len(doc) > 1200 else ""))

# ---------------- MODEL COMPARISON ----------------
if query:
    st.subheader("🧪 RAG-Based Summary Comparison")
    for model, info in summaries.items():
        st.markdown(f"### 🤖 {model}")
        st.write(info["summary"])
        st.caption(f"⏱ Time taken: {info['time']} seconds")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "All summaries are generated via a Retrieval-Augmented Generation (RAG) pipeline."
)

# ---------------- DOWNLOAD OPTIONS ----------------
if query:
    st.subheader("💾 Download Options")

    import pandas as pd

    # Combined retrieved sections
    combined_text = " ".join(retrieved_docs)
    df_retrieved = pd.DataFrame({"Combined Retrieved Sections": [combined_text]})

    st.download_button(
        label="Download Retrieved Sections as CSV",
        data=df_retrieved.to_csv(index=False),
        file_name="retrieved_sections.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Retrieved Sections as TXT",
        data=combined_text,
        file_name="retrieved_sections.txt",
        mime="text/plain"
    )

    # Summaries per model
    summary_rows = []
    for model, info in summaries.items():
        summary_rows.append({
            "Model": model,
            "Summary": info["summary"],
            "Time Taken (s)": info["time"]
        })

    df_summaries = pd.DataFrame(summary_rows)
    st.download_button(
        label="Download Summaries as CSV",
        data=df_summaries.to_csv(index=False),
        file_name="rag_summaries.csv",
        mime="text/csv"
    )

    # TXT version of summaries
    txt_summaries = "\n\n".join([f"{row['Model']}:\n{row['Summary']}" 
                                 for _, row in df_summaries.iterrows()])
    st.download_button(
        label="Download Summaries as TXT",
        data=txt_summaries,
        file_name="rag_summaries.txt",
        mime="text/plain"
    )
