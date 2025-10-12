# src/pipeline.py
from retriever import query_faiss
from transformers import pipeline as hf_pipeline
from textwrap import dedent


def query_local_rag(query,top_k=3):
    print("🔍 Retrieving relevant documents...")
    results = query_faiss(query, top_k=top_k)
    print(f"✅ Retrieved {len(results)} relevant documents.\n")
    print("Results",results)

# Handle both Document objects and plain strings
    input_text = " ".join([r.page_content if hasattr(r, "page_content") else str(r) for r in results])

    # 🧠 Load local summarization model (lighter & safe)
    print("🧠 Loading local summarization model (distilbart)...")
    from transformers import pipeline as hf_pipeline
    summarizer = hf_pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # smaller, same structure
        device=-1
    )

    # ✂️ Split text into 1000-token chunks to fit within model limit
    max_chunk_size = 500
    words = input_text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    print("🧩 Generating summary...")
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"   → Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=250, min_length=60, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    final_summary = " ".join(summaries)
    print("\n✅ Final summarized output ready.\n")
    return final_summary

if __name__ == "__main__":
    query = "Which drugs are being studied for Alzheimer’s?"
    answer = query_local_rag(query,3)
    print(answer)