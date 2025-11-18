# src/pipeline.py
from retriever import query_faiss
from transformers import pipeline as hf_pipeline

def query_local_rag(query, top_k=10):   # bump default for augmented corpora
    print("🔍 Retrieving relevant documents...")
    # Enable dedup in retriever; it will over-retrieve internally and collapse by source_nct
    results = query_faiss(query, top_k=top_k, dedup_by_source=True)
    print(f"✅ Retrieved {len(results)} unique documents after de-dup.\n")
    # Optionally print a short preview of each
    # for i, r in enumerate(results, 1):
    #     print(f"[{i}] {r[:220]}...\n")

    # Handle both Document objects and plain strings
    input_text = " ".join([r.page_content if hasattr(r, "page_content") else str(r) for r in results])

    # 🧠 Load local summarization model (lighter & safe)
    print("🧠 Loading local summarization model (distilbart)...")
    summarizer = hf_pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

    # ✂️ Split into chunks to fit within model limit
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
    answer = query_local_rag(query, top_k=10)  # raised from 3
    print(answer)
