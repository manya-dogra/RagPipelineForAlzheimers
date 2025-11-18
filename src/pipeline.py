# src/pipeline.py
from retriever import query_faiss
from transformers import pipeline as hf_pipeline
from textwrap import dedent
import time


def summarize_with_model(text, model_name):
    """Generate summary using a specific model and measure time."""
    print(f"\n🧠 Loading model: {model_name}")
    summarizer = hf_pipeline("summarization", model=model_name, device=-1)
    max_chunk_size = 500
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    summaries = []
    start = time.time()
    for i, chunk in enumerate(chunks):
        print(f"   → Summarizing chunk {i+1}/{len(chunks)}...")
        summary = summarizer(chunk, max_length=250, min_length=60, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    end = time.time()

    total_time = round(end - start, 2)
    final_summary = " ".join(summaries)
    return final_summary, total_time


def query_local_rag(query, top_k=3):
    print("🔍 Retrieving relevant documents...")
    results = query_faiss(query, top_k=top_k)
    print(f"✅ Retrieved {len(results)} relevant documents.\n")

    input_text = " ".join([r.page_content if hasattr(r, 'page_content') else str(r) for r in results])

    # Summarize using both models
    models = [
        "sshleifer/distilbart-cnn-12-6",
        "t5-small"
    ]
    summaries = {}

    for model_name in models:
        summary, duration = summarize_with_model(input_text, model_name)
        summaries[model_name] = {"summary": summary, "time": duration}
        print(f"\n✅ {model_name} finished in {duration} seconds\n")

    return summaries


if __name__ == "__main__":
    query = "Which drugs are being studied for Alzheimer’s?"
    summaries = query_local_rag(query)

    for model, info in summaries.items():
        print(f"\n=== {model} ===")
        print(f"Time: {info['time']} seconds")
        print(f"Summary:\n{info['summary'][:500]}...\n")
