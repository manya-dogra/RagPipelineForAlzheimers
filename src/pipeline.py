from retriever import query_faiss
from transformers import pipeline as hf_pipeline
import time


def summarize_with_model(text, model_name):
    """Generate summary using a specific model and measure time."""
    max_chunk_size = 500
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    if not chunks:
        return "No content retrieved to summarize.", 0.0

    print(f"\n Loading model: {model_name}")
    try:
        summarizer = hf_pipeline("summarization", model=model_name, device=-1)

        summaries = []
        start = time.time()
        for i, chunk in enumerate(chunks):
            print(f"    → Summarizing chunk {i+1}/{len(chunks)}...")
            summary = summarizer(chunk, max_length=250, min_length=60, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        end = time.time()

        total_time = round(end - start, 2)
        final_summary = " ".join(summaries)
        return final_summary, total_time

    except Exception as e:
        error_message = f"Error summarizing with {model_name}: {e}"
        print(f" {error_message}")
        return error_message, 0.0


def query_local_rag(query, top_k=10):
    """
    Performs RAG with over-retrieval/deduplication, and summarizes the results
    using multiple models for comparison.
    """
    print("🔍 Retrieving relevant documents...")
    results = query_faiss(query, top_k=top_k, dedup_by_source=True)
    print(f" Retrieved {len(results)} unique documents after de-dup.\n")

    input_text = " ".join([r.page_content if hasattr(r, 'page_content') else str(r) for r in results])

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
        summary, duration = summarize_with_model(input_text, model_name)
        summaries[model_name] = {"summary": summary, "time": duration}
        print(f"\n {model_name} finished in {duration} seconds\n")

    return summaries


if __name__ == "__main__":
    query = "Which drugs are being studied for Alzheimer’s?"
    print(f"Running RAG pipeline for query: \"{query}\"")
    summaries = query_local_rag(query)

    print("\n\n" + "="*50)
    print("FINAL SUMMARY COMPARISON")
    print("="*50)
    for model, info in summaries.items():
        print(f"\n=== {model} ===")
        print(f"Time: {info['time']} seconds")
        print(f"Summary:\n{info['summary'][:500]}...\n")