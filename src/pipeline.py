# src/pipeline.py
from retriever import query_faiss
from transformers import pipeline as hf_pipeline
from long_context_summarizer import summarize_long_context
from models_config import MODEL_CAPABILITIES
import time
import os


# --------------------------
# Short-context summarization
# --------------------------
def summarize_with_model(text, model_name):
    """Generate summary for short-context HuggingFace models (chunked)."""
    max_chunk_size = 500
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    if not chunks:
        return "No content retrieved to summarize.", 0.0

    print(f"\nLoading HF model: {model_name}")
    try:
        summarizer = hf_pipeline("summarization", model=model_name, device=-1)

        summaries = []
        start = time.time()

        for i, chunk in enumerate(chunks):
            print(f"    → Summarizing chunk {i+1}/{len(chunks)}...")
            summary = summarizer(
                chunk,
                max_length=250,
                min_length=60,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)

        total_time = round(time.time() - start, 2)
        return " ".join(summaries), total_time

    except Exception as e:
        return f"Error with {model_name}: {e}", 0.0


# --------------------------
# Main pipeline
# --------------------------
def run_pipeline(text, query, selected_models=None, top_k=10):
    """
    text  : Full document (used by long-context models)
    query : RAG query (used by short-context models)
    """

    if selected_models is None:
        selected_models = list(MODEL_CAPABILITIES.keys())

    summaries = {}

    for model_name in selected_models:
        model_info = MODEL_CAPABILITIES.get(model_name)
        model_type = model_info.get("type", "short")

        print(f"\n=== Processing with {model_name} ({model_type}-context) ===")

        # --------------------------
        # LONG-CONTEXT MODELS
        # --------------------------
        if model_type == "long":

            provider = model_info.get("provider")

            # 🔐 Check API key availability
            if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
                print("⚠️ GEMINI_API_KEY not set. Skipping Gemini model.")
                continue

            if provider == "grok" and not os.getenv("GROK_API_KEY"):
                print("⚠️ GROK_API_KEY not set. Skipping Grok model.")
                continue

            start = time.time()
            try:
                summary = summarize_long_context(text, model_name)
                duration = round(time.time() - start, 2)

                summaries[model_name] = {
                    "summary": summary,
                    "time": duration
                }
                print(f" {model_name} finished in {duration} seconds")

            except Exception as e:
                summaries[model_name] = {
                    "summary": f"Error: {e}",
                    "time": 0.0
                }

        # --------------------------
        # SHORT-CONTEXT MODELS (RAG)
        # --------------------------
        else:
            print(" Retrieving relevant documents...")
            results = query_faiss(query, top_k=top_k, dedup_by_source=True)
            print(f" Retrieved {len(results)} unique documents.\n")

            input_text = " ".join(
                r.page_content if hasattr(r, "page_content") else str(r)
                for r in results
            )

            summary, duration = summarize_with_model(input_text, model_name)
            summaries[model_name] = {
                "summary": summary,
                "time": duration
            }

            print(f" {model_name} finished in {duration} seconds")

    return summaries


# --------------------------
# Example execution
# --------------------------
if __name__ == "__main__":
    query = "Which drugs are being studied for Alzheimer’s?"

    full_text_example = """
    Alzheimer’s disease clinical trial report:
    Drug: ExampleDrugX
    Phase: 3
    Sample Size: 500
    Outcomes: Improved cognitive function
    Limitations: Short follow-up period
    """

    print(f"Running pipeline for query: \"{query}\"")

    summaries = run_pipeline(full_text_example, query)

    print("\n" + "=" * 50)
    print("FINAL SUMMARY COMPARISON")
    print("=" * 50)

    for model, info in summaries.items():
        print(f"\n=== {model} ===")
        print(f"Time: {info['time']} seconds")
        print(f"Summary:\n{info['summary'][:500]}...\n")