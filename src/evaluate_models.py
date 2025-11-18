import time
import csv
from transformers import pipeline as hf_pipeline
import evaluate
from retriever import load_index  

def compare_models(query):
    index, docs = load_index()
    model_names = [
        "sshleifer/distilbart-cnn-12-6",
        "t5-small",
        "facebook/bart-large-cnn",
        "google/flan-t5-base",
        "google/pegasus-xsum",
        "facebook/bart-base"
    ]

    results = {}
    input_text = " ".join(docs[:5])  

    max_chunk_size = 350
    words = input_text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")

    for model_name in model_names:
        print(f"\n=== Evaluating {model_name} ===")
        summarizer = hf_pipeline("summarization", model=model_name, device=-1)

        start = time.time()
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"   → Summarizing chunk {i+1}/{len(chunks)}...")
            try:
                summary = summarizer(chunk, max_length=250, min_length=60, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                print(f" Skipping chunk {i+1} due to error: {e}")
        end = time.time()

        final_summary = " ".join(summaries)
        reference = input_text[:1000]  

        rouge_score = rouge.compute(predictions=[final_summary], references=[reference])
        bleu_score = bleu.compute(predictions=[final_summary], references=[reference])
        bert_score = bertscore.compute(predictions=[final_summary], references=[reference], lang="en")

        results[model_name] = {
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score.get("rouge2", 0.0),
            "rougeL": rouge_score["rougeL"],
            "bleu": bleu_score["bleu"],
            "bertscore_f1": bert_score["f1"][0],
            "time": round(end - start, 2),
            "summary": final_summary[:400]
        }

        print(f"Time: {round(end - start, 2)}s | ROUGE-1: {rouge_score['rouge1']:.4f} | "
              f"ROUGE-2: {results[model_name]['rouge2']:.4f} | ROUGE-L: {rouge_score['rougeL']:.4f} | "
              f"BLEU: {bleu_score['bleu']:.4f} | BERTScore-F1: {bert_score['f1'][0]:.4f}")
        print(" Summary snippet:", final_summary[:250])

    return results


if __name__ == "__main__":
    print("Running model evaluation...")
    query = "Alzheimer's clinical trial summaries"
    results = compare_models(query)

    # === Save results to CSV ===
    with open("model_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore-F1", "Time (s)", "Summary Snippet"])
        for model, metrics in results.items():
            writer.writerow([
                model,
                metrics["rouge1"],
                metrics["rouge2"],
                metrics["rougeL"],
                metrics["bleu"],
                metrics["bertscore_f1"],
                metrics["time"],
                metrics["summary"]
            ])
    print("\n Results saved to model_metrics.csv")
