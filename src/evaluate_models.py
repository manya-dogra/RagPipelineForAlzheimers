import csv
import evaluate

# === Load evaluation metrics ===
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


def evaluate_summaries(summaries_dict, docs, top_n_docs=5):
    """
    summaries_dict: {"model_name": {"summary": "...", "time": ...}, ...}
    docs: list of original documents (text)
    top_n_docs: how many docs to use for reference (for consistency)
    """
    input_text = " ".join(docs[:top_n_docs])  # reference text
    results = {}

    for model_name, info in summaries_dict.items():
        summary_text = info["summary"]

        rouge_score = rouge.compute(predictions=[summary_text], references=[input_text])
        bleu_score = bleu.compute(predictions=[summary_text], references=[input_text])
        bert_score = bertscore.compute(predictions=[summary_text], references=[input_text], lang="en")

        results[model_name] = {
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score.get("rouge2", 0.0),
            "rougeL": rouge_score["rougeL"],
            "bleu": bleu_score["bleu"],
            "bertscore_f1": bert_score["f1"][0]
        }

        print(f"\n=== Metrics for {model_name} ===")
        print(f"ROUGE-1: {results[model_name]['rouge1']:.4f} | "
              f"ROUGE-2: {results[model_name]['rouge2']:.4f} | "
              f"ROUGE-L: {results[model_name]['rougeL']:.4f} | "
              f"BLEU: {results[model_name]['bleu']:.4f} | "
              f"BERTScore-F1: {results[model_name]['bertscore_f1']:.4f}")

    return results


if __name__ == "__main__":
    from retriever import load_index
    from pipeline import query_local_rag  # your pipeline that generates summaries

    # Retrieve docs and summaries using your RAG pipeline
    index, docs = load_index()
    query = "Which drugs are being studied for Alzheimer’s?"
    summaries_dict = query_local_rag(query)

    # Evaluate the summaries
    metrics_results = evaluate_summaries(summaries_dict, docs)

    # Save metrics to CSV
    with open("pipeline_model_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore-F1"])
        for model, metrics in metrics_results.items():
            writer.writerow([
                model,
                metrics["rouge1"],
                metrics["rouge2"],
                metrics["rougeL"],
                metrics["bleu"],
                metrics["bertscore_f1"]
            ])

    print("\nMetrics saved to pipeline_model_metrics.csv")
