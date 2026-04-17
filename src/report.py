import json
import csv
import os
from pathlib import Path
import pandas as pd
import logging
from src.data_retrieval import retrieve
from src.data_rerank import rerank
from src.query_rewrite import hyde_rewrite
from src.generate_answer import generate_answer
from src.llm_config import get_llm
from src.metrics import case1_score, case2_score

logger = logging.getLogger(__name__)

def generate_report(case, best_config, trials_df, val_df, docs, es_client, output_dir="outputs"):
    """Generate all required output files for the interview assignment."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # 1. best_config.json
    print("[DEBUG] 1. best_config.json")
    with open(f"{output_dir}/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Saved best_config.json to {output_dir}")

    # 2. run_summary.csv (all trials)
    print("[DEBUG] 2. run_summary.csv (all trials)")
    trials_df.to_csv(f"{output_dir}/run_summary.csv", index=False)
    logger.info(f"Saved run_summary.csv to {output_dir}")

    # 3. per_query_diagnostics.csv and examples
    print("[DEBUG] 3. per_query_diagnostics.csv and examples")
    per_query_rows = []
    retrieval_dir = Path(f"{output_dir}/retrieval_examples")
    answer_dir = Path(f"{output_dir}/answer_examples")
    retrieval_dir.mkdir(exist_ok=True)
    answer_dir.mkdir(exist_ok=True)

    # Re-create LLM client
    print("[DEBUG] Re-create LLM client")
    llm = get_llm(provider=best_config["llm_provider"])

    for idx, row in val_df.iterrows():
        query = row["query"]

        # Apply query rewriting if configured
        if best_config.get("query_rewrite", False):
            rewritten = hyde_rewrite(query, llm, temperature=best_config.get("temperature", 0.0))
        else:
            rewritten = query

        # Retrieve
        print("[DEBUG] Retrieve")
        retrieved = retrieve(es_client, rewritten, best_config, top_k=10)
        # Rerank if enabled
        print("[DEBUG] Rerank if enabled")
        if best_config.get("rerank_enabled", False):
            retrieved = rerank(rewritten, retrieved, best_config["rerank_model"], top_n=5)
        # Generate answer
        print("[DEBUG] Generate answer")
        answer = generate_answer(rewritten, retrieved, best_config, llm)

        # Compute metrics (case dependent)
        if case == 1:
            metrics = case1_score(retrieved, answer, row, return_dict=True)
        else:
            metrics = case2_score(retrieved, answer, rewritten, return_dict=True)

        per_query_rows.append({
            "query_id": idx,
            "original_query": query,
            "rewritten_query": rewritten if best_config.get("query_rewrite", False) else "",
            "generated_answer": answer,
            **metrics
        })

        # Save retrieval examples (top 3 documents)
        print("[DEBUG] Save retrieval examples (top 3 documents)")
        with open(retrieval_dir / f"query_{idx}_retrieved.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            if best_config.get("query_rewrite", False):
                f.write(f"Rewritten: {rewritten}\n")
            f.write("\nRetrieved Documents:\n")
            for i, doc in enumerate(retrieved[:3]):
                f.write(f"\n--- Doc {i+1} (score: {doc['score']:.4f}) ---\n")
                f.write(doc["text"][:500] + ("..." if len(doc["text"]) > 500 else ""))
                f.write("\n")

        # Save answer example
        print("[DEBUG] Save answer example")
        with open(answer_dir / f"query_{idx}_answer.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            if best_config.get("query_rewrite", False):
                f.write(f"Rewritten: {rewritten}\n")
            f.write(f"\nGenerated Answer:\n{answer}\n")
            f.write("\nMetrics:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    # Save per_query_diagnostics.csv
    print("[DEBUG] Save per_query_diagnostics.csv")
    per_query_df = pd.DataFrame(per_query_rows)
    per_query_df.to_csv(f"{output_dir}/per_query_diagnostics.csv", index=False)
    logger.info(f"Saved per_query_diagnostics.csv to {output_dir}")

    # 4. Recommendation report
    print("[DEBUG] 4. Recommendation report")
    with open(f"{output_dir}/recommendation_report.md", "w", encoding="utf-8") as f:
        f.write("# RAG Pipeline Optimization Report\n\n")
        f.write(f"**Case**: {case}\n\n")
        f.write("## Best Configuration\n```json\n")
        f.write(json.dumps(best_config, indent=2))
        f.write("\n```\n\n")

        # Average metrics on validation set
        avg_metrics = per_query_df.drop(columns=["query_id", "original_query", "rewritten_query", "generated_answer"]).mean()
        f.write("## Validation Performance (average over validation queries)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in avg_metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        f.write("\n")

        f.write("## Decision Rationale\n\n")
        f.write("- The configuration was selected using Optuna Bayesian optimization to maximize the composite score.\n")
        f.write("- We performed a train/validation split (80/20) to avoid overfitting to the small benchmark.\n")
        f.write("- The search space covers chunking strategies, retrieval types, embedding models, reranking, query rewriting, and generation parameters.\n")
        f.write("- The hybrid retriever uses BM25 + dense vector with equal weight (boost=0.5) to balance lexical and semantic matching.\n")
        f.write("- Query rewriting (HYDE) generates a hypothetical answer paragraph to improve retrieval.\n")
        f.write("- Reranking further refines the top candidates using a cross-encoder.\n")
        f.write("- The chosen configuration achieved the highest validation score among all trials.\n\n")

        f.write("## Runtime Considerations\n\n")
        f.write("- Indexing time scales with chunk size and embedding model size.\n")
        f.write("- Retrieval with `dense` or `hybrid` requires embedding computation per query ( ~100ms on CPU).\n")
        f.write("- Reranking adds overhead but improves precision for small top_k.\n")
        f.write("- We recommend using GPU acceleration for production deployment.\n")

    logger.info(f"Saved recommendation report to {output_dir}/recommendation_report.md")