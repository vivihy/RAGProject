import argparse
import logging
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Suppress redundant logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.WARNING)

from src.loader import load_corpus, load_case1, load_case2
from src.es_config import ESClient
import src.optimizer as opt
from src.report import generate_report
from src.llm_config import get_llm
from src.data_retrieval import retrieve
from src.data_rerank import rerank
from src.query_rewrite import hyde_rewrite
from src.generate_answer import generate_answer
from src.metrics import case1_score, case2_score

# Evaluate the configuration on the given dataset and return the average score as well as a list of scores for each row.
def evaluate_config(config, queries_df, docs, es_client, case):
    opt.ensure_index(es_client, docs, config)
    llm = get_llm(provider=config["llm_provider"])
    scores = []
    for _, row in queries_df.iterrows():
        query = row["query"]
        if config["query_rewrite"]:
            query = hyde_rewrite(query, llm, temperature=config["temperature"])
        retrieved = retrieve(es_client, query, config, top_k=10)
        if config["rerank_enabled"]:
            retrieved = rerank(query, retrieved, config["rerank_model"], top_n=5)
        answer = generate_answer(query, retrieved, config, llm)
        if case == 1:
            score = case1_score(retrieved, answer, row)
        else:
            score = case2_score(retrieved, answer, query)
        scores.append(score)
    return sum(scores) / len(scores), scores


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Optimizer")
    parser.add_argument("--case", type=int, choices=[1, 2], required=True,
                        help="Case 1 (with reference answers) or Case 2 (without)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (ignored if --report_only)")
    parser.add_argument("--test_ratio", type=float, default=0.0,
                        help="Holdout test ratio (ignored if --report_only)")
    parser.add_argument("--report_only", action="store_true",
                        help="Skip optimization, generate report from existing outputs/")
    args = parser.parse_args()

    docs = load_corpus("data/reference_corpus.jsonl")
    if args.case == 1:
        queries_df = load_case1("data/case1_eval_dataset.csv")
        print("Case 1")
    else:
        queries_df = load_case2("data/case2_query_doc_dataset.csv")
        print("Case 2")

    es_client = ESClient()

    # report
    if args.report_only:
        best_config_path = "outputs/best_config.json"
        run_summary_path = "outputs/run_summary.csv"
        if not Path(best_config_path).exists():
            raise FileNotFoundError(f"Missing {best_config_path}. Please run optimization first or remove --report_only.")

        with open(best_config_path, 'r') as f:
            best_config = json.load(f)

        if Path(run_summary_path).exists():
            trials_df = pd.read_csv(run_summary_path)
        else:
            trials_df = pd.DataFrame()

        # Re-divide the validation set (with the same random seed as during optimization)
        _, val_df = train_test_split(queries_df, test_size=0.2, random_state=42)

        generate_report(args.case, best_config, trials_df, val_df, docs, es_client)
        print("Report generated from existing outputs.")
        return

    best_config, study, poor_params, val_df, test_df = opt.optimize(
        args.case, queries_df, docs, es_client,
        n_trials=args.trials,
        test_ratio=args.test_ratio
    )

    generate_report(args.case, best_config, study.trials_dataframe(), val_df, docs, es_client)

    if test_df is not None:
        test_score, test_scores_list = evaluate_config(best_config, test_df, docs, es_client, args.case)
        with open("outputs/recommendation_report.md", "a", encoding="utf-8") as f:
            f.write("\n\n## Holdout Test Set Performance\n\n")
            f.write(f"**Average Score on Test Set**: {test_score:.4f}\n\n")
        pd.DataFrame([{"score": s} for s in test_scores_list]).to_csv("outputs/test_diagnostics.csv", index=False)
        print(f"Test set average score: {test_score:.4f}")

    print("Optimization finished.")
    print("Best config:", best_config)
    print("Diagonal analysis:", poor_params)


if __name__ == "__main__":
    main()