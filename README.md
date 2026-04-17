# RAG Pipeline Optimizer

**Automated RAG pipeline optimizer** – Automatically search for the optimal strategies for partitioning, indexing, reordering, query rewriting and generation, applicable to two evaluation scenarios:

- **Case 1**：With reference answers and relevant context → Maximize the alignment of search results and answer quality.  
- **Case 2**：Only queries and documents (without reference answers) → Maximize grounding (faithfulness) and coverage.

Use **Optuna** Perform Bayesian hyperparameter optimization, with built-in index caching, hybrid retrieval, reordering, HyDE query rewriting, and generate a complete experimental report.
- This experiment has deployed the project to Alibaba Cloud.
---

## Functional features

- Configurable search space (block size/strategy, retrieval type, embedding model, reordering, query rewriting, generation parameters) 
- Two-scenario evaluation metrics (strong supervision / weak supervision) 
- The index is cached based on the configuration hash to avoid redundant construction, significantly accelerating the optimization process. 
- Support Elasticsearch as a vector/keyword search engine (BM25, dense kNN, hybrid search) 
- Automatically generate all necessary outputs: optimal configuration JSON, run summary CSV, query-by-query diagnosis CSV, retrieval/answer examples, recommendation report 
- Support train/validation/test split to prevent overfitting 
- Modular design, facilitating the expansion of new strategies or LLMs

---

## System Requirements

- Python 3.9+
- Elasticsearch 8.x (local or remote)
- (Optional) GPU acceleration (sentence-transformers, cross-encoder)

---

## Installation

### Clone

```bash
git clone <your-repo-url>
cd rag-optimizer
```

###  Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
###  Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-experimental
```


###  Run Elasticsearch

```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
curl http://localhost:9200
```


## Configuration

### Environment variable (LLM API key)
```bash
# This project has taken into account the consumption of resource costs and has chosen Zhipu AI (with glm-4-flash as the default option)
ZHIPU_API_KEY=your_key_here

# Optional: OpenAI
OPENAI_API_KEY=your_key_here
```
### Search space configuration
Edit configs/optimizer_config_template.json：

- search_space：Define the candidate values for each hyperparameter (block size, retrieval type, embedding model, reordering model, generation parameters, etc.)
- model_registry：Register LLM providers along with their corresponding environment variables and API addresses
The search space can be freely adjusted, and the optimizer will automatically explore it.

## Data Preparation
Place the following file in the "data/" directory.
- reference_corpus.jsonl
- case1_eval_dataset.csv
- case2_query_doc_dataset.csv

## Run the optimizer
```bash
python main.py --case {1|2} [--trials N] [--test_ratio R] [--report_only]
```
For example
```bash
# Case 1 Optimize and conduct 30 trials. Retain 20% of the data as the test set.
python main.py --case 1 --trials 30 --test_ratio 0.2

# Case 2 Optimization, rapid verification (10 trials)
python main.py --case 2 --trials 10

# Generate the report based solely on the existing results (without re-running the optimization)
python main.py --case 1 --report_only
```

## Output file
All the results are saved in the "outputs/" directory:
- best_config.json
- run_summary.csv
- per_query_diagnostics.csv
- retrieval_examples/
- answer_examples/
- recommendation_report.md
- test_diagnostics.csv

Due to resource constraints, only 3 trials will be run this time. The generated case results have been placed in the 'outputs_case_result' folder. This result is for reference and comparison only. For more accurate analysis, please reset the number of trials.

## Explanation of Evaluation Indicators
### Case 1
- Search term: Recall@5, MRR 
Answer: BERTScore and F1 
- Context relevance: The maximum cosine similarity between the retrieved document and the reference_relevant_context 
Overall score: Weighted sum (default weights: 0.3 for recall + 0.4 for BERTScore + 0.3 for context similarity)

### Case 2
- All the quantities have been normalized to the range [0,1]: 
- Contextual relevance: CrossEncoder assesses the query-document relevance (using sigmoid normalization) 
- Accuracy: CrossEncoder assesses the relevance between the generated answer and the retrieved document (using the highest score from the top 3) 
Answer Relevance: Cosine similarity between the generated answer and the query 
Overall score: 0.4 × Context relevance + 0.3 × Accuracy + 0.3 × Answer relevance

## License
This project is designed solely for interview assessment and does not include an open-source license.