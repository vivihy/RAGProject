import optuna
import numpy as np
import hashlib
import logging
from sklearn.model_selection import train_test_split
from src.es_config import ESClient
from src.data_chunk import index_documents
from src.data_retrieval import retrieve
from src.data_rerank import rerank
from src.query_rewrite import hyde_rewrite
from src.generate_answer import generate_answer
from src.llm_config import get_llm
from src.metrics import case1_score, case2_score
from src.config import load_search_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_CACHE = {}

EMBEDDING_DIM_MAP = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en": 768,
    "all-mpnet-base-v2": 768,
    "intfloat/e5-base-v2": 768,
    "BAAI/bge-small-en": 384,
}

def get_embedding_dim(model_name: str) -> int:
    if model_name in EMBEDDING_DIM_MAP:
        return EMBEDDING_DIM_MAP[model_name]
    import re
    match = re.search(r'(\d{3})', model_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unknown embedding model: {model_name}. Please add its dimension to EMBEDDING_DIM_MAP.")

def _get_config_hash(config: dict) -> str:
    key_fields = {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "chunk_strategy": config["chunk_strategy"],
        "embedding_model": config["embedding_model"],
        "metadata_enrichment": config.get("metadata_enrichment", False),
    }
    key_str = str(sorted(key_fields.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


# Ensure that the index exists and its dimensions are matched. Use caching to avoid redundant construction.
def ensure_index(es_client: ESClient, docs, config: dict) -> str:
    config_hash = _get_config_hash(config)
    required_dim = get_embedding_dim(config["embedding_model"])

    # If there is a cache entry and the index actually exists with the correct dimensions, simply reuse it.
    if config_hash in INDEX_CACHE:
        index_name = INDEX_CACHE[config_hash]
        es_client.set_index_name(index_name)
        if es_client.index_exists() and es_client.get_vector_dimension() == required_dim:
            logger.info(f"Reusing cached index: {index_name}")
            return index_name
        else:
            logger.warning(f"Cached index {index_name} invalid, rebuilding...")
            del INDEX_CACHE[config_hash]

    base_name = getattr(es_client, 'base_index', "rag_chunks")
    index_name = f"{base_name}_{config_hash}"
    es_client.set_index_name(index_name)

    # Check whether the index exists and its dimensions are correct.
    if es_client.index_exists():
        existing_dim = es_client.get_vector_dimension()
        if existing_dim == required_dim:
            logger.info(f"Index {index_name} already exists with correct dimension {required_dim}, reusing")
            INDEX_CACHE[config_hash] = index_name
            return index_name
        else:
            logger.warning(f"Index {index_name} has dimension {existing_dim}, but required {required_dim}. Deleting...")
            es_client.delete_index()
            logger.info(f"Re-creating index {index_name} with dimension {required_dim}")
            es_client.create_index(required_dim)
            index_documents(es_client, docs, config)
            INDEX_CACHE[config_hash] = index_name
            return index_name
    else:
        # The index does not exist. Create it directly and fill it in.
        logger.info(f"Creating new index: {index_name} with dimension {required_dim}")
        es_client.create_index(required_dim)
        index_documents(es_client, docs, config)
        INDEX_CACHE[config_hash] = index_name
        return index_name

# Run the optimization loop and support the optional holdout test set
def optimize(case, queries_df, docs, es_client, n_trials=30, test_ratio=0.0):
    if test_ratio > 0:
        train_val_df, test_df = train_test_split(queries_df, test_size=test_ratio, random_state=42)
        val_ratio = 0.2 / (1 - test_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=42)
        logger.info(f"Holdout enabled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    else:
        train_df, val_df = train_test_split(queries_df, test_size=0.2, random_state=42)
        test_df = None
        logger.info(f"No holdout: train={len(train_df)}, val={len(val_df)}")

    def objective(trial):
        config = {}
        search_space = load_search_space()

        config["chunk_size"] = trial.suggest_categorical("chunk_size", search_space["chunking"]["size"])
        config["chunk_overlap"] = trial.suggest_categorical("chunk_overlap", search_space["chunking"]["overlap"])
        config["chunk_strategy"] = trial.suggest_categorical("chunk_strategy", search_space["chunking"]["strategy"])
        config["retriever_type"] = trial.suggest_categorical("retriever_type", search_space["indexing"]["retriever"])
        config["embedding_model"] = trial.suggest_categorical("embedding_model", search_space["indexing"]["embedding_model"])
        config["metadata_enrichment"] = trial.suggest_categorical("metadata_enrichment", search_space["indexing"]["metadata_enrichment"])
        config["rerank_enabled"] = trial.suggest_categorical("rerank_enabled", search_space["reranking"]["enabled"])
        config["rerank_model"] = trial.suggest_categorical("rerank_model", search_space["reranking"]["model"])
        config["query_rewrite"] = trial.suggest_categorical("query_rewrite", search_space["query_refinement"]["rewrite"])
        config["llm_provider"] = trial.suggest_categorical("llm_provider", search_space["generation"]["llm"])
        config["temperature"] = trial.suggest_categorical("temperature", search_space["generation"]["temperature"])
        config["answer_style"] = trial.suggest_categorical("answer_style", search_space["generation"]["answer_style"])

        try:
            ensure_index(es_client, docs, config)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return -1.0

        llm = get_llm(provider=config["llm_provider"])

        scores = []
        for _, row in train_df.iterrows():
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

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = study.trials_dataframe()
    poor_trials = trials_df[trials_df["value"] < trials_df["value"].quantile(0.25)]
    poor_params = poor_trials.mean(numeric_only=True).to_dict()

    best_config = study.best_params
    search_space = load_search_space()
    best_config["rerank_model"] = best_config.get("rerank_model", search_space["reranking"]["model"][0])
    best_config["answer_style"] = best_config.get("answer_style", "concise")
    best_config["llm_provider"] = best_config.get("llm_provider", "zhipu-glm4-flash")
    best_config["rerank_enabled"] = best_config.get("rerank_enabled", False)
    best_config["query_rewrite"] = best_config.get("query_rewrite", False)
    best_config["temperature"] = best_config.get("temperature", 0.0)
    best_config["metadata_enrichment"] = best_config.get("metadata_enrichment", False)

    return best_config, study, poor_params, val_df, test_df