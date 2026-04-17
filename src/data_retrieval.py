from sentence_transformers import SentenceTransformer
from src.es_config import ESClient

def retrieve(es_client: ESClient, query: str, config: dict, top_k: int = 10) -> list:
    retriever_type = config["retriever_type"]
    embedding_model_name = config["embedding_model"]
    emb_model = SentenceTransformer(embedding_model_name)
    query_emb = emb_model.encode(query).tolist()
    vector_field = "embedding"
    if retriever_type == "bm25":
        body = {"query": {"match": {"text": query}}}
    elif retriever_type == "dense":
        body = {
            "knn": [
                {
                    "field": vector_field,
                    "query_vector": query_emb,
                    "k": top_k,
                    "num_candidates": 100
                }
            ]
        }
    else:  # hybrid
        body = {
            "query": {"match": {"text": query}},
            "knn": [
                {
                    "field": vector_field,
                    "query_vector": query_emb,
                    "k": top_k,
                    "num_candidates": 100,
                    "boost": 0.5
                }
            ]
        }

    res = es_client.search(body, size=top_k)
    hits = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        hits.append({
            "doc_id": src["doc_id"],
            "text": src["text"],
            "score": hit["_score"]
        })
    return hits