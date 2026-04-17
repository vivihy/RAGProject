from sentence_transformers import CrossEncoder

_rerankers = {}  # key: model_name, value: CrossEncoder instance

def get_reranker(model_name: str):
    global _rerankers
    if model_name not in _rerankers:
        _rerankers[model_name] = CrossEncoder(model_name)
    return _rerankers[model_name]

def rerank(query: str, retrieved_docs: list, model_name: str, top_n: int = 5) -> list:
    if not retrieved_docs:
        return []
    reranker = get_reranker(model_name)
    pairs = [[query, doc["text"]] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
    sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    return sorted_docs[:top_n]