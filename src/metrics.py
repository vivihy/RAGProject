import numpy as np
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the global model
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))  # 防止溢出

# Calculate the retrieval indicators
def retrieval_metrics(retrieved_docs, gold_doc_ids, k=5):
    retrieved_ids = [d["doc_id"] for d in retrieved_docs[:k]]
    relevant = set(gold_doc_ids)
    if not relevant:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0}
    recall = len(set(retrieved_ids) & relevant) / len(relevant)
    precision = len(set(retrieved_ids) & relevant) / k
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            mrr = 1 / (i + 1)
            break
    return {"recall": recall, "precision": precision, "mrr": mrr}

# Use BERTScore to calculate the similarity (F1 score) between the generated answer and the reference answer.
def answer_similarity(generated, reference):
    _, _, F1 = bert_scorer.score([generated], [reference])
    return F1.item()

# When there is a reference answer: Calculate the maximum cosine similarity between the retrieved documents and the reference context.
def context_relevance_with_ref(retrieved_docs, reference_context):
    if not retrieved_docs:
        return 0.0
    retrieved_texts = [d["text"] for d in retrieved_docs[:3]]
    if not retrieved_texts:
        return 0.0
    ref_emb = sim_model.encode([reference_context])
    doc_embs = sim_model.encode(retrieved_texts)
    sims = cosine_similarity(ref_emb, doc_embs)[0]
    return float(np.max(sims))

def case1_score(retrieved_docs, generated_answer, row, alpha=0.3, beta=0.4, gamma=0.3, return_dict=False):
    ret_met = retrieval_metrics(retrieved_docs, row["reference_doc_ids"], k=5)
    recall = ret_met["recall"]
    ans_sim = answer_similarity(generated_answer, row["reference_answer"])
    ctx_rel = context_relevance_with_ref(retrieved_docs, row["reference_relevant_context"])
    total = alpha * recall + beta * ans_sim + gamma * ctx_rel
    if return_dict:
        return {
            "recall": recall,
            "answer_similarity": ans_sim,
            "context_relevance": ctx_rel,
            "total_score": total
        }
    return total

def context_relevance_unsupervised(query, retrieved_docs):
    if not retrieved_docs:
        return 0.0
    pairs = [(query, doc["text"]) for doc in retrieved_docs[:3]]
    scores = cross_encoder.predict(pairs)  # 原始输出可能是 logits（负值或正值）
    # 归一化到 (0,1)
    norm_scores = sigmoid(np.array(scores))
    return float(np.mean(norm_scores))

def faithfulness_unsupervised(generated_answer, retrieved_docs):
    if not retrieved_docs:
        return 0.0
    pairs = [(generated_answer, doc["text"]) for doc in retrieved_docs[:3]]
    scores = cross_encoder.predict(pairs)
    norm_scores = sigmoid(np.array(scores))
    return float(np.max(norm_scores))

# When there is no reference answer: Calculate the semantic similarity between the generated answer and the query.
def answer_relevance_unsupervised(query, generated_answer):
    emb_q = sim_model.encode([query])
    emb_a = sim_model.encode([generated_answer])
    sim = cosine_similarity(emb_q, emb_a)[0][0]
    return float((sim + 1) / 2)

def case2_score(retrieved_docs, generated_answer, query, return_dict=False):
    ctx_rel = context_relevance_unsupervised(query, retrieved_docs)
    faithful = faithfulness_unsupervised(generated_answer, retrieved_docs)
    ans_rel = answer_relevance_unsupervised(query, generated_answer)
    total = 0.4 * ctx_rel + 0.3 * faithful + 0.3 * ans_rel
    if return_dict:
        return {
            "context_relevance": ctx_rel,
            "faithfulness": faithful,
            "answer_relevance": ans_rel,
            "total_score": total
        }
    return total