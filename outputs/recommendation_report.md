### Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| Elasticsearch connection lost | Exception on search/index | Retry with exponential backoff (max 3 retries); abort with clear error if persistent |
| Empty retrieval results | Retrieved docs list empty | Use original query (if rewritten) or return "no answer"; score set to 0 |
| LLM API timeout / rate limit | HTTP 4xx/5xx error | Retry once; if still fails, return placeholder "Unable to generate answer" |
| Index dimension mismatch | `get_vector_dimension()` returns different dim | Automatically delete and rebuild index with correct dimension |
| BERTScore / Cross-encoder OOM | MemoryError | Reduce batch size or fallback to smaller model (e.g., MiniLM instead of BGE) |
| Overfitting to small benchmark | Train/val split + test set | Report validation and test scores; use cross-validation if needed |