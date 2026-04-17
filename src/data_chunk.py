import logging
from typing import List, Dict

from langchain.text_splitter import (
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

from src.es_config import ESClient

logger = logging.getLogger(__name__)

# Import semantic segmenter
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning(
        "langchain-experimental not installed. Semantic chunking will fall back to sentence splitting."
    )

# Based on the configuration, the documents are divided into blocks and indexed into Elasticsearch.
def index_documents(es_client: ESClient, docs: List[Dict], config: dict):
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    strategy = config["chunk_strategy"]
    embedding_model_name = config["embedding_model"]
    metadata_enrichment = config.get("metadata_enrichment", False)

    # Create a splitter based on the strategy
    splitter = _create_splitter(strategy, chunk_size, chunk_overlap, embedding_model_name)

    # Prepare the LangChain Document object
    lc_docs = [
        Document(page_content=d["text"], metadata={"doc_id": d["doc_id"]})
        for d in docs
    ]

    # Chunking
    if strategy == "semantic" and SEMANTIC_AVAILABLE:
        texts = [d["text"] for d in docs]
        metadatas = [{"doc_id": d["doc_id"]} for d in docs]
        chunks = splitter.create_documents(texts, metadatas=metadatas)
    else:
        chunks = splitter.split_documents(lc_docs)

    # Load the embedding model
    emb_model = SentenceTransformer(embedding_model_name)

    # Block indexing
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        doc_id = chunk.metadata["doc_id"]

        if metadata_enrichment:
            title = text[:50].strip()
            enriched_text = f"Title: {title}\nContent: {text}"
        else:
            enriched_text = text

        # Generate vector
        embedding = emb_model.encode(enriched_text).tolist()

        # Write to ES
        es_client.index_document(
            chunk_id=f"chunk_{i}_{doc_id}",
            doc_id=doc_id,
            text=enriched_text,
            embedding=embedding,
        )

    logger.info(f"Indexed {len(chunks)} chunks with config: "
                f"strategy={strategy}, size={chunk_size}, overlap={chunk_overlap}, "
                f"model={embedding_model_name}, metadata_enrichment={metadata_enrichment}")

# Create the corresponding text splitter based on the strategy
def _create_splitter(strategy: str, chunk_size: int, chunk_overlap: int, embedding_model_name: str):
    if strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == "sentence":
        # Use recursive character segmentation, and prioritize splitting by sentence delimiters (periods, exclamation marks, question marks, line breaks)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=False,
        )
    elif strategy == "semantic":
        if not SEMANTIC_AVAILABLE:
            logger.warning(
                "SemanticChunker not available (install langchain-experimental). "
                "Falling back to sentence splitting."
            )
            # Downgrade to the "sentence" strategy
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "。", "！", "？", "；", "，", " ", ""],
                keep_separator=False,
            )
        # An embedding model is required for semantic segmentation.
        emb_model = SentenceTransformer(embedding_model_name)
        return SemanticChunker(
            embeddings=emb_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            breakpoint_threshold_type="percentile",  # "standard_deviation", "interquartile"
        )
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}. "
                         f"Allowed: token, sentence, semantic")