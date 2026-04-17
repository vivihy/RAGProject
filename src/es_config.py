from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESClient:
    def __init__(self, host="http://localhost:9200", base_index="rag_chunks"):
        self.host = host
        self.base_index = base_index
        self.index_name = base_index          # The currently used index name (can be dynamically modified)
        self.es = None
        self._connect()

    def _connect(self):
        try:
            self.es = Elasticsearch(
                self.host,
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True,
                headers={"Content-Type": "application/json"}
            )
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch {info['version']['number']} at {self.host}")
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            raise

    def set_index_name(self, index_name: str):
        self.index_name = index_name
        logger.debug(f"Current index set to {self.index_name}")

    def index_exists(self) -> bool:
        return self.es.indices.exists(index=self.index_name)

# Obtain the dimension of the 'embedding' field in the current index. If the index does not exist or does not have this field, return None.
    def get_vector_dimension(self) -> int | None:
        if not self.index_exists():
            return None
        try:
            mapping = self.es.indices.get_mapping(index=self.index_name)
            props = mapping[self.index_name]['mappings']['properties']
            if 'embedding' in props and 'dims' in props['embedding']:
                return props['embedding']['dims']
        except Exception as e:
            logger.warning(f"Failed to get vector dimension: {e}")
        return None

    def create_index(self, dim: int):
        mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        logger.info(f"Created index {self.index_name} with dim={dim}")

    def delete_index(self):
        if self.index_exists():
            self.es.indices.delete(index=self.index_name)
            logger.info(f"Deleted index {self.index_name}")
# Delete and rebuild the index (make sure the dimensions are correct)
    def recreate_index(self, dim: int):
        if self.index_exists():
            self.es.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index {self.index_name}")
        self.create_index(dim)

# Ensure that the index exists and that the dimensions are matched.
    def ensure_index(self, dim: int) -> bool:
        if not self.index_exists():
            self.create_index(dim)
            return True
        else:
            existing_dim = self.get_vector_dimension()
            if existing_dim == dim:
                logger.info(f"Index {self.index_name} already exists with correct dimension {dim}, reusing")
                return True
            else:
                logger.warning(f"Index {self.index_name} has dimension {existing_dim}, but required {dim}. "
                               f"Please delete or recreate it.")
                return False

    def index_document(self, chunk_id: str, doc_id: str, text: str, embedding: list):
        doc = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "embedding": embedding
        }
        self.es.index(index=self.index_name, document=doc)

    def search(self, body: dict, size: int = 10):
        return self.es.search(index=self.index_name, body=body, size=size)