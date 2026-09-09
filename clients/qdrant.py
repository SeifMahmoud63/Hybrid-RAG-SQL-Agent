from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from clients.ingestion_RAG import embedding_model
from helpers.config import settings
from helpers.logger import logger


# Connect to Qdrant service
logger.info(f"Connecting to Qdrant at http://{settings.QDRANT_HOST}:6333...")
client = QdrantClient(
    url=f"http://{settings.QDRANT_HOST}:6333",
)


def ensure_collection_exists():
    """Check if demo_collection exists; create it if not."""
    try:
        if not client.collection_exists("demo_collection"):
            logger.info("Collection 'demo_collection' not found. Creating it...")
            client.create_collection(
                collection_name="demo_collection",
                vectors_config=VectorParams(size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            logger.info("Collection 'demo_collection' created successfully.")
    except Exception as e:
        logger.error(f"Error checking/creating Qdrant collection: {str(e)}", exc_info=True)


# Ensure collection is ready
ensure_collection_exists()

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embedding_model()
)