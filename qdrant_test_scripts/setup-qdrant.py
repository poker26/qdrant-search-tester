"""
Настройка Qdrant коллекции для BGE-M3 hybrid search.
Коллекция distill_hybrid_v2: dense (1024d) + sparse (BGE-M3 learned).
"""
import os
import sys
import time
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_client import get_embedding_client, EMBEDDING_DIMS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantSetup:
    def __init__(self):
        url = os.getenv('QDRANT_URL')
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', '6333'))
        api_key = os.getenv('QDRANT_API_KEY')

        if url:
            kwargs = {"url": url, "check_compatibility": False}
            if api_key:
                kwargs["api_key"] = api_key
            self.client = QdrantClient(**kwargs)
        else:
            self.client = QdrantClient(host=host, port=port)

        self.embedding_client = get_embedding_client()
        self.embedding_dim = self.embedding_client.get_embedding_dim()
        logger.info(f"Модель: {self.embedding_client.get_model_name()}, dim={self.embedding_dim}")

    def create_collection_v2(self, collection_name: str = "distill_hybrid_v2", recreate: bool = False):
        """Создание коллекции для BGE-M3: dense + один sparse."""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == collection_name for c in collections.collections)
            if exists:
                if recreate:
                    logger.info(f"Удаляю {collection_name}...")
                    self.client.delete_collection(collection_name)
                    time.sleep(1)
                else:
                    logger.info(f"{collection_name} уже существует")
                    return True
        except Exception:
            pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )
        logger.info(f"✅ Коллекция {collection_name} создана (dense={self.embedding_dim}d + sparse)")

        # Payload indexes
        for field, schema in [
            ("recipe_name", models.PayloadSchemaType.TEXT),
            ("category", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema
                )
            except Exception:
                pass

        return True

    def check_collection(self, collection_name: str):
        try:
            info = self.client.get_collection(collection_name)
            count = self.client.count(collection_name=collection_name).count
            logger.info(f"{collection_name}: {count} points, status={info.status}")
            return True
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=os.getenv('COLLECTION_NAME', 'distill_hybrid_v2'))
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    setup = QdrantSetup()

    if args.recreate:
        setup.create_collection_v2(args.collection, recreate=True)
    elif args.check:
        setup.check_collection(args.collection)
    else:
        setup.create_collection_v2(args.collection)
        setup.check_collection(args.collection)
