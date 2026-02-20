"""
Скрипт для настройки Qdrant и загрузки тестовых данных.
Поддерживает OpenAI и self-hosted модели (bgm-m3).
"""
import json
import time
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging
from dotenv import load_dotenv

# Добавляем путь к embedding_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_client import get_embedding_client, EMBEDDING_DIMS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantSetup:
    def __init__(self, host=None, port=None, url=None, api_key=None):
        qdrant_url = url or os.getenv('QDRANT_URL')
        qdrant_host = host or os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = port or int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = api_key or os.getenv('QDRANT_API_KEY')

        if qdrant_url:
            logger.info(f"Подключение к облачному Qdrant: {qdrant_url}")
            if qdrant_api_key:
                self.client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    check_compatibility=False
                )
            else:
                self.client = QdrantClient(url=qdrant_url, check_compatibility=False)
        else:
            logger.info(f"Подключение к локальному Qdrant: {qdrant_host}:{qdrant_port}")
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Инициализируем клиент эмбеддингов (OpenAI или bgm-m3)
        try:
            self.embedding_client = get_embedding_client()
            embedding_type = os.getenv('EMBEDDING_MODEL', 'bgm-m3').lower()
            self.embedding_dim = EMBEDDING_DIMS.get(embedding_type, 1024)
            logger.info(f"Используется модель эмбеддингов: {self.embedding_client.get_model_name()} (размерность: {self.embedding_dim})")
        except Exception as e:
            raise ValueError(f"Ошибка инициализации клиента эмбеддингов: {e}")
        
    def delete_collection(self, collection_name: str) -> bool:
        """Удаление коллекции"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Коллекция {collection_name} удалена")
            time.sleep(1)
            return True
        except Exception as e:
            logger.error(f"Ошибка удаления коллекции: {e}")
            return False

    def create_collection(
        self,
        collection_name: str = "test_recipes",
        vector_size: int = None,
        recreate: bool = False
    ) -> bool:
        """Создание коллекции с нужной схемой"""
        if vector_size is None:
            vector_size = self.embedding_dim
        """Создание коллекции с нужной схемой (OpenAI text-embedding-3-small, размер 1536)"""
        try:
            collections = self.client.get_collections()
            existing = any(c.name == collection_name for c in collections.collections)

            if existing:
                if recreate:
                    logger.info(f"Пересоздание коллекции {collection_name}...")
                    self.delete_collection(collection_name)
                else:
                    logger.info(f"Коллекция {collection_name} уже существует")
                    return True
        except Exception as e:
            if "403" in str(e) or "forbidden" in str(e).lower():
                logger.warning("Нет прав на чтение списка коллекций. Пробуем создать напрямую...")
            else:
                raise

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "categories": models.SparseVectorParams(),
                    "ingredients": models.SparseVectorParams(),
                    "measurements": models.SparseVectorParams(),
                    "spices_herbs": models.SparseVectorParams(),
                    "techniques": models.SparseVectorParams()
                }
            )
            logger.info(f"Коллекция {collection_name} создана (размер вектора: {vector_size})")
            time.sleep(1)
            return True
        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            return False
    
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Получение эмбеддингов через универсальный клиент (батчами)"""
        embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_client.get_embedding(batch)
            if isinstance(batch_embeddings[0], list):
                embeddings.extend(batch_embeddings)
            else:
                embeddings.append(batch_embeddings)
        return embeddings

    def load_recipes_data(self):
        """Загрузка и подготовка данных рецептов с эмбеддингами через OpenAI"""
        with open('data/recipes_structured.json', 'r', encoding='utf-8') as f:
            structured = json.load(f)['recipes']

        with open('data/recipes_full_text.json', 'r', encoding='utf-8') as f:
            full_texts = json.load(f)['texts']

        text_map = {text['id']: text['full_text'] for text in full_texts}
        texts_for_embedding = []

        for recipe in structured:
            recipe_id = recipe['id']
            full_text = text_map.get(recipe_id, '')
            text_for_embedding = (
                f"{recipe['name']} {recipe['subtitle']} "
                f"{recipe['preparation']['description']} {full_text[:500]}"
            )
            texts_for_embedding.append(text_for_embedding)

        logger.info(f"Получение эмбеддингов через {self.embedding_client.get_model_name()}...")
        embeddings_list = self._get_embeddings_batch(texts_for_embedding)

        points = []
        for idx, recipe in enumerate(structured):
            recipe_id = recipe['id']
            full_text = text_map.get(recipe_id, '')
            text_for_embedding = texts_for_embedding[idx]
            embedding = embeddings_list[idx]

            payload = {
                "id": recipe_id,
                "name": recipe['name'],
                "subtitle": recipe['subtitle'],
                "category": recipe['category'],
                "preparation": recipe['preparation'],
                "ingredients": recipe['ingredients'],
                "process": recipe['process'],
                "notes": recipe['notes'],
                "full_text": full_text,
                "search_text": text_for_embedding
            }

            sparse_vectors = recipe.get('sparse_vectors', {})

            point = models.PointStruct(
                id=hash(recipe_id) % (2**63),
                vector={"dense": embedding},
                payload=payload,
                sparse_vector=sparse_vectors
            )
            points.append(point)

        logger.info(f"Подготовлено {len(points)} точек данных (размер вектора: {len(embedding)})")
        return points
    
    def upload_data(self, collection_name="test_recipes", batch_size=50):
        """Загрузка данных в Qdrant"""
        points = self.load_recipes_data()
        
        try:
            # Загружаем батчами
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Загружено {min(i+batch_size, len(points))}/{len(points)} точек")
            
            # Создаем payload индексы для быстрого поиска
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="name",
                field_schema=models.PayloadSchemaType.TEXT
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info("Данные успешно загружены и индексированы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def verify_upload(self, collection_name="test_recipes"):
        """Проверка успешной загрузки"""
        try:
            count = self.client.count(collection_name=collection_name).count
            logger.info(f"В коллекции {collection_name} содержится {count} точек")
            
            # Проверяем, что все рецепты загружены
            expected_recipes = ["vodka_potato_tech", "vodka_beetroot_tech", "vodka_topinambur_tech"]
            
            for recipe_id in expected_recipes:
                results = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="id", match=models.MatchValue(value=recipe_id))]
                    ),
                    limit=1
                )
                
                if results[0]:
                    logger.info(f"✓ Рецепт {recipe_id} найден")
                else:
                    logger.warning(f"✗ Рецепт {recipe_id} не найден")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка проверки: {e}")
            return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Настройка Qdrant и загрузка данных")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Пересоздать коллекцию (удалить и создать заново с размером 1536)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Загрузить тестовые данные (требует data/recipes_structured.json)"
    )
    args = parser.parse_args()

    print("Проверка подключения к Qdrant")
    print("=" * 50)

    collection_name = os.getenv('COLLECTION_NAME', 'distill_hybrid')

    try:
        setup = QdrantSetup()
    except ValueError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

    if args.recreate or args.upload:
        # Определяем размерность вектора из модели эмбеддингов
        embedding_type = os.getenv('EMBEDDING_MODEL', 'bgm-m3').lower()
        default_dim = EMBEDDING_DIMS.get(embedding_type, 1024)
        vector_size = int(os.getenv('COLLECTION_VECTOR_SIZE', str(default_dim)))
        if setup.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            recreate=args.recreate
        ) and args.upload:
            setup.upload_data(collection_name=collection_name)
            setup.verify_upload(collection_name=collection_name)
        print("\nДля запуска тестов: python qdrant_test_scripts/test-runner.py")
        print("Для запуска дашборда: streamlit run streamlit_dashboard/test-dashboard.py")
        sys.exit(0)
    
    # Проверяем подключение и коллекцию
    print(f"\n1. Проверка подключения к коллекции '{collection_name}'...")
    try:
        collection_info = setup.client.get_collection(collection_name)
        count = setup.client.count(collection_name=collection_name).count
        
        print(f"Коллекция '{collection_name}' найдена! Содержит {count} точек")
        
        print(f"\nИнформация о коллекции:")
        print(f"   - Векторов: {collection_info.indexed_vectors_count}")
        print(f"   - Точек: {collection_info.points_count}")
        print(f"   - Статус: {collection_info.status}")
        
        vectors_config = collection_info.config.params.vectors
        if vectors_config:
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                dense = vectors_config["dense"]
                print(f"   - Dense вектор: размер={dense.size}, расстояние={dense.distance}")
            elif hasattr(vectors_config, 'size'):
                print(f"   - Размер вектора: {vectors_config.size}")
        
        print("\nПодключение успешно! Коллекция готова к использованию.")
        print("\nДля запуска тестов выполните: python qdrant_test_scripts/test-runner.py")
        print("Для запуска дашборда выполните: streamlit run streamlit_dashboard/test-dashboard.py")
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "forbidden" in error_msg.lower():
            print("Ошибка доступа (403 Forbidden)")
            try:
                test_result = setup.client.scroll(collection_name=collection_name, limit=1)
                print(f"Подключение работает! Можно выполнять поиск в коллекции '{collection_name}'")
                print("\nДля запуска тестов выполните: python qdrant_test_scripts/test-runner.py")
                print("Для запуска дашборда выполните: streamlit run streamlit_dashboard/test-dashboard.py")
            except Exception as e2:
                print(f"Ошибка подключения: {e2}")
                sys.exit(1)
        else:
            print(f"Ошибка подключения: {e}")
            sys.exit(1)
