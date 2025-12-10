"""
Скрипт для настройки Qdrant и загрузки тестовых данных
"""
import json
import time
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantSetup:
    def __init__(self, host=None, port=None, url=None, api_key=None):
        # Читаем параметры из переменных окружения, если не переданы явно
        qdrant_url = url or os.getenv('QDRANT_URL')
        qdrant_host = host or os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = port or int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = api_key or os.getenv('QDRANT_API_KEY')
        
        # Для облачного Qdrant используем URL и API ключ
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
            # Для локального Qdrant используем host и port
            logger.info(f"Подключение к локальному Qdrant: {qdrant_host}:{qdrant_port}")
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-small')
        
    def create_collection(self, collection_name="test_recipes"):
        """Создание коллекции с нужной схемой"""
        
        # Проверяем, существует ли коллекция
        try:
            collections = self.client.get_collections()
            existing = any(c.name == collection_name for c in collections.collections)
            
            if existing:
                logger.info(f"Коллекция {collection_name} уже существует")
                return True
        except Exception as e:
            logger.error(f"Ошибка при проверке коллекций: {e}")
            # Если нет прав на чтение коллекций, пробуем создать напрямую
            if "403" in str(e) or "forbidden" in str(e).lower():
                logger.warning("Нет прав на чтение списка коллекций. Пробуем создать коллекцию...")
            else:
                raise
        
        # Создаем новую коллекцию
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Для multilingual-e5-small
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={
                    "categories": models.SparseVectorParams(),
                    "ingredients": models.SparseVectorParams(),
                    "measurements": models.SparseVectorParams(),
                    "spices_herbs": models.SparseVectorParams(),
                    "techniques": models.SparseVectorParams()
                }
            )
            logger.info(f"Коллекция {collection_name} создана успешно")
            time.sleep(1)  # Даем время на создание
            return True
            
        except Exception as e:
            logger.error(f"Ошибка создания коллекции: {e}")
            return False
    
    def load_recipes_data(self):
        """Загрузка и подготовка данных рецептов"""
        with open('data/recipes_structured.json', 'r', encoding='utf-8') as f:
            structured = json.load(f)['recipes']
        
        with open('data/recipes_full_text.json', 'r', encoding='utf-8') as f:
            full_texts = json.load(f)['texts']
        
        # Создаем словарь полных текстов
        text_map = {text['id']: text['full_text'] for text in full_texts}
        
        points = []
        for recipe in structured:
            recipe_id = recipe['id']
            full_text = text_map.get(recipe_id, '')
            
            # Создаем эмбеддинг из названия и описания
            text_for_embedding = f"{recipe['name']} {recipe['subtitle']} {recipe['preparation']['description']} {full_text[:500]}"
            embedding = self.embedder.encode(text_for_embedding).tolist()
            
            # Подготавливаем payload
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
            
            # Получаем sparse vectors
            sparse_vectors = recipe.get('sparse_vectors', {})
            
            point = models.PointStruct(
                id=hash(recipe_id) % (2**63),  # Генерируем числовой ID
                vector=embedding,
                payload=payload,
                sparse_vector=sparse_vectors
            )
            
            points.append(point)
        
        logger.info(f"Подготовлено {len(points)} точек данных")
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
    import sys
    
    print("🚀 Проверка подключения к Qdrant")
    print("=" * 50)
    
    collection_name = os.getenv('COLLECTION_NAME', 'distill_hybrid')
    
    setup = QdrantSetup()
    
    # Проверяем подключение и коллекцию
    print(f"\n1. Проверка подключения к коллекции '{collection_name}'...")
    try:
        # Пытаемся получить информацию о коллекции напрямую
        collection_info = setup.client.get_collection(collection_name)
        count = setup.client.count(collection_name=collection_name).count
        
        print(f"✅ Коллекция '{collection_name}' найдена! Содержит {count} точек")
        
        # Получаем информацию о коллекции
        print(f"\n📊 Информация о коллекции:")
        print(f"   - Векторов: {collection_info.indexed_vectors_count}")
        print(f"   - Точек: {collection_info.points_count}")
        print(f"   - Статус: {collection_info.status}")
        
        if collection_info.config.params.vectors:
            if hasattr(collection_info.config.params.vectors, 'dense'):
                dense = collection_info.config.params.vectors.dense
                print(f"   - Размер dense вектора: {dense.size}")
                print(f"   - Расстояние: {dense.distance}")
        
        print("\n✅ Подключение успешно! Коллекция готова к использованию.")
        print("\nДля запуска тестов выполните: python qdrant_test_scripts/test-runner.py")
        print("Для запуска дашборда выполните: streamlit run streamlit_dashboard/test-dashboard.py")
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "forbidden" in error_msg.lower():
            print(f"⚠️  Ошибка доступа (403 Forbidden)")
            print(f"   Возможно, API ключ имеет ограниченные права доступа.")
            print(f"   Попробуем проверить подключение другим способом...")
            try:
                # Пробуем выполнить простой поиск для проверки доступа
                test_result = setup.client.scroll(collection_name=collection_name, limit=1)
                print(f"✅ Подключение работает! Можно выполнять поиск в коллекции '{collection_name}'")
                print("\nДля запуска тестов выполните: python qdrant_test_scripts/test-runner.py")
                print("Для запуска дашборда выполните: streamlit run streamlit_dashboard/test-dashboard.py")
            except Exception as e2:
                print(f"❌ Ошибка подключения: {e2}")
                print(f"   Проверьте правильность API ключа и прав доступа.")
                sys.exit(1)
        else:
            print(f"❌ Ошибка подключения: {e}")
            sys.exit(1)