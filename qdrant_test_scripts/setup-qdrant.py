"""
Скрипт для настройки Qdrant и загрузки тестовых данных
"""
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantSetup:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-small')
        
    def create_collection(self, collection_name="test_recipes"):
        """Создание коллекции с нужной схемой"""
        
        # Проверяем, существует ли коллекция
        collections = self.client.get_collections()
        existing = any(c.name == collection_name for c in collections.collections)
        
        if existing:
            logger.info(f"Коллекция {collection_name} уже существует")
            return True
        
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
    print("🚀 Настройка тестовой среды Qdrant")
    print("=" * 50)
    
    setup = QdrantSetup()
    
    # 1. Создаем коллекцию
    print("\n1. Создание коллекции...")
    if not setup.create_collection():
        print("❌ Не удалось создать коллекцию")
        exit(1)
    
    # 2. Загружаем данные
    print("\n2. Загрузка данных...")
    if not setup.upload_data():
        print("❌ Не удалось загрузить данные")
        exit(1)
    
    # 3. Проверяем загрузку
    print("\n3. Проверка загрузки...")
    if setup.verify_upload():
        print("✅ Настройка завершена успешно!")
    else:
        print("⚠️ Настройка завершена с предупреждениями")
    
    print("\nДля запуска тестов выполните: python qdrant_test_scripts/test_runner.py")
    print("Для запуска дашборда выполните: streamlit run streamlit_dashboard/test_dashboard.py")