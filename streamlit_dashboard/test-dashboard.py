"""
Streamlit дашборд для интерактивного тестирования поиска в Qdrant
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Универсальный клиент эмбеддингов (OpenAI или bgm-m3, размерность по модели)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_client import get_embedding_client

# Настройки страницы
st.set_page_config(
    page_title="Qdrant Search Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация клиентов
@st.cache_resource
def init_qdrant_client():
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if qdrant_url:
        if qdrant_api_key:
            return QdrantClient(
                url=qdrant_url, 
                api_key=qdrant_api_key,
                check_compatibility=False
            )
        else:
            return QdrantClient(url=qdrant_url, check_compatibility=False)
    else:
        return QdrantClient(host=qdrant_host, port=qdrant_port)

@st.cache_resource
def init_embedder():
    """Клиент эмбеддингов: OpenAI (1536) или bgm-m3 (1024) по EMBEDDING_MODEL"""
    try:
        return get_embedding_client()
    except Exception:
        return None


def get_query_embedding(embedder, text: str):
    if embedder is None:
        return None
    return embedder.get_embedding(text)

# Загрузка данных
@st.cache_data
def load_recipes_data():
    with open('data/recipes_structured.json', 'r', encoding='utf-8') as f:
        recipes = json.load(f)['recipes']
    
    # Создаем DataFrame для отображения
    df_data = []
    for recipe in recipes:
        df_data.append({
            "ID": recipe['id'],
            "Название": recipe['name'],
            "Описание": recipe['preparation']['description'][:100] + "...",
            "Ингредиенты": len(recipe['ingredients']),
            "Шаги": len(recipe['process']),
            "Категория": recipe['category']
        })
    
    return recipes, pd.DataFrame(df_data)

# Инициализация
client = init_qdrant_client()
embedder = init_embedder()
recipes, recipes_df = load_recipes_data()

# Заголовок
st.title("🔍 Qdrant Search Test Dashboard")
st.markdown("---")

# Сайдбар с настройками
with st.sidebar:
    st.header("⚙️ Настройки поиска")
    
    # Получаем список коллекций
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        default_collection = os.getenv('COLLECTION_NAME', 'distill_hybrid')
        default_index = collection_names.index(default_collection) if default_collection in collection_names else 0
    except:
        collection_names = ["distill_hybrid"]
        default_index = 0
    
    collection_name = st.selectbox(
        "Коллекция:",
        collection_names,
        index=default_index
    )
    
    search_type = st.radio(
        "Тип поиска:",
        ["Векторный (семантический)", "Гибридный", "По ключевым словам"],
        index=0
    )
    
    limit_results = st.slider(
        "Количество результатов:",
        min_value=1,
        max_value=20,
        value=5
    )
    
    score_threshold = st.slider(
        "Порог релевантности:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    
    if st.button("🔄 Проверить подключение"):
        try:
            count = client.count(collection_name=collection_name).count
            st.success(f"✅ Подключено! В коллекции {count} записей")
        except Exception as e:
            st.error(f"❌ Ошибка подключения: {e}")

# Основное содержимое
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Поиск", "📊 Аналитика", "🧪 Тесты", "📚 Данные"])

with tab1:
    st.header("Интерактивный поиск")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_area(
            "Поисковый запрос:",
            "технология производства водки из картофеля",
            height=100,
            help="Введите запрос для семантического поиска"
        )
    
    with col2:
        st.markdown("### Дополнительно")
        show_details = st.checkbox("Показать детали", value=True)
        show_embeddings = st.checkbox("Показать эмбеддинги", value=False)
    
    if st.button("🔎 Выполнить поиск", type="primary", use_container_width=True):
        with st.spinner("Выполняю поиск..."):
            try:
                if embedder is None:
                    st.error("Модель эмбеддингов не инициализирована. Проверьте EMBEDDING_MODEL и настройки в .env")
                else:
                    query_embedding = get_query_embedding(embedder, search_query)
                if query_embedding is None:
                    st.error("Не удалось получить эмбеддинг")
                else:
                    start_time = time.time()
                    results = client.search(
                        collection_name=collection_name,
                        query_vector=models.NamedVector(name="dense", vector=query_embedding),
                        limit=limit_results,
                        score_threshold=score_threshold,
                        with_payload=True,
                        with_vectors=show_embeddings
                    )
                    search_time = time.time() - start_time
                
                    # Отображаем результаты
                    st.subheader(f"Результаты поиска ({len(results)} найдено, время: {search_time:.2f}с)")
                
                    if not results:
                        st.warning("Ничего не найдено. Попробуйте изменить запрос или снизить порог релевантности.")
                    else:
                        # Таблица с результатами
                        result_data = []
                        for i, hit in enumerate(results, 1):
                            result_data.append({
                                "№": i,
                                "Название": hit.payload.get('name', 'N/A'),
                                "ID": hit.payload.get('id', 'N/A'),
                                "Score": f"{hit.score:.3f}",
                                "Категория": hit.payload.get('category', 'N/A'),
                                "Ингредиентов": len(hit.payload.get('ingredients', []))
                            })
                        
                        result_df = pd.DataFrame(result_data)
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Детали для каждого результата
                        if show_details:
                            for i, hit in enumerate(results, 1):
                                with st.expander(f"#{i}: {hit.payload.get('name')} (score: {hit.score:.3f})"):
                                    col_a, col_b = st.columns(2)
                                    
                                    with col_a:
                                        st.markdown("**Основная информация:**")
                                        st.write(f"**ID:** `{hit.payload.get('id')}`")
                                        st.write(f"**Категория:** {hit.payload.get('category')}")
                                        st.write(f"**Описание:** {hit.payload.get('preparation', {}).get('description', 'N/A')}")
                                    
                                    with col_b:
                                        st.markdown("**Статистика:**")
                                        st.write(f"**Ингредиентов:** {len(hit.payload.get('ingredients', []))}")
                                        st.write(f"**Шагов процесса:** {len(hit.payload.get('process', []))}")
                                        st.write(f"**Примечаний:** {len(hit.payload.get('notes', []))}")
                                    
                                    # Ингредиенты
                                    if hit.payload.get('ingredients'):
                                        st.markdown("**Ингредиенты:**")
                                        ingredients_text = ", ".join([
                                            f"{ing.get('name')} ({ing.get('amount', '?')} {ing.get('unit', '')})"
                                            for ing in hit.payload.get('ingredients', [])
                                        ])
                                        st.write(ingredients_text[:200] + "...")
                                    
                                    # Ключевые слова из sparse vectors
                                    if hasattr(hit, 'sparse_vector') and hit.sparse_vector:
                                        st.markdown("**Ключевые слова:**")
                                        for category, terms in hit.sparse_vector.items():
                                            if terms:
                                                top_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:5]
                                                terms_text = ", ".join([f"{term}" for term, _ in top_terms])
                                                st.write(f"*{category}:* {terms_text}")
                
            except Exception as e:
                st.error(f"Ошибка при поиске: {e}")

with tab2:
    st.header("Аналитика поиска")
    
    # Статистика коллекции
    try:
        count = client.count(collection_name=collection_name).count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего записей", count)
        with col2:
            st.metric("Коллекция", collection_name)
        with col3:
            st.metric("Тип поиска", search_type)
        
        # Визуализация категорий
        st.subheader("Распределение по категориям")
        
        # Получаем все записи для анализа
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True
        )
        
        if scroll_result[0]:
            # Анализируем категории
            categories = {}
            for point in scroll_result[0]:
                category = point.payload.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            if categories:
                # Создаем график
                cat_df = pd.DataFrame({
                    'Категория': list(categories.keys()),
                    'Количество': list(categories.values())
                })
                
                fig = px.pie(cat_df, values='Количество', names='Категория',
                           title='Распределение по категориям',
                           hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Гистограмма по количеству ингредиентов
                st.subheader("Количество ингредиентов в рецептах")
                
                ingredient_counts = []
                for point in scroll_result[0]:
                    count = len(point.payload.get('ingredients', []))
                    ingredient_counts.append(count)
                
                if ingredient_counts:
                    fig2 = px.histogram(x=ingredient_counts, 
                                      nbins=10,
                                      title='Распределение по количеству ингредиентов',
                                      labels={'x': 'Количество ингредиентов', 'y': 'Количество рецептов'})
                    st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Ошибка при аналитике: {e}")

with tab3:
    st.header("Автоматические тесты")
    
    if st.button("🚀 Запустить все тесты", type="primary"):
        with st.spinner("Выполняю тесты..."):
            # Импортируем и запускаем тестер
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../qdrant_test_scripts'))
            
            try:
                from test_runner import QdrantTester
                tester = QdrantTester()
                results = tester.run_all_tests()
                
                # Отображаем результаты
                st.success("Тесты завершены!")
                
                # Сводка
                summary = results['summary']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Всего тестов", summary['total_tests'])
                with col2:
                    st.metric("Успешно", summary['total_passed'])
                with col3:
                    st.metric("С предупреждениями", summary['total_warning'])
                with col4:
                    st.metric("Неудачно", summary['total_failed'])
                
                # Детальные результаты
                for recipe_result in results['detailed_results']:
                    with st.expander(f"{recipe_result['recipe_name']} - {recipe_result['summary']['success_rate']}"):
                        for query_result in recipe_result['results']:
                            status_icon = "✅" if query_result['status'] == 'PASSED' else "⚠️" if query_result['status'] == 'WARNING' else "❌"
                            st.write(f"{status_icon} **{query_result['query']}**")
                            st.write(f"   {query_result['message']}")
                            
            except Exception as e:
                st.error(f"Ошибка при выполнении тестов: {e}")

with tab4:
    st.header("Просмотр данных")
    
    # Показываем все рецепты
    st.subheader("Все рецепты в базе")
    st.dataframe(recipes_df, use_container_width=True)
    
    # Выбор рецепта для детального просмотра
    selected_recipe_id = st.selectbox(
        "Выберите рецепт для детального просмотра:",
        recipes_df['ID'].tolist()
    )
    
    if selected_recipe_id:
        recipe = next(r for r in recipes if r['id'] == selected_recipe_id)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(recipe['name'])
            st.write(f"**Категория:** {recipe['category']}")
            st.write(f"**Описание:** {recipe['preparation']['description']}")
            
            # Ингредиенты
            st.markdown("**Ингредиенты:**")
            for ing in recipe['ingredients']:
                st.write(f"- {ing['name']}: {ing.get('amount', '?')} {ing.get('unit', '')} {ing.get('notes', '')}")
            
            # Процесс
            st.markdown("**Процесс приготовления:**")
            for step in recipe['process']:
                st.write(f"{step['step']}. **{step['action']}**: {step['description']}")
        
        with col2:
            # Sparse vectors
            if 'sparse_vectors' in recipe:
                st.markdown("**Ключевые слова для поиска:**")
                for category, vectors in recipe['sparse_vectors'].items():
                    with st.expander(f"{category}"):
                        top_terms = sorted(vectors.items(), key=lambda x: x[1], reverse=True)[:10]
                        for term, weight in top_terms:
                            st.progress(weight, text=f"{term}: {weight:.2f}")
            
            # Статистика
            st.markdown("**Статистика:**")
            st.write(f"Ингредиентов: {len(recipe['ingredients'])}")
            st.write(f"Шагов процесса: {len(recipe['process'])}")
            st.write(f"Примечаний: {len(recipe['notes'])}")

# Футер
st.markdown("---")
st.caption("🔍 Qdrant Search Test Dashboard • Тестовая среда для проверки поиска")