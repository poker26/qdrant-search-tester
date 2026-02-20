"""
Streamlit дашборд для интерактивного тестирования поиска в Qdrant
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
import json
import time
import os
from dotenv import load_dotenv
import httpx
import numpy as np

load_dotenv()

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
    qdrant_url = os.getenv('QDRANT_URL', '').strip()
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost').strip()
    qdrant_port_str = os.getenv('QDRANT_PORT', '6333').strip()
    qdrant_port = int(qdrant_port_str) if qdrant_port_str else 6333
    qdrant_api_key = os.getenv('QDRANT_API_KEY', '').strip()
    
    if qdrant_url:
        # Если URL содержит порт, извлекаем его отдельно
        if ':' in qdrant_url and qdrant_url.count(':') > 1:  # Есть порт в URL (https://host:port)
            from urllib.parse import urlparse
            parsed = urlparse(qdrant_url.strip())
            base_url = f"{parsed.scheme}://{parsed.hostname}"
            port = parsed.port if parsed.port else 443
            
            if qdrant_api_key:
                return QdrantClient(
                    url=base_url,
                    port=port,
                    api_key=qdrant_api_key,
                    https=True,
                    check_compatibility=False
                )
            else:
                return QdrantClient(
                    url=base_url,
                    port=port,
                    https=True,
                    check_compatibility=False
                )
        else:
            # URL без порта
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

# Импортируем универсальный клиент эмбеддингов
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_client import get_embedding_client, EMBEDDING_DIMS


@st.cache_resource
def init_embedder():
    """Инициализация клиента для эмбеддингов (OpenAI или bgm-m3)"""
    try:
        client = get_embedding_client()
        model_name = client.get_model_name()
        dim = client.get_embedding_dim()
        return client
    except Exception as e:
        st.error(f"❌ Ошибка инициализации модели эмбеддингов: {e}")
        return None


def get_query_embedding(embedder, text: str):
    """Получение эмбеддинга через универсальный клиент"""
    if embedder is None:
        return None
    try:
        return embedder.get_embedding(text)
    except Exception as e:
        st.error(f"Ошибка получения эмбеддинга: {e}")
        return None

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

# Инициализация (ленивая - выполняется только при первом обращении)
@st.cache_resource
def get_client():
    return init_qdrant_client()

@st.cache_resource  
def get_embedder():
    return init_embedder()

@st.cache_data
def get_recipes_data():
    return load_recipes_data()

# Инициализация при первом использовании
client = get_client()
embedder = get_embedder()
# recipes, recipes_df = get_recipes_data()  # Временно отключено, раздел "Данные" скрыт

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
tab1, tab2, tab3 = st.tabs(["🔍 Поиск", "📊 Аналитика", "🧪 Тесты"])

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
                # Создаем эмбеддинг через OpenAI
                query_embedding = get_query_embedding(embedder, search_query)
                if query_embedding is None:
                    st.error("Не удалось создать эмбеддинг. Проверьте OPENAI_API_KEY.")
                else:
                    # Выполняем поиск
                    # Сначала пробуем с using="dense", при ошибке — без using (default vector)
                    start_time = time.time()
                    try:
                        query_response = client.query_points(
                            collection_name=collection_name,
                            query=query_embedding,
                            using="dense",
                            limit=limit_results,
                            score_threshold=score_threshold,
                            with_payload=True,
                            with_vectors=show_embeddings
                        )
                    except Exception as vec_err:
                        err_msg = str(vec_err).lower()
                        if "dense" in err_msg and ("not existing" in err_msg or "vector name" in err_msg):
                            # Коллекция с default-вектором (без имени)
                            query_response = client.query_points(
                                collection_name=collection_name,
                                query=query_embedding,
                                limit=limit_results,
                                score_threshold=score_threshold,
                                with_payload=True,
                                with_vectors=show_embeddings
                            )
                        else:
                            raise
                    results = query_response.points
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
    
    # Подразделы для управления тестами
    test_tab1, test_tab2 = st.tabs(["📋 Управление тестами", "▶️ Запуск тестов"])
    
    with test_tab1:
        st.subheader("Создание и редактирование тестов")
        
        # Импортируем менеджер тестов
        import sys
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(base_dir, '..'))
        
        try:
            from test_manager import TestManager, TestCase
            from datetime import datetime
            
            tests_file = os.path.join(base_dir, '..', 'tests.json')
            test_manager = TestManager(tests_file=tests_file)
            
            # Форма создания нового теста
            with st.expander("➕ Создать новый тест", expanded=False):
                with st.form("new_test_form"):
                    test_name = st.text_input("Название теста*", placeholder="Например: Поиск рецепта водки из картофеля")
                    test_query = st.text_area("Поисковый запрос*", placeholder="Введите запрос для тестирования поиска", height=100)
                    col1, col2 = st.columns(2)
                    with col1:
                        expected_id = st.text_input("Ожидаемый ID результата", placeholder="vodka_potato_tech")
                        max_rank = st.number_input("Максимальная позиция", min_value=1, max_value=20, value=3)
                    with col2:
                        expected_ids_str = st.text_input("Или список ID (через запятую)", placeholder="id1, id2, id3")
                        min_score = st.number_input("Минимальный score", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                    description = st.text_area("Описание (опционально)", placeholder="Дополнительная информация о тесте")
                    
                    submitted = st.form_submit_button("💾 Сохранить тест", type="primary")
                    
                    if submitted:
                        if not test_name or not test_query:
                            st.error("Пожалуйста, заполните название и запрос")
                        else:
                            expected_ids = None
                            if expected_ids_str:
                                expected_ids = [id.strip() for id in expected_ids_str.split(',') if id.strip()]
                            
                            new_test = TestCase(
                                id="",
                                name=test_name,
                                query=test_query,
                                expected_result_id=expected_id if expected_id else None,
                                expected_result_ids=expected_ids if expected_ids else None,
                                max_rank=max_rank,
                                min_score=min_score,
                                description=description
                            )
                            
                            if test_manager.add_test(new_test):
                                st.success(f"✅ Тест '{test_name}' успешно создан!")
                                st.rerun()
                            else:
                                st.error("Ошибка: тест с таким ID уже существует")
            
            # Список существующих тестов
            st.subheader("📝 Существующие тесты")
            tests = test_manager.get_all_tests()
            
            if not tests:
                st.info("Пока нет созданных тестов. Создайте первый тест выше.")
            else:
                for i, test in enumerate(tests):
                    with st.expander(f"🔍 {test.name} (ID: {test.id})", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Запрос:** {test.query}")
                            if test.expected_result_id:
                                st.write(f"**Ожидаемый ID:** `{test.expected_result_id}`")
                            if test.expected_result_ids:
                                st.write(f"**Ожидаемые ID:** {', '.join(test.expected_result_ids)}")
                            st.write(f"**Макс. позиция:** {test.max_rank}, **Мин. score:** {test.min_score}")
                            if test.description:
                                st.write(f"**Описание:** {test.description}")
                            if test.created_at:
                                st.caption(f"Создан: {test.created_at}")
                        with col2:
                            if st.button("🗑️ Удалить", key=f"delete_{test.id}"):
                                if test_manager.delete_test(test.id):
                                    st.success("Тест удален")
                                    st.rerun()
                                else:
                                    st.error("Ошибка при удалении")
        
        except ImportError as e:
            st.error(f"Не удалось импортировать test_manager: {e}")
            st.info("Убедитесь, что файл test_manager.py находится в корне проекта")
        except Exception as e:
            st.error(f"Ошибка при работе с тестами: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    with test_tab2:
        st.subheader("Запуск тестов")
        
        # Импортируем менеджер тестов для выбора
        import sys
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(base_dir, '..'))
        
        try:
            from test_manager import TestManager
            
            tests_file = os.path.join(base_dir, '..', 'tests.json')
            test_manager = TestManager(tests_file=tests_file)
            all_tests = test_manager.get_all_tests()
            
            if not all_tests:
                st.warning("⚠️ Нет созданных тестов. Перейдите на вкладку 'Управление тестами' для создания тестов.")
            else:
                # Выбор тестов для запуска
                test_options = {f"{t.name} ({t.id})": t.id for t in all_tests}
                selected_test_names = st.multiselect(
                    "Выберите тесты для запуска (оставьте пустым для запуска всех):",
                    options=list(test_options.keys()),
                    default=[]
                )
                
                selected_test_ids = [test_options[name] for name in selected_test_names] if selected_test_names else None
                
                col1, col2 = st.columns(2)
                with col1:
                    run_all = st.button("🚀 Запустить все тесты", type="primary", use_container_width=True)
                with col2:
                    run_selected = st.button("▶️ Запустить выбранные", type="secondary", use_container_width=True, disabled=not selected_test_ids)
                
                if run_all or run_selected:
                    with st.spinner("Выполняю тесты..."):
                        # Импортируем новый тестер
                        import importlib.util
                        runner_path = None
                        for rel in ['../qdrant_test_scripts', '../qdrant-search-tester/qdrant_test_scripts']:
                            candidate = os.path.normpath(os.path.join(base_dir, rel, 'test-runner-v2.py'))
                            if os.path.isfile(candidate):
                                runner_path = candidate
                                break
                        
                        if not runner_path:
                            runner_path = os.path.normpath(os.path.join(base_dir, '..', 'qdrant_test_scripts', 'test-runner-v2.py'))
                        
                        try:
                            spec = importlib.util.spec_from_file_location("test_runner_v2", runner_path)
                            test_runner_v2 = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(test_runner_v2)
                            QdrantTesterV2 = test_runner_v2.QdrantTesterV2
                            
                            tester = QdrantTesterV2(tests_file=tests_file)
                            test_ids_to_run = selected_test_ids if run_selected else None
                            results = tester.run_tests(test_ids=test_ids_to_run)
                            
                            # Отображаем результаты
                            st.success("✅ Тесты завершены!")
                            
                            # Сводка
                            summary = results['summary']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Всего тестов", summary['total_tests'])
                            with col2:
                                st.metric("Успешно", summary['total_passed'], delta=f"{results['success_rate']:.1f}%")
                            with col3:
                                st.metric("С предупреждениями", summary['total_warning'])
                            with col4:
                                st.metric("Неудачно", summary['total_failed'])
                            
                            # Детальные результаты
                            st.subheader("📊 Детальные результаты")
                            for result in results['detailed_results']:
                                status_icon = "✅" if result['status'] == 'PASSED' else "⚠️" if result['status'] == 'WARNING' else "❌"
                                with st.expander(f"{status_icon} {result['test_name']} - {result['status']}", expanded=False):
                                    st.write(f"**Запрос:** {result['query']}")
                                    st.write(f"**Результат:** {result['message']}")
                                    st.write(f"**Позиция:** {result['rank']}, **Score:** {result['score']}")
                                    if result['found_id'] != 'N/A':
                                        st.write(f"**Найденный ID:** `{result['found_id']}`")
                                    if result['expected_ids']:
                                        st.write(f"**Ожидались ID:** {', '.join(result['expected_ids'])}")
                                    
                                    # Топ-5 результатов
                                    if result['top_results']:
                                        st.write("**Топ-5 результатов поиска:**")
                                        top_df = pd.DataFrame(result['top_results'])
                                        st.dataframe(top_df, use_container_width=True, hide_index=True)
                                    
                        except FileNotFoundError:
                            st.error(f"❌ Файл test-runner-v2.py не найден по пути: {runner_path}")
                            st.info("Убедитесь, что файл test-runner-v2.py находится в директории qdrant_test_scripts")
                        except Exception as e:
                            st.error(f"Ошибка при выполнении тестов: {e}")
                            import traceback
                            with st.expander("Детали ошибки"):
                                st.code(traceback.format_exc())
        
        except ImportError as e:
            st.error(f"Не удалось импортировать test_manager: {e}")
            st.info("Убедитесь, что файл test_manager.py находится в корне проекта")
        except Exception as e:
            st.error(f"Ошибка при работе с тестами: {e}")
            import traceback
            st.code(traceback.format_exc())

# Раздел "Данные" временно скрыт
# with tab4:
#     st.header("Просмотр данных")
#     
#     # Показываем все рецепты
#     st.subheader("Все рецепты в базе")
#     st.dataframe(recipes_df, use_container_width=True)
#     
#     # Выбор рецепта для детального просмотра
#     selected_recipe_id = st.selectbox(
#         "Выберите рецепт для детального просмотра:",
#         recipes_df['ID'].tolist()
#     )
#     
#     if selected_recipe_id:
#         recipe = next(r for r in recipes if r['id'] == selected_recipe_id)
#         
#         col1, col2 = st.columns([2, 1])
#         
#         with col1:
#             st.subheader(recipe['name'])
#             st.write(f"**Категория:** {recipe['category']}")
#             st.write(f"**Описание:** {recipe['preparation']['description']}")
#             
#             # Ингредиенты
#             st.markdown("**Ингредиенты:**")
#             for ing in recipe['ingredients']:
#                 st.write(f"- {ing['name']}: {ing.get('amount', '?')} {ing.get('unit', '')} {ing.get('notes', '')}")
#             
#             # Процесс
#             st.markdown("**Процесс приготовления:**")
#             for step in recipe['process']:
#                 st.write(f"{step['step']}. **{step['action']}**: {step['description']}")
#         
#         with col2:
#             # Sparse vectors
#             if 'sparse_vectors' in recipe:
#                 st.markdown("**Ключевые слова для поиска:**")
#                 for category, vectors in recipe['sparse_vectors'].items():
#                     with st.expander(f"{category}"):
#                         top_terms = sorted(vectors.items(), key=lambda x: x[1], reverse=True)[:10]
#                         for term, weight in top_terms:
#                             st.progress(weight, text=f"{term}: {weight:.2f}")
#             
#             # Статистика
#             st.markdown("**Статистика:**")
#             st.write(f"Ингредиентов: {len(recipe['ingredients'])}")
#             st.write(f"Шагов процесса: {len(recipe['process'])}")
#             st.write(f"Примечаний: {len(recipe['notes'])}")

# Футер
st.markdown("---")
st.caption("🔍 Qdrant Search Test Dashboard • Тестовая среда для проверки поиска")