"""
Streamlit дашборд для тестирования поиска в Qdrant.
Поддерживает: dense, sparse, hybrid (RRF), сравнение коллекций.
"""
import streamlit as st
import pandas as pd
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from embedding_client import get_embedding_client, EmbeddingResult

from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Page config ---
st.set_page_config(
    page_title="Qdrant Search Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Init ---

@st.cache_resource
def init_qdrant():
    url = os.getenv('QDRANT_URL', '').strip()
    host = os.getenv('QDRANT_HOST', 'localhost').strip()
    port = int(os.getenv('QDRANT_PORT', '6333'))
    api_key = os.getenv('QDRANT_API_KEY', '').strip()
    if url:
        kwargs = {"url": url, "check_compatibility": False}
        if api_key:
            kwargs["api_key"] = api_key
        return QdrantClient(**kwargs)
    return QdrantClient(host=host, port=port)

@st.cache_resource
def init_embedder():
    try:
        return get_embedding_client()
    except Exception as e:
        st.error(f"❌ Ошибка модели эмбеддингов: {e}")
        return None

client = init_qdrant()
embedder = init_embedder()


def get_collections():
    try:
        cols = client.get_collections()
        return [c.name for c in cols.collections]
    except Exception:
        return ["distill_hybrid", "distill_hybrid_v2"]


def do_search(collection: str, emb: EmbeddingResult, mode: str, limit: int, score_threshold: float):
    """
    Выполняет поиск в Qdrant.
    mode: 'dense', 'sparse', 'hybrid'
    """
    start = time.time()

    if mode == "dense":
        resp = client.query_points(
            collection_name=collection,
            query=emb.dense,
            using="dense",
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results = resp.points

    elif mode == "sparse" and emb.sparse:
        resp = client.query_points(
            collection_name=collection,
            query=models.SparseVector(
                indices=emb.sparse["indices"],
                values=emb.sparse["values"]
            ),
            using="sparse",
            limit=limit,
            with_payload=True,
        )
        results = resp.points

    elif mode == "hybrid" and emb.sparse:
        # RRF fusion: prefetch dense + sparse, fuse
        resp = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=emb.dense,
                    using="dense",
                    limit=limit * 3,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=emb.sparse["indices"],
                        values=emb.sparse["values"]
                    ),
                    using="sparse",
                    limit=limit * 3,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        results = resp.points
    else:
        # fallback to dense
        resp = client.query_points(
            collection_name=collection,
            query=emb.dense,
            using="dense",
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results = resp.points

    elapsed = time.time() - start
    return results, elapsed


def results_to_df(results):
    rows = []
    for i, hit in enumerate(results, 1):
        p = hit.payload
        rows.append({
            "№": i,
            "Score": f"{hit.score:.4f}",
            "Название": p.get("recipe_name", p.get("name", "N/A")),
            "ID": p.get("recipe_id", p.get("id", "N/A")),
            "Категория": p.get("category", ""),
            "Длина": p.get("content_length", ""),
        })
    return pd.DataFrame(rows)


def show_result_details(results):
    for i, hit in enumerate(results, 1):
        p = hit.payload
        name = p.get("recipe_name", p.get("name", "N/A"))
        with st.expander(f"#{i} {name} (score: {hit.score:.4f})"):
            st.write(f"**ID:** `{p.get('recipe_id', p.get('id'))}`")
            if p.get("category"):
                st.write(f"**Категория:** {p['category']}")
            content = p.get("content", p.get("full_text", ""))
            if content:
                st.text_area("Содержание", content[:2000], height=200, disabled=True, key=f"content_{i}_{id(hit)}")
            if p.get("sparse_token_count"):
                st.caption(f"Sparse tokens: {p['sparse_token_count']}, Dense dim: {p.get('vector_dimension', '')}")


# --- Sidebar ---
st.title("🔍 Qdrant Search Tester")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Настройки")

    collection_names = get_collections()
    default_col = os.getenv('COLLECTION_NAME', 'distill_hybrid_v2')
    default_idx = collection_names.index(default_col) if default_col in collection_names else 0

    collection = st.selectbox("Коллекция:", collection_names, index=default_idx)

    search_mode = st.radio(
        "Тип поиска:",
        ["hybrid", "dense", "sparse"],
        format_func=lambda x: {"hybrid": "🔀 Гибридный (RRF)", "dense": "🧠 Dense (семантика)", "sparse": "📝 Sparse (лексика)"}[x]
    )

    limit = st.slider("Результатов:", 1, 20, 5)
    score_threshold = st.slider("Порог score:", 0.0, 1.0, 0.0, 0.05)

    if st.button("🔄 Проверить подключение"):
        try:
            cnt = client.count(collection_name=collection).count
            model = embedder.get_model_name() if embedder else "N/A"
            sparse_ok = "✅" if embedder and embedder.supports_sparse() else "❌"
            st.success(f"✅ {collection}: {cnt} записей\nМодель: {model}\nSparse: {sparse_ok}")
        except Exception as e:
            st.error(f"❌ {e}")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Поиск", "⚖️ Сравнение коллекций", "📊 Сравнение режимов", "🧪 Тесты"])

# === TAB 1: Поиск ===
with tab1:
    st.header("Интерактивный поиск")

    query_text = st.text_area(
        "Запрос:",
        "рецепт водки с анисом и корицей",
        height=80
    )
    show_details = st.checkbox("Показать детали", value=True, key="t1_details")

    if st.button("🔎 Искать", type="primary", use_container_width=True, key="t1_search"):
        if not embedder:
            st.error("Embedder не инициализирован")
        else:
            with st.spinner("Поиск..."):
                emb = embedder.get_embedding_full(query_text)
                results, elapsed = do_search(collection, emb, search_mode, limit, score_threshold)

                st.subheader(f"Результаты: {len(results)} ({elapsed:.2f}с, режим: {search_mode})")
                if not results:
                    st.warning("Ничего не найдено")
                else:
                    st.dataframe(results_to_df(results), use_container_width=True, hide_index=True)
                    if show_details:
                        show_result_details(results)

# === TAB 2: Сравнение коллекций ===
with tab2:
    st.header("Сравнение коллекций")
    st.caption("Один запрос — результаты из двух коллекций рядом")

    col_a, col_b = st.columns(2)
    with col_a:
        coll_1 = st.selectbox("Коллекция 1:", collection_names, index=0, key="cmp_c1")
    with col_b:
        idx2 = min(1, len(collection_names) - 1)
        coll_2 = st.selectbox("Коллекция 2:", collection_names, index=idx2, key="cmp_c2")

    cmp_query = st.text_area("Запрос:", "водка с померанцевой коркой", height=80, key="cmp_query")
    cmp_mode = st.radio("Режим:", ["hybrid", "dense", "sparse"], horizontal=True, key="cmp_mode",
                        format_func=lambda x: {"hybrid": "Гибридный", "dense": "Dense", "sparse": "Sparse"}[x])

    if st.button("⚖️ Сравнить", type="primary", use_container_width=True, key="cmp_go"):
        if not embedder:
            st.error("Embedder не инициализирован")
        else:
            with st.spinner("Поиск..."):
                emb = embedder.get_embedding_full(cmp_query)

                r1, t1 = do_search(coll_1, emb, cmp_mode, limit, score_threshold)
                r2, t2 = do_search(coll_2, emb, cmp_mode, limit, score_threshold)

                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(f"{coll_1} ({t1:.2f}с)")
                    if r1:
                        st.dataframe(results_to_df(r1), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Пусто")
                with c2:
                    st.subheader(f"{coll_2} ({t2:.2f}с)")
                    if r2:
                        st.dataframe(results_to_df(r2), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Пусто")

# === TAB 3: Сравнение режимов ===
with tab3:
    st.header("Сравнение режимов поиска")
    st.caption("Один запрос — dense vs sparse vs hybrid")

    modes_coll = st.selectbox("Коллекция:", collection_names, key="modes_coll")
    modes_query = st.text_area("Запрос:", "перегонка через куб с травами", height=80, key="modes_query")

    if st.button("📊 Сравнить режимы", type="primary", use_container_width=True, key="modes_go"):
        if not embedder:
            st.error("Embedder не инициализирован")
        else:
            with st.spinner("Поиск..."):
                emb = embedder.get_embedding_full(modes_query)

                rd, td = do_search(modes_coll, emb, "dense", limit, score_threshold)
                rs, ts = do_search(modes_coll, emb, "sparse", limit, 0.0) if emb.sparse else ([], 0)
                rh, th = do_search(modes_coll, emb, "hybrid", limit, 0.0) if emb.sparse else ([], 0)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.subheader(f"Dense ({td:.2f}с)")
                    st.dataframe(results_to_df(rd), use_container_width=True, hide_index=True) if rd else st.warning("Пусто")
                with c2:
                    st.subheader(f"Sparse ({ts:.2f}с)")
                    st.dataframe(results_to_df(rs), use_container_width=True, hide_index=True) if rs else st.warning("Пусто / нет sparse")
                with c3:
                    st.subheader(f"Hybrid RRF ({th:.2f}с)")
                    st.dataframe(results_to_df(rh), use_container_width=True, hide_index=True) if rh else st.warning("Пусто / нет sparse")

# === TAB 4: Тесты ===
with tab4:
    st.header("Автоматические тесты")

    test_tab1, test_tab2 = st.tabs(["📋 Управление тестами", "▶️ Запуск"])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests_file = os.path.join(base_dir, '..', 'tests.json')

    try:
        from test_manager import TestManager, TestCase
        from datetime import datetime

        test_manager = TestManager(tests_file=tests_file)
    except ImportError as e:
        st.error(f"test_manager не найден: {e}")
        test_manager = None

    with test_tab1:
        if test_manager:
            with st.expander("➕ Создать тест", expanded=False):
                with st.form("new_test"):
                    t_name = st.text_input("Название*")
                    t_query = st.text_area("Запрос*", height=80)
                    c1, c2 = st.columns(2)
                    with c1:
                        t_id = st.text_input("Ожидаемый ID")
                        t_rank = st.number_input("Макс. позиция", 1, 20, 3)
                    with c2:
                        t_ids = st.text_input("Или список ID (через запятую)")
                        t_score = st.number_input("Мин. score", 0.0, 1.0, 0.3, 0.05)
                    t_mode = st.selectbox("Режим поиска", ["hybrid", "dense", "sparse"])
                    t_collection = st.text_input("Коллекция (пусто = из сайдбара)")
                    t_desc = st.text_area("Описание")

                    if st.form_submit_button("💾 Сохранить"):
                        if not t_name or not t_query:
                            st.error("Заполните название и запрос")
                        else:
                            ids_list = [x.strip() for x in t_ids.split(',') if x.strip()] if t_ids else None
                            new_test = TestCase(
                                id="", name=t_name, query=t_query,
                                expected_result_id=t_id or None,
                                expected_result_ids=ids_list,
                                max_rank=t_rank, min_score=t_score,
                                search_mode=t_mode,
                                collection=t_collection or None,
                                description=t_desc
                            )
                            if test_manager.add_test(new_test):
                                st.success(f"✅ Тест '{t_name}' создан")
                                st.rerun()

            tests = test_manager.get_all_tests()
            if not tests:
                st.info("Нет тестов. Создайте первый.")
            else:
                st.subheader(f"📝 Тесты ({len(tests)})")
                for t in tests:
                    with st.expander(f"🔍 {t.name} [{t.search_mode}]"):
                        st.write(f"**Запрос:** {t.query}")
                        st.write(f"**Ожидаемый ID:** `{t.expected_result_id or '-'}`")
                        if t.expected_result_ids:
                            st.write(f"**Или ID:** {', '.join(t.expected_result_ids)}")
                        st.write(f"**Макс. позиция:** {t.max_rank}, **Мин. score:** {t.min_score}")
                        if t.collection:
                            st.write(f"**Коллекция:** {t.collection}")
                        if t.description:
                            st.write(f"**Описание:** {t.description}")
                        if st.button("🗑️ Удалить", key=f"del_{t.id}"):
                            test_manager.delete_test(t.id)
                            st.rerun()

    with test_tab2:
        if test_manager:
            tests = test_manager.get_all_tests()
            if not tests:
                st.warning("Нет тестов")
            else:
                test_opts = {f"{t.name} [{t.search_mode}]": t.id for t in tests}
                selected = st.multiselect("Выбрать тесты (пусто = все):", list(test_opts.keys()))
                sel_ids = [test_opts[n] for n in selected] if selected else None

                if st.button("🚀 Запустить", type="primary", use_container_width=True):
                    if not embedder:
                        st.error("Embedder не инициализирован")
                    else:
                        run_tests = [t for t in tests if sel_ids is None or t.id in sel_ids]
                        progress = st.progress(0)
                        results_list = []
                        total_p, total_w, total_f = 0, 0, 0

                        for idx, tc in enumerate(run_tests):
                            progress.progress((idx + 1) / len(run_tests))
                            coll = tc.collection or collection
                            mode = tc.search_mode or "hybrid"

                            try:
                                emb = embedder.get_embedding_full(tc.query)
                                hits, elapsed = do_search(coll, emb, mode, 10, 0.0)

                                expected_ids = []
                                if tc.expected_result_id:
                                    expected_ids.append(tc.expected_result_id)
                                if tc.expected_result_ids:
                                    expected_ids.extend(tc.expected_result_ids)

                                found_rank = None
                                found_score = 0.0
                                found_id = None
                                for rank, hit in enumerate(hits, 1):
                                    hit_id = hit.payload.get('recipe_id', hit.payload.get('id'))
                                    if hit_id in expected_ids:
                                        found_rank = rank
                                        found_score = hit.score
                                        found_id = hit_id
                                        break

                                if found_rank is None:
                                    status = "FAILED"
                                    total_f += 1
                                elif found_rank > tc.max_rank:
                                    status = "WARNING"
                                    total_w += 1
                                elif found_score < tc.min_score:
                                    status = "WARNING"
                                    total_w += 1
                                else:
                                    status = "PASSED"
                                    total_p += 1

                                results_list.append({
                                    "Тест": tc.name,
                                    "Режим": mode,
                                    "Статус": {"PASSED": "✅", "WARNING": "⚠️", "FAILED": "❌"}[status],
                                    "Позиция": found_rank or "-",
                                    "Score": f"{found_score:.4f}" if found_score else "-",
                                    "Найден ID": found_id or "-",
                                    "Ожидали": ", ".join(expected_ids),
                                    "Время": f"{elapsed:.2f}с",
                                    "Топ-1": hits[0].payload.get('recipe_name', hits[0].payload.get('name', '?')) if hits else "-",
                                })
                            except Exception as e:
                                total_f += 1
                                results_list.append({
                                    "Тест": tc.name, "Режим": mode,
                                    "Статус": "❌", "Позиция": "-", "Score": "-",
                                    "Найден ID": "-", "Ожидали": "-",
                                    "Время": "-", "Топ-1": str(e)[:50],
                                })

                        progress.empty()

                        c1, c2, c3 = st.columns(3)
                        c1.metric("✅ Passed", total_p)
                        c2.metric("⚠️ Warning", total_w)
                        c3.metric("❌ Failed", total_f)

                        st.dataframe(pd.DataFrame(results_list), use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("---")
st.caption("🔍 Qdrant Search Tester • Hybrid search testing for BGE-M3 + Qdrant")
