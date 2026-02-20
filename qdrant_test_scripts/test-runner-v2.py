"""
Обновленный тестовый раннер для работы с гибкими тестами из JSON
"""
import json
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import sys

# Добавляем путь к test_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test_manager import TestManager, TestCase

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


class QdrantTesterV2:
    def __init__(self, host=None, port=None, url=None, api_key=None, collection_name=None, tests_file=None):
        qdrant_url = url or os.getenv('QDRANT_URL')
        qdrant_host = host or os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = port or int(os.getenv('QDRANT_PORT', '6333'))
        qdrant_api_key = api_key or os.getenv('QDRANT_API_KEY')
        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'distill_hybrid')
        
        if qdrant_url:
            if qdrant_api_key:
                self.client = QdrantClient(
                    url=qdrant_url, 
                    api_key=qdrant_api_key,
                    check_compatibility=False
                )
            else:
                self.client = QdrantClient(url=qdrant_url, check_compatibility=False)
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY не задан. Укажите ключ в .env или переменных окружения."
            )
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Инициализируем менеджер тестов
        tests_file = tests_file or os.path.join(os.path.dirname(__file__), '..', 'tests.json')
        self.test_manager = TestManager(tests_file=tests_file)
    
    def run_single_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Запуск одного тест-кейса"""
        # Получаем эмбеддинг запроса
        response = self.openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=test_case.query
        )
        query_embedding = response.data[0].embedding
        
        # Выполняем поиск
        try:
            query_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                limit=10,
                with_payload=True
            )
            search_result = query_response.points
        except Exception as vec_err:
            err_msg = str(vec_err).lower()
            if "dense" in err_msg and ("not existing" in err_msg or "vector name" in err_msg):
                # Fallback для коллекций с default вектором
                query_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    limit=10,
                    with_payload=True
                )
                search_result = query_response.points
            else:
                raise
        
        # Проверяем результаты
        found_rank = None
        found_score = 0.0
        found_id = None
        
        # Определяем ожидаемые ID
        expected_ids = []
        if test_case.expected_result_id:
            expected_ids.append(test_case.expected_result_id)
        if test_case.expected_result_ids:
            expected_ids.extend(test_case.expected_result_ids)
        
        # Ищем ожидаемый результат в результатах поиска
        for rank, hit in enumerate(search_result, 1):
            hit_id = hit.payload.get('id')
            if hit_id in expected_ids:
                found_rank = rank
                found_score = hit.score
                found_id = hit_id
                break
        
        # Определяем статус теста
        if found_rank is None:
            status = "FAILED"
            message = f"Ожидаемый результат не найден в топ-10"
            if expected_ids:
                message += f" (ожидались ID: {', '.join(expected_ids)})"
        elif found_rank > test_case.max_rank:
            status = "WARNING"
            message = f"Найден на позиции {found_rank} (допустимо до {test_case.max_rank})"
        elif found_score < test_case.min_score:
            status = "WARNING"
            message = f"Низкий score: {found_score:.3f} (минимум {test_case.min_score})"
        else:
            status = "PASSED"
            message = f"Найден на позиции {found_rank} (score: {found_score:.3f}, ID: {found_id})"
        
        return {
            "test_id": test_case.id,
            "test_name": test_case.name,
            "query": test_case.query,
            "status": status,
            "rank": found_rank or "N/A",
            "score": f"{found_score:.3f}" if found_score else "N/A",
            "found_id": found_id or "N/A",
            "expected_ids": expected_ids,
            "message": message,
            "top_results": [
                {
                    "rank": i + 1,
                    "id": hit.payload.get('id', 'N/A'),
                    "name": hit.payload.get('name', 'N/A'),
                    "score": f"{hit.score:.3f}"
                }
                for i, hit in enumerate(search_result[:5])
            ]
        }
    
    def run_tests(self, test_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Запуск тестов (всех или выбранных)"""
        if test_ids is None:
            # Запускаем все тесты
            tests = self.test_manager.get_all_tests()
        else:
            # Запускаем только выбранные
            tests = [self.test_manager.get_test(tid) for tid in test_ids if self.test_manager.get_test(tid)]
            tests = [t for t in tests if t is not None]
        
        if not tests:
            return {
                "summary": {
                    "total_tests": 0,
                    "total_passed": 0,
                    "total_warning": 0,
                    "total_failed": 0
                },
                "detailed_results": [],
                "success_rate": 0.0
            }
        
        results = []
        summary = {
            "total_tests": 0,
            "total_passed": 0,
            "total_warning": 0,
            "total_failed": 0
        }
        
        for test_case in tests:
            result = self.run_single_test(test_case)
            results.append(result)
            
            summary['total_tests'] += 1
            if result['status'] == 'PASSED':
                summary['total_passed'] += 1
            elif result['status'] == 'WARNING':
                summary['total_warning'] += 1
            else:
                summary['total_failed'] += 1
        
        success_rate = (summary['total_passed'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        return {
            "summary": summary,
            "detailed_results": results,
            "success_rate": success_rate
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Запуск тестов Qdrant')
    parser.add_argument('--tests', nargs='+', help='ID тестов для запуска (если не указано - все тесты)')
    parser.add_argument('--tests-file', help='Путь к файлу с тестами')
    args = parser.parse_args()
    
    tester = QdrantTesterV2(tests_file=args.tests_file)
    results = tester.run_tests(test_ids=args.tests)
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("=" * 60)
    print(f"Всего тестов: {results['summary']['total_tests']}")
    print(f"Успешно: {results['summary']['total_passed']}")
    print(f"С предупреждениями: {results['summary']['total_warning']}")
    print(f"Неудачно: {results['summary']['total_failed']}")
    print(f"Успешность: {results['success_rate']:.1f}%")
    print("\nДетальные результаты:")
    for result in results['detailed_results']:
        status_icon = "✅" if result['status'] == 'PASSED' else "⚠️" if result['status'] == 'WARNING' else "❌"
        print(f"\n{status_icon} [{result['test_name']}] {result['query']}")
        print(f"   {result['message']}")
