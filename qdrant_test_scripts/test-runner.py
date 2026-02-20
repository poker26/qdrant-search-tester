"""
Автоматические тесты для проверки поиска в Qdrant.
Использует OpenAI text-embedding-3-small для эмбеддингов.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class TestCase:
    """Тест-кейс для проверки поиска"""
    recipe_id: str
    recipe_name: str
    test_queries: List[str]
    expected_keywords: List[str]
    max_allowed_rank: int = 3  # Максимальная позиция в результатах
    min_score_threshold: float = 0.3  # Минимальный порог релевантности

class QdrantTester:
    def __init__(self, host=None, port=None, url=None, api_key=None, collection_name=None):
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

        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self) -> List[TestCase]:
        """Загружаем тест-кейсы"""
        return [
            TestCase(
                recipe_id="vodka_potato_tech",
                recipe_name="Водка из картофеля",
                test_queries=[
                    "технология запаривания картофеля",
                    "дробилка Браунфельзера для измельчения",
                    "температура брожения картофельного сусла",
                    "использование солода для осахаривания крахмала",
                    "производство спирта из картофеля на заводе"
                ],
                expected_keywords=["запарник", "крахмал", "осахаривание", "дробилка", "брожение"]
            ),
            TestCase(
                recipe_id="vodka_beetroot_tech",
                recipe_name="Водка из свеклы",
                test_queries=[
                    "метод Шампонуа для свеклы",
                    "трёхкратное вымачивание бардой",
                    "производство водки из свекольного сока",
                    "использование серной кислоты при брожении",
                    "технология переработки свеклы на спирт"
                ],
                expected_keywords=["Шампонуа", "барда", "вымачивание", "свекла", "серная кислота"]
            ),
            TestCase(
                recipe_id="vodka_topinambur_tech",
                recipe_name="Водка из топинамбура",
                test_queries=[
                    "эксперименты с топинамбуром 1857 год",
                    "диффузионная батарея Шюценбаха",
                    "производство водки из земляной груши",
                    "холодное вымачивание топинамбура",
                    "технология переработки топинамбура на спирт"
                ],
                expected_keywords=["топинамбур", "диффузионная батарея", "1857", "земляная груша", "холодное вымачивание"]
            )
        ]
    
    def run_single_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Запуск одного тест-кейса"""
        results = []
        
        for query in test_case.test_queries:
            response = self.openai_client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Используем query_points вместо search (новый API qdrant-client)
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
            
            found_rank = None
            found_score = 0.0
            
            for rank, hit in enumerate(search_result, 1):
                if hit.payload.get('id') == test_case.recipe_id:
                    found_rank = rank
                    found_score = hit.score
                    break
            
            if found_rank is None:
                status = "FAILED"
                message = f"Рецепт не найден в топ-10"
            elif found_rank > test_case.max_allowed_rank:
                status = "WARNING"
                message = f"Найден на позиции {found_rank} (допустимо до {test_case.max_allowed_rank})"
            elif found_score < test_case.min_score_threshold:
                status = "WARNING"
                message = f"Низкий score: {found_score:.3f} (минимум {test_case.min_score_threshold})"
            else:
                status = "PASSED"
                message = f"Найден на позиции {found_rank} (score: {found_score:.3f})"
            
            results.append({
                "query": query,
                "status": status,
                "rank": found_rank or "N/A",
                "score": f"{found_score:.3f}" if found_score else "N/A",
                "message": message
            })
        
        return {
            "recipe_id": test_case.recipe_id,
            "recipe_name": test_case.recipe_name,
            "results": results,
            "summary": self._summarize_test(results)
        }
    
    def _summarize_test(self, results: List[Dict]) -> Dict:
        """Суммируем результаты теста"""
        total = len(results)
        passed = sum(1 for r in results if r['status'] == 'PASSED')
        warning = sum(1 for r in results if r['status'] == 'WARNING')
        failed = sum(1 for r in results if r['status'] == 'FAILED')
        
        return {
            "total_queries": total,
            "passed": passed,
            "warning": warning,
            "failed": failed,
            "success_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%"
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Запуск всех тестов"""
        print("Запуск автоматических тестов поиска в Qdrant")
        print("=" * 60)
        
        all_results = []
        summary = {
            "total_tests": 0,
            "total_passed": 0,
            "total_warning": 0,
            "total_failed": 0
        }
        
        for test_case in self.test_cases:
            print(f"\nТестируем: {test_case.recipe_name}")
            print("-" * 40)
            
            result = self.run_single_test(test_case)
            all_results.append(result)
            
            for res in result['results']:
                status_icon = "OK" if res['status'] == 'PASSED' else "WARN" if res['status'] == 'WARNING' else "FAIL"
                print(f"[{status_icon}] Запрос: '{res['query']}'")
                print(f"   Результат: {res['message']}")
            
            summary['total_tests'] += result['summary']['total_queries']
            summary['total_passed'] += result['summary']['passed']
            summary['total_warning'] += result['summary']['warning']
            summary['total_failed'] += result['summary']['failed']
            
            print(f"\nИтог по рецепту: {result['summary']['success_rate']} успешных запросов")
        
        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЙ ОТЧЕТ")
        print("=" * 60)
        
        success_rate = (summary['total_passed'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        print(f"Всего тестов: {summary['total_tests']}")
        print(f"Успешно: {summary['total_passed']}")
        print(f"С предупреждениями: {summary['total_warning']}")
        print(f"Неудачно: {summary['total_failed']}")
        print(f"Общий успех: {success_rate:.1f}%")
        
        self.save_report(all_results, summary)
        
        return {
            "detailed_results": all_results,
            "summary": summary,
            "success_rate": success_rate
        }
    
    def save_report(self, results: List[Dict], summary: Dict):
        """Сохраняем отчет в файлы"""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            "timestamp": timestamp,
            "summary": summary,
            "detailed_results": results
        }
        
        with open(f'test_report_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        csv_data = []
        for recipe_result in results:
            for query_result in recipe_result['results']:
                csv_data.append({
                    'recipe_id': recipe_result['recipe_id'],
                    'recipe_name': recipe_result['recipe_name'],
                    'query': query_result['query'],
                    'status': query_result['status'],
                    'rank': query_result['rank'],
                    'score': query_result['score'],
                    'message': query_result['message']
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(f'test_report_{timestamp}.csv', index=False, encoding='utf-8')
        
        print(f"\nОтчеты сохранены: test_report_{timestamp}.json, test_report_{timestamp}.csv")

if __name__ == "__main__":
    tester = QdrantTester()
    tester.run_all_tests()
