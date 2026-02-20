"""
Менеджер тестов для Qdrant Search Tester
Позволяет создавать, редактировать и запускать тесты через JSON файл
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TestCase:
    """Тест-кейс для проверки поиска"""
    id: str  # Уникальный ID теста
    name: str  # Название теста для удобства
    query: str  # Поисковый запрос
    expected_result_id: Optional[str] = None  # ID рецепта, который должен быть найден
    expected_result_ids: Optional[List[str]] = None  # Список ID, один из которых должен быть найден
    max_rank: int = 3  # Максимальная позиция в результатах (1 = первое место)
    min_score: float = 0.3  # Минимальный порог релевантности
    description: str = ""  # Описание теста
    created_at: str = ""  # Дата создания
    updated_at: str = ""  # Дата обновления

class TestManager:
    """Менеджер для работы с тестами"""
    
    def __init__(self, tests_file: str = "tests.json"):
        self.tests_file = tests_file
        self.tests: List[TestCase] = []
        self.load_tests()
    
    def load_tests(self) -> List[TestCase]:
        """Загрузить тесты из JSON файла"""
        if os.path.exists(self.tests_file):
            try:
                with open(self.tests_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tests = [TestCase(**test) for test in data.get('tests', [])]
            except Exception as e:
                print(f"Ошибка загрузки тестов: {e}")
                self.tests = []
        else:
            self.tests = []
        return self.tests
    
    def save_tests(self):
        """Сохранить тесты в JSON файл"""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'tests': [asdict(test) for test in self.tests]
        }
        with open(self.tests_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_test(self, test: TestCase) -> bool:
        """Добавить новый тест"""
        if not test.id:
            test.id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if not test.created_at:
            test.created_at = datetime.now().isoformat()
        
        test.updated_at = datetime.now().isoformat()
        
        # Проверяем на дубликаты
        if any(t.id == test.id for t in self.tests):
            return False
        
        self.tests.append(test)
        self.save_tests()
        return True
    
    def update_test(self, test_id: str, **kwargs) -> bool:
        """Обновить существующий тест"""
        for test in self.tests:
            if test.id == test_id:
                for key, value in kwargs.items():
                    if hasattr(test, key):
                        setattr(test, key, value)
                test.updated_at = datetime.now().isoformat()
                self.save_tests()
                return True
        return False
    
    def delete_test(self, test_id: str) -> bool:
        """Удалить тест"""
        original_count = len(self.tests)
        self.tests = [t for t in self.tests if t.id != test_id]
        if len(self.tests) < original_count:
            self.save_tests()
            return True
        return False
    
    def get_test(self, test_id: str) -> Optional[TestCase]:
        """Получить тест по ID"""
        for test in self.tests:
            if test.id == test_id:
                return test
        return None
    
    def get_all_tests(self) -> List[TestCase]:
        """Получить все тесты"""
        return self.tests
    
    def get_test_ids(self) -> List[str]:
        """Получить список ID всех тестов"""
        return [t.id for t in self.tests]
