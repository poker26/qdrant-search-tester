"""
Менеджер тестов для Qdrant Search Tester.
"""
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from datetime import datetime


@dataclass
class TestCase:
    id: str
    name: str
    query: str
    expected_result_id: Optional[str] = None
    expected_result_ids: Optional[List[str]] = None
    max_rank: int = 3
    min_score: float = 0.3
    search_mode: str = "hybrid"       # dense | sparse | hybrid
    collection: Optional[str] = None  # если None — используется текущая из UI
    description: str = ""
    created_at: str = ""
    updated_at: str = ""


class TestManager:
    def __init__(self, tests_file: str = "tests.json"):
        self.tests_file = tests_file
        self.tests: List[TestCase] = []
        self.load_tests()

    def load_tests(self) -> List[TestCase]:
        if os.path.exists(self.tests_file):
            try:
                with open(self.tests_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tests = []
                    for t in data.get('tests', []):
                        # backward compat: ignore unknown fields
                        valid_fields = {f.name for f in TestCase.__dataclass_fields__.values()}
                        filtered = {k: v for k, v in t.items() if k in valid_fields}
                        self.tests.append(TestCase(**filtered))
            except Exception as e:
                print(f"Ошибка загрузки тестов: {e}")
                self.tests = []
        return self.tests

    def save_tests(self):
        data = {
            'version': '2.0',
            'updated_at': datetime.now().isoformat(),
            'tests': [asdict(t) for t in self.tests]
        }
        with open(self.tests_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_test(self, test: TestCase) -> bool:
        if not test.id:
            test.id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not test.created_at:
            test.created_at = datetime.now().isoformat()
        test.updated_at = datetime.now().isoformat()
        if any(t.id == test.id for t in self.tests):
            return False
        self.tests.append(test)
        self.save_tests()
        return True

    def update_test(self, test_id: str, **kwargs) -> bool:
        for test in self.tests:
            if test.id == test_id:
                for k, v in kwargs.items():
                    if hasattr(test, k):
                        setattr(test, k, v)
                test.updated_at = datetime.now().isoformat()
                self.save_tests()
                return True
        return False

    def delete_test(self, test_id: str) -> bool:
        orig = len(self.tests)
        self.tests = [t for t in self.tests if t.id != test_id]
        if len(self.tests) < orig:
            self.save_tests()
            return True
        return False

    def get_test(self, test_id: str) -> Optional[TestCase]:
        return next((t for t in self.tests if t.id == test_id), None)

    def get_all_tests(self) -> List[TestCase]:
        return self.tests
