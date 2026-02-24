"""
CLI тестовый раннер для Qdrant с поддержкой hybrid/dense/sparse.
"""
import json
import os
import sys
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test_manager import TestManager, TestCase
from embedding_client import get_embedding_client

load_dotenv()


class QdrantTesterV2:
    def __init__(self, collection_name=None, tests_file=None):
        url = os.getenv('QDRANT_URL')
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', '6333'))
        api_key = os.getenv('QDRANT_API_KEY')

        if url:
            kwargs = {"url": url, "check_compatibility": False}
            if api_key:
                kwargs["api_key"] = api_key
            self.client = QdrantClient(**kwargs)
        else:
            self.client = QdrantClient(host=host, port=port)

        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'distill_hybrid_v2')
        self.embedding_client = get_embedding_client()
        tests_file = tests_file or os.path.join(os.path.dirname(__file__), '..', 'tests.json')
        self.test_manager = TestManager(tests_file=tests_file)

    def _search(self, collection, emb_result, mode, limit=10):
        if mode == "dense":
            resp = self.client.query_points(
                collection_name=collection,
                query=emb_result.dense,
                using="dense",
                limit=limit,
                with_payload=True,
            )
            return resp.points

        elif mode == "sparse" and emb_result.sparse:
            resp = self.client.query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=emb_result.sparse["indices"],
                    values=emb_result.sparse["values"]
                ),
                using="sparse",
                limit=limit,
                with_payload=True,
            )
            return resp.points

        elif mode == "hybrid" and emb_result.sparse:
            resp = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(query=emb_result.dense, using="dense", limit=limit * 3),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=emb_result.sparse["indices"],
                            values=emb_result.sparse["values"]
                        ),
                        using="sparse",
                        limit=limit * 3,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            return resp.points

        # fallback
        resp = self.client.query_points(
            collection_name=collection,
            query=emb_result.dense,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        return resp.points

    def run_single_test(self, tc: TestCase) -> Dict[str, Any]:
        collection = tc.collection or self.collection_name
        mode = tc.search_mode or "hybrid"

        emb = self.embedding_client.get_embedding_full(tc.query)
        hits = self._search(collection, emb, mode)

        expected_ids = []
        if tc.expected_result_id:
            expected_ids.append(tc.expected_result_id)
        if tc.expected_result_ids:
            expected_ids.extend(tc.expected_result_ids)

        found_rank, found_score, found_id = None, 0.0, None
        for rank, hit in enumerate(hits, 1):
            hit_id = hit.payload.get('recipe_id', hit.payload.get('id'))
            if hit_id in expected_ids:
                found_rank = rank
                found_score = hit.score
                found_id = hit_id
                break

        if not expected_ids:
            status = "PASSED" if hits else "FAILED"
            message = f"Найдено {len(hits)} результатов (expected_id не задан)"
        elif found_rank is None:
            status = "FAILED"
            message = f"Не найден в топ-10 (ожидали: {', '.join(expected_ids)})"
        elif found_rank > tc.max_rank:
            status = "WARNING"
            message = f"Позиция {found_rank} > макс {tc.max_rank}"
        elif found_score < tc.min_score:
            status = "WARNING"
            message = f"Score {found_score:.3f} < мин {tc.min_score}"
        else:
            status = "PASSED"
            message = f"Позиция {found_rank}, score {found_score:.3f}"

        return {
            "test_id": tc.id,
            "test_name": tc.name,
            "query": tc.query,
            "mode": mode,
            "status": status,
            "rank": found_rank or "N/A",
            "score": f"{found_score:.3f}" if found_score else "N/A",
            "found_id": found_id or "N/A",
            "expected_ids": expected_ids,
            "message": message,
            "top_results": [
                {
                    "rank": i + 1,
                    "id": h.payload.get('recipe_id', h.payload.get('id', 'N/A')),
                    "name": h.payload.get('recipe_name', h.payload.get('name', 'N/A')),
                    "score": f"{h.score:.3f}"
                }
                for i, h in enumerate(hits[:5])
            ]
        }

    def run_tests(self, test_ids=None):
        if test_ids:
            tests = [t for t in self.test_manager.get_all_tests() if t.id in test_ids]
        else:
            tests = self.test_manager.get_all_tests()

        if not tests:
            return {"summary": {"total": 0, "passed": 0, "warning": 0, "failed": 0},
                    "results": [], "success_rate": 0.0}

        results = []
        p, w, f = 0, 0, 0
        for tc in tests:
            r = self.run_single_test(tc)
            results.append(r)
            if r['status'] == 'PASSED': p += 1
            elif r['status'] == 'WARNING': w += 1
            else: f += 1

        total = len(tests)
        return {
            "summary": {"total": total, "passed": p, "warning": w, "failed": f},
            "results": results,
            "success_rate": (p / total * 100) if total else 0
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Qdrant search tests')
    parser.add_argument('--tests', nargs='+', help='Test IDs')
    parser.add_argument('--tests-file', help='Path to tests.json')
    parser.add_argument('--collection', help='Collection name')
    args = parser.parse_args()

    tester = QdrantTesterV2(collection_name=args.collection, tests_file=args.tests_file)
    out = tester.run_tests(test_ids=args.tests)

    print("\n" + "=" * 60)
    s = out['summary']
    print(f"РЕЗУЛЬТАТЫ: {s['total']} тестов | ✅ {s['passed']} | ⚠️ {s['warning']} | ❌ {s['failed']} | {out['success_rate']:.0f}%")
    print("=" * 60)
    for r in out['results']:
        icon = "✅" if r['status'] == 'PASSED' else "⚠️" if r['status'] == 'WARNING' else "❌"
        print(f"{icon} [{r['mode']}] {r['test_name']}: {r['message']}")
        if r['top_results']:
            for tr in r['top_results'][:3]:
                print(f"     #{tr['rank']} {tr['name']} (score: {tr['score']})")
