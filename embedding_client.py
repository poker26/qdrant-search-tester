"""
Универсальный клиент для работы с моделями эмбеддингов.
Поддерживает:
  - BGE-M3 self-hosted (dense + sparse через /embed/qdrant)
  - OpenAI API (только dense)
"""
import os
import httpx
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMS = {
    "openai": 1536,
    "bge-m3": 1024,
}


@dataclass
class EmbeddingResult:
    """Результат: dense вектор + опционально sparse"""
    dense: List[float]
    sparse: Optional[Dict[str, Any]] = None  # {"indices": [...], "values": [...]}


class EmbeddingClient:
    def __init__(self):
        self.embedding_type = os.getenv('EMBEDDING_MODEL', 'bge-m3').lower()
        if self.embedding_type == 'bgm-m3':
            self.embedding_type = 'bge-m3'
        self._init_client()

    def _init_client(self):
        if self.embedding_type == 'openai':
            self._init_openai()
        elif self.embedding_type == 'bge-m3':
            self._init_bge_m3()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.embedding_type}")

    def _init_openai(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY не установлен")
        base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        proxy = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        http_client = httpx.Client(proxies=proxy, timeout=30.0) if proxy else None
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _init_bge_m3(self):
        url = os.getenv('BGE_M3_URL', os.getenv('BGM_M3_URL', 'http://172.17.0.1'))
        port = os.getenv('BGE_M3_PORT', os.getenv('BGM_M3_PORT', '8100'))
        if url.count(':') > 1:
            base = url
        else:
            base = f"{url}:{port}"
        self.api_url_qdrant = f"{base}/embed/qdrant"
        self.bge_m3_url = url
        self.bge_m3_port = port
        timeout = float(os.getenv('BGE_M3_TIMEOUT', os.getenv('BGM_M3_TIMEOUT', '60.0')))
        proxy = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        self.http_client = httpx.Client(proxies=proxy, timeout=timeout, follow_redirects=True)

    # --- public API ---

    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Dense-эмбеддинг(и). Обратная совместимость."""
        if self.embedding_type == 'openai':
            return self._get_openai_embedding(text)
        results = self._get_bge_m3_embeddings(text if isinstance(text, list) else [text])
        if isinstance(text, str):
            return results[0].dense
        return [r.dense for r in results]

    def get_embedding_full(self, text: Union[str, List[str]]) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Полный результат — dense + sparse (sparse только для BGE-M3)."""
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        if self.embedding_type == 'openai':
            dense_list = self._get_openai_embedding(texts)
            if not isinstance(dense_list[0], list):
                dense_list = [dense_list]
            results = [EmbeddingResult(dense=d) for d in dense_list]
        else:
            results = self._get_bge_m3_embeddings(texts)
        return results[0] if is_single else results

    # --- OpenAI ---

    def _get_openai_embedding(self, text):
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        response = self.client.embeddings.create(model="text-embedding-3-small", input=text)
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if is_single else embeddings

    # --- BGE-M3 ---

    def _get_bge_m3_embeddings(self, texts: List[str]) -> List[EmbeddingResult]:
        response = self.http_client.post(
            self.api_url_qdrant,
            json={"texts": texts},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append(EmbeddingResult(
                dense=item["dense"],
                sparse=item.get("sparse")
            ))
        if not results:
            raise RuntimeError(f"Пустой ответ от BGE-M3: {str(data)[:500]}")
        return results

    # --- meta ---

    def get_embedding_dim(self) -> int:
        return EMBEDDING_DIMS.get(self.embedding_type, 1024)

    def get_model_name(self) -> str:
        if self.embedding_type == 'openai':
            return "text-embedding-3-small"
        return f"bge-m3 ({self.bge_m3_url}:{self.bge_m3_port})"

    def supports_sparse(self) -> bool:
        return self.embedding_type == 'bge-m3'


_embedding_client: Optional[EmbeddingClient] = None

def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
