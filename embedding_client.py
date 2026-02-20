"""
Универсальный клиент для работы с моделями эмбеддингов
Поддерживает OpenAI API и self-hosted модели (bgm-m3)
"""
import os
import httpx
from typing import List, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# Размерности векторов для разных моделей
EMBEDDING_DIMS = {
    "openai": 1536,  # text-embedding-3-small
    "bgm-m3": 1024,  # BAAI General Multilingual M3
}


class EmbeddingClient:
    """Универсальный клиент для получения эмбеддингов"""
    
    def __init__(self):
        # По умолчанию используем bgm-m3 (1024 dim), можно переопределить через EMBEDDING_MODEL=openai
        self.embedding_type = os.getenv('EMBEDDING_MODEL', 'bgm-m3').lower()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Инициализация клиента в зависимости от типа модели"""
        if self.embedding_type == 'openai':
            self._init_openai()
        elif self.embedding_type == 'bgm-m3':
            self._init_bgm_m3()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {self.embedding_type}")
    
    def _init_openai(self):
        """Инициализация OpenAI клиента"""
        try:
            from openai import OpenAI
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY не установлен")
            
            openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
            http_client = httpx.Client(proxies=proxy_url, timeout=30.0) if proxy_url else None
            
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url,
                http_client=http_client
            )
        except ImportError:
            raise ImportError("Библиотека openai не установлена. Выполните: pip install openai")
    
    def _init_bgm_m3(self):
        """Инициализация клиента для self-hosted bgm-m3"""
        self.bgm_m3_url = os.getenv('BGM_M3_URL', 'http://46.173.25.31')
        self.bgm_m3_port = os.getenv('BGM_M3_PORT', '8000')
        self.bgm_m3_endpoint = os.getenv('BGM_M3_ENDPOINT', '/embed')
        
        # Полный URL для API (если порт уже в URL, не добавляем его)
        if ':' in self.bgm_m3_url and self.bgm_m3_url.count(':') > 1:
            # URL уже содержит порт (http://host:port)
            base_url = self.bgm_m3_url
        else:
            base_url = f"{self.bgm_m3_url}:{self.bgm_m3_port}"
        
        self.api_url = f"{base_url}{self.bgm_m3_endpoint}"
        
        # Таймаут для HTTP запросов
        timeout = float(os.getenv('BGM_M3_TIMEOUT', '60.0'))
        proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        
        self.http_client = httpx.Client(
            proxies=proxy_url,
            timeout=timeout,
            follow_redirects=True
        )
    
    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Получить эмбеддинг(и) для текста(ов)
        
        Args:
            text: Строка или список строк
            
        Returns:
            Для одной строки - список чисел (вектор)
            Для списка строк - список векторов
        """
        if self.embedding_type == 'openai':
            return self._get_openai_embedding(text)
        elif self.embedding_type == 'bgm-m3':
            return self._get_bgm_m3_embedding(text)
    
    def _get_openai_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Получить эмбеддинг через OpenAI API"""
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        embeddings = [item.embedding for item in response.data]
        
        if is_single:
            return embeddings[0]
        return embeddings
    
    def _get_bgm_m3_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Получить эмбеддинг через self-hosted bgm-m3 API"""
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # Пробуем разные форматы запроса для bgm-m3
        # Формат 1: {"inputs": ["text1", "text2"]}
        # Формат 2: {"texts": ["text1", "text2"]}
        # Формат 3: OpenAI-compatible {"input": "text"} или {"input": ["text1", "text2"]}
        
        payloads_to_try = [
            {"inputs": text},
            {"texts": text},
            {"input": text if len(text) > 1 else text[0]},
        ]
        
        last_error = None
        for payload in payloads_to_try:
            try:
                response = self.http_client.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Разные форматы ответа в зависимости от реализации API
                embeddings = None
                
                if isinstance(result, list):
                    # Прямой список векторов
                    embeddings = result
                elif isinstance(result, dict):
                    # Ищем embeddings в разных ключах
                    for key in ["embeddings", "data", "vectors", "embedding"]:
                        if key in result:
                            val = result[key]
                            if isinstance(val, list):
                                embeddings = val
                                break
                    
                    # Если не нашли, ищем первый список в значениях
                    if not embeddings:
                        for value in result.values():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], (int, float)):
                                    # Это один вектор
                                    embeddings = [value]
                                elif isinstance(value[0], list):
                                    # Это список векторов
                                    embeddings = value
                                break
                
                if embeddings and len(embeddings) > 0:
                    if is_single:
                        return embeddings[0] if isinstance(embeddings[0], list) else embeddings
                    return embeddings
                    
            except httpx.HTTPError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue
        
        # Если все форматы не сработали
        if last_error:
            raise ConnectionError(f"Ошибка подключения к bgm-m3 API ({self.api_url}): {last_error}")
        raise RuntimeError(f"Не удалось получить эмбеддинг от bgm-m3. Проверьте формат API.")
    
    def get_embedding_dim(self) -> int:
        """Получить размерность вектора для текущей модели"""
        # По умолчанию 1024 (BGM-M3), если модель не найдена
        return EMBEDDING_DIMS.get(self.embedding_type, 1024)
    
    def get_model_name(self) -> str:
        """Получить название модели"""
        if self.embedding_type == 'openai':
            return "text-embedding-3-small"
        elif self.embedding_type == 'bgm-m3':
            return f"bgm-m3 ({self.bgm_m3_url}:{self.bgm_m3_port})"
        return self.embedding_type


# Глобальный экземпляр клиента (singleton)
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """Получить глобальный экземпляр клиента эмбеддингов"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
