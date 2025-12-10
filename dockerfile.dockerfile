# Dockerfile для тестовой среды
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt requirements-dev.txt ./

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Копирование исходного кода
COPY . .

# Создание директории для отчетов
RUN mkdir -p /app/reports

# Настройка переменных окружения
ENV PYTHONPATH=/app
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV COLLECTION_NAME=test_recipes

# Команда по умолчанию
CMD ["python", "qdrant_test_scripts/test_runner.py"]