# Настройка подключения к облачному Qdrant

## Шаг 1: Создайте файл .env

Создайте файл `.env` в корне проекта со следующим содержимым:

```env
# Облачный Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key-here

# Имя коллекции
COLLECTION_NAME=test_recipes
```

## Шаг 2: Получите параметры подключения

1. **QDRANT_URL** - URL вашего облачного кластера Qdrant
   - Обычно имеет формат: `https://xxxxx-xxxxx-xxxxx.qdrant.io`
   - Или: `https://your-cluster-name.qdrant.io`

2. **QDRANT_API_KEY** - API ключ для аутентификации
   - Можно получить в панели управления вашего облачного Qdrant
   - Обычно находится в разделе "API Keys" или "Settings"

## Шаг 3: Запустите скрипт настройки

После создания файла `.env` с правильными параметрами, запустите:

```bash
python qdrant_test_scripts/setup-qdrant.py
```

## Альтернативный способ: переменные окружения

Вы также можете задать параметры через переменные окружения:

**Windows PowerShell:**
```powershell
$env:QDRANT_URL="https://your-cluster.qdrant.io"
$env:QDRANT_API_KEY="your-api-key"
python qdrant_test_scripts/setup-qdrant.py
```

**Windows CMD:**
```cmd
set QDRANT_URL=https://your-cluster.qdrant.io
set QDRANT_API_KEY=your-api-key
python qdrant_test_scripts/setup-qdrant.py
```

## Примечания

- Если указан `QDRANT_URL`, скрипты будут использовать его для подключения к облачному Qdrant
- Если `QDRANT_URL` не указан, будет использоваться локальное подключение через `QDRANT_HOST` и `QDRANT_PORT`
- API ключ обязателен для большинства облачных инстансов Qdrant

## Устранение проблем

### Ошибка 403 Forbidden

Если вы получаете ошибку `403 Forbidden`, проверьте:

1. **Правильность API ключа** - убедитесь, что ключ скопирован полностью без лишних пробелов
2. **Права доступа API ключа** - ключ должен иметь права на чтение коллекций и выполнение поиска
3. **Срок действия ключа** - проверьте, не истек ли срок действия API ключа
4. **Имя коллекции** - убедитесь, что коллекция `distill_hybrid` существует и доступна

### Проверка подключения

После настройки `.env` файла, проверьте подключение:

```bash
python qdrant_test_scripts/setup-qdrant.py
```

Скрипт проверит подключение к коллекции и выведет информацию о ней.

