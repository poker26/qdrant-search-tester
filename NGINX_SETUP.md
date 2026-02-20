# Настройка Nginx для доступа к Streamlit через домен

## Требования

- Ubuntu сервер с доменом `gopa.begemot26.ru`
- Streamlit запущен на `localhost:8501`
- Root доступ к серверу

## Шаги установки

### 1. Установка Nginx

```bash
sudo apt update
sudo apt install -y nginx
```

### 2. Установка Certbot для SSL сертификатов

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 3. Копирование конфигурации Nginx

```bash
# Скопируйте файл nginx/qtester.conf в /etc/nginx/sites-available/
sudo cp nginx/qtester.conf /etc/nginx/sites-available/qtester.conf

# Создайте символическую ссылку
sudo ln -s /etc/nginx/sites-available/qtester.conf /etc/nginx/sites-enabled/qtester.conf

# Удалите дефолтный конфиг (опционально)
sudo rm /etc/nginx/sites-enabled/default
```

### 4. Проверка конфигурации Nginx

```bash
sudo nginx -t
```

Если всё ок, вы увидите:
```
nginx: configuration file /etc/nginx/nginx.conf test is successful
```

### 5. Настройка Streamlit для работы с base path

Создайте или обновите файл `.streamlit/config.toml` в директории проекта:

```bash
mkdir -p ~/qdrant-search-tester/.streamlit
cp streamlit_config.toml.example ~/qdrant-search-tester/.streamlit/config.toml
```

Или создайте вручную:

```bash
mkdir -p ~/qdrant-search-tester/.streamlit
cat > ~/qdrant-search-tester/.streamlit/config.toml << EOF
[server]
baseUrlPath = "/qtester"
enableCORS = false
enableXsrfProtection = false
EOF
```

Или установите через переменную окружения в `ecosystem.config.cjs`:

```javascript
env: {
    NODE_ENV: 'production',
    STREAMLIT_SERVER_BASE_URL_PATH: '/qtester'
}
```

### 6. Перезапуск Nginx

```bash
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### 7. Получение SSL сертификата

```bash
sudo certbot --nginx -d gopa.begemot26.ru
```

Certbot автоматически:
- Получит SSL сертификат от Let's Encrypt
- Обновит конфигурацию Nginx
- Настроит автообновление сертификата

### 8. Перезапуск Streamlit с новыми настройками

```bash
cd ~/qdrant-search-tester
pm2 restart qdrant-tester
```

Или если используете npm:

```bash
npm run restart
```

## Проверка

После настройки вы сможете открыть сервис по адресу:
- **HTTPS:** https://gopa.begemot26.ru/qtester/
- **HTTP:** автоматически редиректит на HTTPS

## Устранение проблем

### Проверка статуса Nginx
```bash
sudo systemctl status nginx
```

### Просмотр логов
```bash
sudo tail -f /var/log/nginx/qtester_error.log
sudo tail -f /var/log/nginx/qtester_access.log
```

### Проверка портов
```bash
sudo netstat -tlnp | grep 8501
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :443
```

### Если Streamlit не работает с base path

Убедитесь, что в `.streamlit/config.toml` или переменных окружения установлен:
```
baseUrlPath = "/qtester"
```

### Обновление SSL сертификата

Certbot автоматически настроит cron для обновления. Проверить можно:
```bash
sudo certbot renew --dry-run
```

## Альтернатива: без base path (если Streamlit не поддерживает)

Если Streamlit не работает с base path, можно использовать другой подход - проксировать весь домен на Streamlit:

```nginx
location / {
    proxy_pass http://127.0.0.1:8501/;
    # ... остальные настройки proxy
}
```

Но тогда Streamlit будет доступен по корню домена, а не по `/qtester`.
