# create_structure.ps1
Write-Host "Creating project structure for Qdrant Search Tester..." -ForegroundColor Green

# Create main directories
$directories = @(
    ".github/workflows",
    "data",
    "qdrant_test_scripts",
    "streamlit_dashboard/assets",
    "tests",
    "examples",
    "docs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Cyan
    }
    else {
        Write-Host "Already exists: $dir" -ForegroundColor Yellow
    }
}

Write-Host "`nCreating main files..." -ForegroundColor Green

# 1. requirements.txt
$requirements = @"
qdrant-client>=1.6.0
streamlit>=1.28.0
sentence-transformers>=2.2.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
plotly>=5.17.0
"@

Set-Content -Path "requirements.txt" -Value $requirements -Encoding UTF8
Write-Host "Created: requirements.txt" -ForegroundColor Cyan

# 2. .gitignore
$gitignore = @'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Test Reports
test_report_*.json
test_report_*.csv
reports/
test_reports/

# Qdrant Data
qdrant_data/
storage/

# Docker
docker-compose.override.yml

# Logs
*.log
logs/

# Secrets
secrets.ini
config.ini
config.json

# Temporary files
*.tmp
*.temp

# Streamlit
.streamlit/
'@

Set-Content -Path ".gitignore" -Value $gitignore -Encoding UTF8
Write-Host "Created: .gitignore" -ForegroundColor Cyan

# 3. .env.example
$envExample = @'
# Qdrant Connection
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Collection Settings
COLLECTION_NAME=distill_hybrid
COLLECTION_VECTOR_SIZE=1536
COLLECTION_DISTANCE=COSINE

# OpenAI (обязательно для эмбеддингов)
OPENAI_API_KEY=

# Embedding Model
EMBEDDING_MODEL=openai

# Test Settings
MAX_ALLOWED_RANK=3
MIN_SCORE_THRESHOLD=0.3
TEST_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Dashboard
DASHBOARD_PORT=8501
DASHBOARD_HOST=0.0.0.0
DASHBOARD_THEME=light

# Report Settings
REPORT_FORMAT=json,csv
REPORT_DIR=./reports
REPORT_RETENTION_DAYS=30
'@

Set-Content -Path ".env.example" -Value $envExample -Encoding UTF8
Write-Host "Created: .env.example" -ForegroundColor Cyan

# 4. Python init files
$initContent = "# Package initialization`r`n__version__ = '1.0.0'"

Set-Content -Path "qdrant_test_scripts\__init__.py" -Value $initContent -Encoding UTF8
Set-Content -Path "streamlit_dashboard\__init__.py" -Value $initContent -Encoding UTF8
Set-Content -Path "tests\__init__.py" -Value $initContent -Encoding UTF8

Write-Host "Created: __init__.py files" -ForegroundColor Cyan

Write-Host "`n✅ Project structure created successfully!" -ForegroundColor Green
Write-Host "Now add the remaining files:" -ForegroundColor Yellow
Write-Host "1. recipes_structured.json to data/" -ForegroundColor Yellow
Write-Host "2. recipes_full_text.json to data/" -ForegroundColor Yellow
Write-Host "3. Main scripts to respective directories" -ForegroundColor Yellow