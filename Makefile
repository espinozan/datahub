# DataHub - Makefile
# Comandos de desarrollo y deployment de nivel empresarial

.PHONY: help install dev test lint format clean docker-up docker-down migrate

# Colores para output
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
BLUE   := $(shell tput -Txterm setaf 4)
RESET  := $(shell tput -Txterm sgr0)

help: ## Mostrar esta ayuda
	@echo '$(GREEN)DataHub - Sistema de Gestión de Datasets para IA$(RESET)'
	@echo ''
	@echo 'Comandos disponibles:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'

install: ## Instalar dependencias del proyecto
	@echo '$(YELLOW)Instalando dependencias...$(RESET)'
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo '$(GREEN)✓ Dependencias instaladas$(RESET)'

dev: ## Configurar entorno de desarrollo
	@echo '$(YELLOW)Configurando entorno de desarrollo...$(RESET)'
	cp .env.example .env
	pre-commit install
	docker-compose up -d postgres redis minio
	@echo '$(GREEN)✓ Entorno listo$(RESET)'

test: ## Ejecutar todos los tests
	@echo '$(YELLOW)Ejecutando tests...$(RESET)'
	pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

test-unit: ## Ejecutar solo tests unitarios
	@echo '$(YELLOW)Ejecutando tests unitarios...$(RESET)'
	pytest tests/unit/ -v

test-integration: ## Ejecutar tests de integración
	@echo '$(YELLOW)Ejecutando tests de integración...$(RESET)'
	pytest tests/integration/ -v

test-e2e: ## Ejecutar tests end-to-end
	@echo '$(YELLOW)Ejecutando tests e2e...$(RESET)'
	pytest tests/e2e/ -v

lint: ## Ejecutar linters y type checking
	@echo '$(YELLOW)Ejecutando linters...$(RESET)'
	ruff check app/ tests/
	mypy app/
	@echo '$(GREEN)✓ Linting completo$(RESET)'

format: ## Formatear código
	@echo '$(YELLOW)Formateando código...$(RESET)'
	black app/ tests/
	isort app/ tests/
	@echo '$(GREEN)✓ Código formateado$(RESET)'

clean: ## Limpiar archivos generados
	@echo '$(YELLOW)Limpiando archivos...$(RESET)'
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/ dist/ build/
	@echo '$(GREEN)✓ Limpieza completa$(RESET)'

docker-up: ## Iniciar servicios Docker
	@echo '$(YELLOW)Iniciando servicios Docker...$(RESET)'
	docker-compose up -d
	@echo '$(GREEN)✓ Servicios iniciados$(RESET)'

docker-down: ## Detener servicios Docker
	@echo '$(YELLOW)Deteniendo servicios Docker...$(RESET)'
	docker-compose down
	@echo '$(GREEN)✓ Servicios detenidos$(RESET)'

docker-logs: ## Ver logs de servicios Docker
	docker-compose logs -f

migrate: ## Ejecutar migraciones de base de datos
	@echo '$(YELLOW)Ejecutando migraciones...$(RESET)'
	alembic upgrade head
	@echo '$(GREEN)✓ Migraciones aplicadas$(RESET)'

migrate-create: ## Crear nueva migración
	@read -p "Nombre de la migración: " name; \
	alembic revision --autogenerate -m "$$name"

run: ## Ejecutar servidor de desarrollo
	@echo '$(YELLOW)Iniciando servidor de desarrollo...$(RESET)'
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Ejecutar servidor de producción
	@echo '$(YELLOW)Iniciando servidor de producción...$(RESET)'
	gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

docs: ## Generar documentación
	@echo '$(YELLOW)Generando documentación...$(RESET)'
	mkdocs build
	@echo '$(GREEN)✓ Documentación generada en site/$(RESET)'

docs-serve: ## Servir documentación localmente
	mkdocs serve

shell: ## Abrir shell de Python con contexto de aplicación
	@echo '$(YELLOW)Abriendo shell...$(RESET)'
	python -i scripts/shell.py

db-shell: ## Conectar a base de datos PostgreSQL
	docker-compose exec postgres psql -U datahub_user -d datahub

redis-cli: ## Conectar a Redis CLI
	docker-compose exec redis redis-cli -a redis_password_2024

coverage-report: ## Generar y abrir reporte de coverage
	pytest --cov=app --cov-report=html
	open htmlcov/index.html

security-check: ## Ejecutar análisis de seguridad
	@echo '$(YELLOW)Ejecutando análisis de seguridad...$(RESET)'
	bandit -r app/ -f json -o security-report.json
	safety check --json
	@echo '$(GREEN)✓ Análisis completo$(RESET)'

benchmark: ## Ejecutar benchmarks de performance
	@echo '$(YELLOW)Ejecutando benchmarks...$(RESET)'
	pytest tests/performance/ --benchmark-only

init-db: ## Inicializar base de datos con data de ejemplo
	@echo '$(YELLOW)Inicializando base de datos...$(RESET)'
	python scripts/init_database.py
	@echo '$(GREEN)✓ Base de datos inicializada$(RESET)'

seed-data: ## Cargar datos de ejemplo
	@echo '$(YELLOW)Cargando datos de ejemplo...$(RESET)'
	python scripts/seed_data.py
	@echo '$(GREEN)✓ Datos cargados$(RESET)'

build: ## Build imagen Docker
	@echo '$(YELLOW)Construyendo imagen Docker...$(RESET)'
	docker build -t datahub:latest .
	@echo '$(GREEN)✓ Imagen construida$(RESET)'

deploy-staging: ## Deploy a staging
	@echo '$(YELLOW)Deployando a staging...$(RESET)'
	./scripts/deploy.sh staging

deploy-prod: ## Deploy a producción
	@echo '$(YELLOW)Deployando a producción...$(RESET)'
	./scripts/deploy.sh production

health-check: ## Verificar health de servicios
	@echo '$(YELLOW)Verificando health de servicios...$(RESET)'
	curl -f http://localhost:8000/health || echo "API: DOWN"
	docker-compose ps

all: clean format lint test ## Ejecutar pipeline completo de CI
	@echo '$(GREEN)✓ Pipeline completo ejecutado$(RESET)'
