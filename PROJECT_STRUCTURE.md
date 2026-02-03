# DataHub - Estructura del Proyecto

## Visión General de la Arquitectura

Este documento describe la estructura organizacional del proyecto DataHub, diseñada siguiendo principios de Clean Architecture y Domain-Driven Design (DDD).

```
datahub-project/
│
├── app/                          # Código fuente principal
│   ├── __init__.py
│   ├── main.py                   # Entry point de FastAPI
│   │
│   ├── domain/                   # Capa de Dominio (Lógica de Negocio)
│   │   ├── __init__.py
│   │   ├── entities.py           # Entidades y Value Objects
│   │   ├── ports.py              # Interfaces abstractas (Ports)
│   │   └── services/             # Servicios de dominio
│   │       ├── ranking.py        # Algoritmos de ranking (BM25, Semantic)
│   │       ├── filtering.py      # Chain of Responsibility para filtros
│   │       └── validation.py     # Validación de reglas de negocio
│   │
│   ├── application/              # Capa de Aplicación (Use Cases)
│   │   ├── __init__.py
│   │   ├── commands/             # Comandos (CQRS Write Side)
│   │   │   ├── download_dataset.py
│   │   │   └── index_provider.py
│   │   ├── queries/              # Queries (CQRS Read Side)
│   │   │   ├── search_datasets.py
│   │   │   └── get_dataset_details.py
│   │   └── handlers/             # Command/Query Handlers
│   │       ├── download_handler.py
│   │       └── search_handler.py
│   │
│   ├── infrastructure/           # Capa de Infraestructura (Adapters)
│   │   ├── __init__.py
│   │   ├── config.py             # Configuración (Pydantic Settings)
│   │   ├── logging.py            # Structured logging (structlog)
│   │   │
│   │   ├── database/             # PostgreSQL + pgvector
│   │   │   ├── connection.py     # Pool de conexiones
│   │   │   ├── repositories/     # Implementaciones de Repository
│   │   │   │   ├── dataset_repository.py
│   │   │   │   ├── user_repository.py
│   │   │   │   └── download_job_repository.py
│   │   │   └── models.py         # SQLAlchemy models
│   │   │
│   │   ├── cache/                # Redis caching
│   │   │   ├── redis_cache.py
│   │   │   └── multi_level_cache.py
│   │   │
│   │   ├── storage/              # Object storage (MinIO/S3)
│   │   │   ├── minio_storage.py
│   │   │   └── s3_storage.py
│   │   │
│   │   ├── adapters/             # Provider Adapters
│   │   │   ├── huggingface_adapter.py
│   │   │   ├── kaggle_adapter.py
│   │   │   ├── github_adapter.py
│   │   │   └── provider_registry.py
│   │   │
│   │   ├── messaging/            # Event Bus & Celery
│   │   │   ├── event_bus.py
│   │   │   └── celery_tasks.py
│   │   │
│   │   └── security/             # Auth & Security
│   │       ├── jwt.py
│   │       └── rate_limit.py
│   │
│   └── api/                      # Capa de Presentación (FastAPI)
│       ├── __init__.py
│       ├── routes/               # API Routes
│       │   ├── datasets.py
│       │   ├── downloads.py
│       │   ├── users.py
│       │   └── health.py
│       ├── dependencies.py       # FastAPI dependencies
│       ├── middlewares.py        # Custom middlewares
│       └── schemas/              # Pydantic request/response models
│           ├── dataset_schemas.py
│           └── download_schemas.py
│
├── tests/                        # Tests (Pirámide de Testing)
│   ├── __init__.py
│   ├── unit/                     # 70% - Tests unitarios
│   │   ├── domain/
│   │   │   ├── test_entities.py
│   │   │   └── test_ranking.py
│   │   └── application/
│   │       └── test_search_handler.py
│   ├── integration/              # 20% - Tests de integración
│   │   ├── database/
│   │   ├── providers/
│   │   └── api/
│   └── e2e/                      # 10% - Tests end-to-end
│       └── workflows/
│
├── docs/                         # Documentación
│   ├── ARCHITECTURE.md           # Arquitectura detallada
│   ├── API.md                    # API Reference
│   ├── DEPLOYMENT.md             # Guía de deployment
│   └── PLUGINS.md                # Guía de plugins
│
├── scripts/                      # Scripts de utilidad
│   ├── init_database.py          # Inicialización de DB
│   ├── seed_data.py              # Data de ejemplo
│   └── deploy.sh                 # Script de deployment
│
├── config/                       # Configuraciones externas
│   ├── prometheus.yml            # Config de Prometheus
│   └── grafana/                  # Dashboards de Grafana
│       ├── dashboards/
│       └── datasources/
│
├── migrations/                   # Alembic migrations
│   ├── versions/
│   └── alembic.ini
│
├── .github/                      # CI/CD (GitHub Actions)
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── docker-compose.yml            # Orquestación de servicios
├── Dockerfile                    # Multi-stage build
├── requirements.txt              # Dependencias Python
├── requirements-dev.txt          # Dependencias de desarrollo
├── pyproject.toml                # Configuración de proyecto
├── Makefile                      # Comandos de desarrollo
├── .env.example                  # Template de variables de entorno
├── .gitignore                    # Git ignore rules
├── LICENSE                       # Apache 2.0
├── README.md                     # Documentación principal
├── TECHNICAL_PAPER.md            # Paper técnico completo
├── CONTRIBUTING.md               # Guía de contribución
└── PROJECT_STRUCTURE.md          # Este archivo
```

## Principios de Organización

### 1. Separation of Concerns
Cada capa tiene responsabilidades claramente definidas:
- **Domain**: Lógica de negocio pura, sin dependencias externas
- **Application**: Orquestación de casos de uso
- **Infrastructure**: Detalles técnicos y adaptadores
- **API**: Presentación y handling de requests

### 2. Dependency Rule
Las dependencias apuntan siempre hacia adentro:
```
API → Application → Domain
Infrastructure → Domain
```

### 3. Testability
Estructura diseñada para facilitar testing:
- Inyección de dependencias
- Interfaces abstractas (Ports)
- Mocks y stubs fáciles de crear

### 4. Scalability
Organización que permite escalamiento:
- Microservicios potenciales (cada módulo puede ser servicio)
- Separación de read/write (CQRS)
- Eventos para comunicación asíncrona

## Convenciones de Código

### Naming
- **Clases**: `PascalCase`
- **Funciones/Métodos**: `snake_case`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Módulos**: `lowercase` o `snake_case`

### Imports
```python
# Stdlib imports
import asyncio
from typing import List, Optional

# Third-party imports
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from app.domain.entities import DatasetMetadata
from app.domain.ports import DatasetRepository
```

### Type Hints (Obligatorio)
Todos los métodos públicos deben tener type hints completos.

### Docstrings (Google Style)
Todas las clases y funciones públicas deben estar documentadas.

## Flujo de Datos Típico

### Ejemplo: Búsqueda de Datasets

```
1. HTTP Request (POST /api/v1/datasets/search)
   ↓
2. API Router (app/api/routes/datasets.py)
   ↓
3. Search Handler (app/application/handlers/search_handler.py)
   ↓
4. Domain Service (app/domain/services/ranking.py)
   ↓
5. Provider Adapters (app/infrastructure/adapters/)
   ↓
6. External APIs (Hugging Face, Kaggle, etc.)
```

## Herramientas de Desarrollo

### Code Quality
- **black**: Formateo de código
- **ruff**: Linting rápido
- **mypy**: Type checking
- **isort**: Ordenamiento de imports

### Testing
- **pytest**: Framework de testing
- **pytest-asyncio**: Async testing
- **pytest-cov**: Coverage reporting
- **testcontainers**: Integration testing

### Documentation
- **mkdocs**: Generación de documentación
- **mkdocstrings**: API docs desde docstrings

## Comandos Útiles

```bash
# Desarrollo
make install       # Instalar dependencias
make dev           # Setup entorno de desarrollo
make run           # Ejecutar servidor

# Testing
make test          # Ejecutar todos los tests
make test-unit     # Solo tests unitarios
make test-integration  # Solo tests de integración

# Code Quality
make lint          # Linting
make format        # Formatear código

# Database
make migrate       # Ejecutar migraciones
make init-db       # Inicializar base de datos

# Docker
make docker-up     # Levantar servicios
make docker-down   # Detener servicios
```

## Recursos Adicionales

- [Clean Architecture (Robert C. Martin)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design (Eric Evans)](https://domainlanguage.com/ddd/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [PostgreSQL Performance Tips](https://www.postgresql.org/docs/current/performance-tips.html)

---

**Mantenido por**: DataHub Engineering Team
**Última actualización**: Febrero 2026
