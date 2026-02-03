# Guía de Contribución - DataHub

## Bienvenido, Ingeniero

Esta guía establece los estándares de ingeniería que hacen de DataHub un sistema de clase mundial. Como contribuyente, eres parte de una comunidad comprometida con la excelencia técnica.

## Filosofía de Desarrollo

### Principios Fundamentales

1. **Clean Code First**: El código es comunicación entre ingenieros
2. **Architecture Matters**: Las decisiones arquitectónicas tienen consecuencias a largo plazo
3. **Testing is Non-Negotiable**: La confianza se construye con tests
4. **Documentation is Code**: Si no está documentado, no existe
5. **Performance by Design**: La optimización no es un pensamiento tardío

## Proceso de Contribución

### 1. Fork & Clone

```bash
git clone https://github.com/your-username/datahub.git
cd datahub
git remote add upstream https://github.com/datahub-org/datahub.git
```

### 2. Branch Strategy

```bash
# Feature branches
git checkout -b feature/semantic-search-optimization

# Bug fixes
git checkout -b fix/download-retry-mechanism

# Documentation
git checkout -b docs/api-reference-update
```

### 3. Development Setup

```bash
# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Instalar pre-commit hooks
pre-commit install

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# Ejecutar migraciones
alembic upgrade head

# Iniciar servicios con Docker
docker-compose up -d postgres redis minio
```

### 4. Escribir Código

#### 4.1 Estándares de Código

**Formateadores y Linters:**
```bash
# Black - Code formatter
black app/ tests/

# Ruff - Fast Python linter
ruff check app/ tests/ --fix

# isort - Import sorting
isort app/ tests/

# mypy - Type checking
mypy app/
```

**Convenciones de Naming:**
```python
# Clases: PascalCase
class DatasetRepository:
    pass

# Funciones y métodos: snake_case
async def download_dataset(dataset_id: str) -> DownloadJob:
    pass

# Constantes: UPPER_SNAKE_CASE
MAX_CONCURRENT_DOWNLOADS = 5

# Variables privadas: _leading_underscore
class MyClass:
    def __init__(self):
        self._internal_state = {}
```

#### 4.2 Type Hints (Obligatorio)

```python
from typing import Optional, List, Dict, Any
from datetime import datetime

async def search_datasets(
    query: str,
    filters: Dict[str, Any],
    limit: int = 20
) -> List[DatasetMetadata]:
    """
    Buscar datasets con filtros.
    
    Args:
        query: Texto de búsqueda
        filters: Filtros adicionales
        limit: Máximo número de resultados
        
    Returns:
        Lista de datasets que coinciden con los criterios
        
    Raises:
        ValueError: Si query está vacío
        ProviderError: Si la búsqueda externa falla
    """
    pass
```

#### 4.3 Docstrings (Google Style)

```python
class DownloadOrchestrator:
    """
    Orquestador de descargas con gestión de concurrencia.
    
    El orquestador coordina descargas paralelas de múltiples proveedores,
    implementando retry automático, validación de integridad y 
    rate limiting.
    
    Attributes:
        max_concurrent: Máximo número de descargas simultáneas
        storage: Backend de almacenamiento para datasets
        metrics: Colector de métricas de performance
        
    Example:
        >>> orchestrator = DownloadOrchestrator(max_concurrent=5)
        >>> job = await orchestrator.download(
        ...     provider="huggingface",
        ...     dataset_id="deepmind/code_contests"
        ... )
    """
    
    def __init__(
        self,
        max_concurrent: int = 5,
        storage: StorageBackend,
        metrics: MetricsCollector
    ):
        """
        Inicializar orquestador.
        
        Args:
            max_concurrent: Número de descargas concurrentes
            storage: Backend de almacenamiento
            metrics: Servicio de métricas
        """
        pass
```

### 5. Testing

#### 5.1 Estructura de Tests

```
tests/
├── unit/                  # Tests unitarios (70% del total)
│   ├── domain/
│   ├── application/
│   └── infrastructure/
├── integration/           # Tests de integración (20%)
│   ├── database/
│   ├── providers/
│   └── api/
└── e2e/                   # Tests end-to-end (10%)
    └── workflows/
```

#### 5.2 Escribir Tests

```python
import pytest
from unittest.mock import AsyncMock, Mock

class TestDownloadOrchestrator:
    """Tests para el orquestador de descargas"""
    
    @pytest.fixture
    def orchestrator(self):
        """Fixture que crea una instancia de orquestador"""
        storage = Mock(spec=StorageBackend)
        metrics = Mock(spec=MetricsCollector)
        return DownloadOrchestrator(
            max_concurrent=2,
            storage=storage,
            metrics=metrics
        )
    
    @pytest.mark.asyncio
    async def test_download_success(self, orchestrator):
        """Debe descargar dataset exitosamente"""
        # Arrange
        provider = AsyncMock(spec=DatasetProvider)
        provider.download.return_value = Path("/tmp/dataset")
        
        # Act
        result = await orchestrator.download(
            provider=provider,
            dataset_id="test/dataset"
        )
        
        # Assert
        assert result.success is True
        assert result.path.exists()
        provider.download.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_retry_on_failure(self, orchestrator):
        """Debe reintentar descarga en caso de fallo"""
        # Arrange
        provider = AsyncMock(spec=DatasetProvider)
        provider.download.side_effect = [
            NetworkError("Connection timeout"),
            NetworkError("Connection timeout"),
            Path("/tmp/dataset")
        ]
        
        # Act
        result = await orchestrator.download(
            provider=provider,
            dataset_id="test/dataset"
        )
        
        # Assert
        assert result.success is True
        assert provider.download.call_count == 3
```

#### 5.3 Coverage Requirements

```bash
# Ejecutar tests con coverage
pytest --cov=app --cov-report=html --cov-report=term

# Coverage mínimo requerido
# - Overall: 80%
# - Domain layer: 90%
# - Application layer: 85%
# - Infrastructure layer: 75%
```

### 6. Commit Guidelines

#### 6.1 Conventional Commits

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Nueva feature
- `fix`: Bug fix
- `docs`: Cambios en documentación
- `style`: Formatting, código que no afecta lógica
- `refactor`: Refactoring sin cambio de comportamiento
- `perf`: Mejora de performance
- `test`: Agregar o corregir tests
- `chore`: Mantenimiento, tooling

**Examples:**
```bash
feat(search): implement hybrid ranking algorithm

Added BM25 + semantic embedding fusion for improved
search relevance. Includes configurable weights and
fallback to lexical search.

Closes #123

---

fix(download): handle network timeout gracefully

Implemented exponential backoff with jitter for
retry mechanism. Max retries configurable via env var.

Fixes #456

---

perf(database): optimize full-text search queries

Added GIN index on search_vector column. Query time
reduced from 500ms to 50ms for typical searches.

Benchmarks in tests/performance/test_search_perf.py
```

### 7. Pull Request

#### 7.1 PR Template

```markdown
## Descripción

Breve descripción de los cambios realizados.

## Tipo de Cambio

- [ ] Bug fix (cambio no breaking que soluciona un issue)
- [ ] Nueva feature (cambio no breaking que agrega funcionalidad)
- [ ] Breaking change (cambio que causa que funcionalidad existente no funcione)
- [ ] Mejora de documentación

## ¿Cómo se ha testeado?

Describir los tests que ejecutaste para verificar tus cambios.

- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Manual testing

## Checklist

- [ ] Mi código sigue los estándares de estilo del proyecto
- [ ] He realizado una auto-revisión de mi código
- [ ] He comentado mi código, particularmente en áreas complejas
- [ ] He actualizado la documentación correspondiente
- [ ] Mis cambios no generan nuevos warnings
- [ ] He agregado tests que prueban mi fix/feature
- [ ] Tests unitarios nuevos y existentes pasan localmente
- [ ] Coverage está por encima del mínimo requerido
```

#### 7.2 Review Process

1. **Automated Checks**: CI/CD ejecuta tests, linters, type checking
2. **Code Review**: Al menos 2 approvals de maintainers
3. **Architecture Review**: Para cambios arquitectónicos significativos
4. **Performance Review**: Para cambios que afecten performance

### 8. Architecture Decision Records (ADRs)

Para decisiones arquitectónicas significativas, crear un ADR:

```markdown
# ADR-001: Implementación de Multi-Level Caching

## Status
Accepted

## Context
Las búsquedas semánticas son computacionalmente costosas (200-500ms).
Necesitamos reducir latencia para mejor UX.

## Decision
Implementar cache multinivel: L1 (in-memory) -> L2 (Redis) -> L3 (DB)

## Consequences
### Positive
- Latencia reducida de 500ms a 50ms para queries populares
- Reducción del 60% en carga de base de datos
- Escalabilidad horizontal mejorada

### Negative
- Mayor complejidad en invalidación de caché
- Overhead de memoria adicional
- Posible inconsistencia temporal entre nodos

## Implementation
Ver `infrastructure/cache/multi_level.py`
```

## Áreas de Contribución

### High Priority
- 🔍 Mejoras en algoritmos de ranking
- ⚡ Optimizaciones de performance
- 📊 Dashboard de analytics
- 🔌 Nuevos proveedores de datasets
- 🧪 Aumento de test coverage

### Medium Priority
- 📝 Mejoras de documentación
- 🎨 UI/UX improvements
- 🌐 Internacionalización
- 🔐 Mejoras de seguridad

### Good First Issues
Busca issues etiquetados con `good-first-issue` para empezar.

## Recursos

- [Arquitectura del Sistema](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Discord Community](https://discord.gg/datahub)

## Código de Conducta

Nos regimos por el [Contributor Covenant](CODE_OF_CONDUCT.md). 
Se espera respeto, profesionalismo y colaboración constructiva.

---

**¡Gracias por contribuir a DataHub!** 🚀

Juntos estamos construyendo la infraestructura de próxima generación
para la investigación en Inteligencia Artificial.
