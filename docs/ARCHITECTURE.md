# DataHub: Especificación Arquitectónica Técnica
## Documento v1.0 - Diseño de Sistema de Elite

---

## 1. Principios Arquitectónicos Fundamentales

### 1.1 Clean Architecture & Hexagonal Design

El sistema DataHub implementa Clean Architecture (Robert C. Martin) con adaptaciones específicas para sistemas distribuidos de gestión de datos:

**Capas Arquitectónicas:**

```
┌─────────────────────────────────────────────────────────┐
│  Capa de Presentación (Infrastructure Layer)            │
│  • REST API (FastAPI)                                   │
│  • CLI (Typer)                                          │
│  • Web UI (React)                                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Capa de Aplicación (Application Layer)                 │
│  • Use Cases                                            │
│  • Command Handlers                                     │
│  • Query Handlers (CQRS)                                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Capa de Dominio (Domain Layer)                         │
│  • Entidades de Negocio                                 │
│  • Agregados                                            │
│  • Servicios de Dominio                                 │
│  • Especificaciones                                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Capa de Infraestructura (Infrastructure Layer)         │
│  • Repositorios                                         │
│  • Adaptadores de Proveedores                           │
│  • Servicios Externos                                   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Dependency Inversion Principle

**Implementación mediante Puertos y Adaptadores:**

```python
# domain/ports.py - Interfaces abstractas
from abc import ABC, abstractmethod
from typing import List, Optional

class DatasetRepository(ABC):
    """Puerto de salida para persistencia de datasets"""
    
    @abstractmethod
    async def find_by_id(self, dataset_id: str) -> Optional[Dataset]:
        """Buscar dataset por ID"""
        pass
    
    @abstractmethod
    async def save(self, dataset: Dataset) -> Dataset:
        """Persistir dataset"""
        pass
    
    @abstractmethod
    async def search(self, criteria: SearchCriteria) -> List[Dataset]:
        """Búsqueda con criterios complejos"""
        pass

class DatasetProvider(ABC):
    """Puerto de salida para proveedores externos"""
    
    @abstractmethod
    async def search(self, query: str, filters: Dict) -> List[DatasetMetadata]:
        """Búsqueda en proveedor externo"""
        pass
    
    @abstractmethod
    async def download(self, dataset_id: str, options: DownloadOptions) -> DownloadResult:
        """Descarga de dataset"""
        pass

# infrastructure/adapters/huggingface.py - Implementación concreta
class HuggingFaceAdapter(DatasetProvider):
    """Adaptador para Hugging Face Hub"""
    
    def __init__(self, api_client: HfApi, cache: Cache):
        self.client = api_client
        self.cache = cache
    
    async def search(self, query: str, filters: Dict) -> List[DatasetMetadata]:
        # Implementación específica de Hugging Face
        cache_key = f"hf_search:{query}:{hash(frozenset(filters.items()))}"
        
        if cached := await self.cache.get(cache_key):
            return cached
        
        results = await self._execute_search(query, filters)
        await self.cache.set(cache_key, results, ttl=3600)
        
        return results
```

### 1.3 Command Query Responsibility Segregation (CQRS)

**Separación de Comandos y Queries:**

```python
# application/commands.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class DownloadDatasetCommand:
    """Comando para descarga de dataset"""
    dataset_id: str
    provider: str
    user_id: str
    options: DownloadOptions

class CommandHandler(Protocol):
    async def handle(self, command) -> CommandResult:
        """Handler genérico de comandos"""
        ...

class DownloadDatasetHandler:
    """Handler específico para descarga de datasets"""
    
    def __init__(
        self,
        provider_registry: ProviderRegistry,
        download_service: DownloadOrchestrator,
        event_bus: EventBus
    ):
        self.providers = provider_registry
        self.downloader = download_service
        self.events = event_bus
    
    async def handle(self, command: DownloadDatasetCommand) -> CommandResult:
        # 1. Validación
        provider = self.providers.get(command.provider)
        if not provider:
            raise ProviderNotFoundError(command.provider)
        
        # 2. Ejecutar descarga
        result = await self.downloader.download(
            provider=provider,
            dataset_id=command.dataset_id,
            options=command.options
        )
        
        # 3. Emitir evento de dominio
        await self.events.publish(
            DatasetDownloadedEvent(
                dataset_id=command.dataset_id,
                user_id=command.user_id,
                path=result.path,
                timestamp=datetime.utcnow()
            )
        )
        
        return CommandResult(success=True, data=result)

# application/queries.py
@dataclass
class SearchDatasetsQuery:
    """Query para búsqueda de datasets"""
    text: str
    filters: SearchFilters
    limit: int = 20
    offset: int = 0

class SearchDatasetsHandler:
    """Handler para búsqueda de datasets"""
    
    def __init__(
        self,
        repository: DatasetRepository,
        search_engine: SearchEngine
    ):
        self.repository = repository
        self.search = search_engine
    
    async def handle(self, query: SearchDatasetsQuery) -> QueryResult:
        # Usar read model optimizado
        results = await self.search.search(
            text=query.text,
            filters=query.filters,
            limit=query.limit,
            offset=query.offset
        )
        
        return QueryResult(
            items=results,
            total=len(results),
            page=query.offset // query.limit + 1
        )
```

---

## 2. Patrones de Diseño Implementados

### 2.1 Strategy Pattern para Ranking

**Múltiples algoritmos de ranking intercambiables:**

```python
# domain/services/ranking.py
from abc import ABC, abstractmethod

class RankingStrategy(ABC):
    """Estrategia abstracta de ranking"""
    
    @abstractmethod
    def rank(self, results: List[SearchResult], query: str) -> List[RankedResult]:
        """Rankear resultados"""
        pass

class BM25RankingStrategy(RankingStrategy):
    """Ranking basado en BM25 (Okapi BM25)"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        self.idf_cache = {}
    
    def rank(self, results: List[SearchResult], query: str) -> List[RankedResult]:
        query_terms = self._tokenize(query)
        
        scored_results = []
        for result in results:
            score = self._calculate_bm25(result, query_terms)
            scored_results.append(
                RankedResult(result=result, score=score, method="bm25")
            )
        
        return sorted(scored_results, key=lambda x: x.score, reverse=True)
    
    def _calculate_bm25(self, result: SearchResult, query_terms: List[str]) -> float:
        """
        BM25 Score = Σ IDF(qi) * [f(qi, D) * (k1 + 1)] / 
                      [f(qi, D) + k1 * (1 - b + b * |D| / avgdl)]
        """
        score = 0.0
        doc_length = len(result.text.split())
        avg_doc_length = self._get_avg_doc_length()
        
        for term in query_terms:
            if term not in result.text.lower():
                continue
            
            # IDF(qi) = log[(N - n(qi) + 0.5) / (n(qi) + 0.5)]
            idf = self._get_idf(term)
            
            # f(qi, D) = frecuencia del término en el documento
            tf = result.text.lower().count(term)
            
            # Normalización por longitud
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / avg_doc_length)
            )
            
            score += idf * (numerator / denominator)
        
        return score

class SemanticRankingStrategy(RankingStrategy):
    """Ranking basado en embeddings semánticos"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def rank(self, results: List[SearchResult], query: str) -> List[RankedResult]:
        # Generar embedding de query
        query_embedding = self.model.encode(query)
        
        scored_results = []
        for result in results:
            # Similitud coseno
            result_embedding = self.model.encode(result.text)
            similarity = self._cosine_similarity(query_embedding, result_embedding)
            
            scored_results.append(
                RankedResult(result=result, score=similarity, method="semantic")
            )
        
        return sorted(scored_results, key=lambda x: x.score, reverse=True)
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class HybridRankingStrategy(RankingStrategy):
    """Fusión de múltiples estrategias (Reciprocal Rank Fusion)"""
    
    def __init__(self, strategies: List[RankingStrategy], weights: List[float] = None):
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
    
    def rank(self, results: List[SearchResult], query: str) -> List[RankedResult]:
        # Obtener rankings de cada estrategia
        all_rankings = [
            strategy.rank(results, query) 
            for strategy in self.strategies
        ]
        
        # Reciprocal Rank Fusion con k=60
        k = 60
        fused_scores = {}
        
        for weight, ranking in zip(self.weights, all_rankings):
            for rank, ranked_result in enumerate(ranking, start=1):
                result_id = ranked_result.result.id
                rrf_score = weight / (k + rank)
                
                fused_scores[result_id] = fused_scores.get(result_id, 0) + rrf_score
        
        # Crear resultados finales
        final_results = [
            RankedResult(
                result=next(r.result for r in all_rankings[0] if r.result.id == result_id),
                score=score,
                method="hybrid_rrf"
            )
            for result_id, score in fused_scores.items()
        ]
        
        return sorted(final_results, key=lambda x: x.score, reverse=True)
```

### 2.2 Chain of Responsibility para Filtros

```python
# domain/services/filtering.py
class FilterChain:
    """Cadena de filtros aplicados secuencialmente"""
    
    def __init__(self):
        self.filters: List[DatasetFilter] = []
    
    def add_filter(self, filter: 'DatasetFilter') -> 'FilterChain':
        self.filters.append(filter)
        return self
    
    async def apply(self, datasets: List[Dataset]) -> List[Dataset]:
        result = datasets
        for filter in self.filters:
            result = await filter.apply(result)
        return result

class DatasetFilter(ABC):
    """Filtro abstracto"""
    
    @abstractmethod
    async def apply(self, datasets: List[Dataset]) -> List[Dataset]:
        pass

class SizeFilter(DatasetFilter):
    """Filtro por tamaño de dataset"""
    
    def __init__(self, min_size: Optional[int] = None, max_size: Optional[int] = None):
        self.min_size = min_size
        self.max_size = max_size
    
    async def apply(self, datasets: List[Dataset]) -> List[Dataset]:
        result = datasets
        
        if self.min_size is not None:
            result = [d for d in result if d.num_samples >= self.min_size]
        
        if self.max_size is not None:
            result = [d for d in result if d.num_samples <= self.max_size]
        
        return result

class LicenseFilter(DatasetFilter):
    """Filtro por tipo de licencia"""
    
    def __init__(self, allowed_licenses: List[str]):
        self.allowed_licenses = set(allowed_licenses)
    
    async def apply(self, datasets: List[Dataset]) -> List[Dataset]:
        return [
            d for d in datasets 
            if d.license in self.allowed_licenses
        ]

class DomainFilter(DatasetFilter):
    """Filtro por dominio de aplicación"""
    
    def __init__(self, domains: List[str]):
        self.domains = set(domains)
    
    async def apply(self, datasets: List[Dataset]) -> List[Dataset]:
        return [
            d for d in datasets 
            if any(domain in self.domains for domain in d.domains)
        ]

# Uso:
filter_chain = (
    FilterChain()
    .add_filter(SizeFilter(min_size=10000))
    .add_filter(LicenseFilter(["mit", "apache-2.0"]))
    .add_filter(DomainFilter(["nlp", "code"]))
)

filtered_datasets = await filter_chain.apply(raw_datasets)
```

### 2.3 Observer Pattern para Event-Driven Architecture

```python
# domain/events.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List
from abc import ABC

@dataclass
class DomainEvent(ABC):
    """Evento de dominio base"""
    occurred_at: datetime
    aggregate_id: str

@dataclass
class DatasetDownloadedEvent(DomainEvent):
    """Evento: Dataset descargado exitosamente"""
    dataset_id: str
    user_id: str
    provider: str
    file_path: str
    file_size: int

@dataclass
class DatasetDownloadFailedEvent(DomainEvent):
    """Evento: Fallo en descarga de dataset"""
    dataset_id: str
    user_id: str
    error_message: str
    retry_count: int

class EventBus:
    """Bus de eventos para comunicación desacoplada"""
    
    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: type, handler: Callable):
        """Suscribirse a un tipo de evento"""
        self._subscribers[event_type].append(handler)
    
    async def publish(self, event: DomainEvent):
        """Publicar evento a todos los suscriptores"""
        event_type = type(event)
        
        if handlers := self._subscribers.get(event_type):
            tasks = [handler(event) for handler in handlers]
            await asyncio.gather(*tasks)

# Event Handlers
class DatasetDownloadedEventHandler:
    """Handler para evento de descarga completada"""
    
    def __init__(self, analytics_service: AnalyticsService, notification_service: NotificationService):
        self.analytics = analytics_service
        self.notifications = notification_service
    
    async def handle(self, event: DatasetDownloadedEvent):
        # Actualizar analytics
        await self.analytics.record_download(
            dataset_id=event.dataset_id,
            user_id=event.user_id,
            file_size=event.file_size
        )
        
        # Notificar al usuario
        await self.notifications.send(
            user_id=event.user_id,
            message=f"Dataset {event.dataset_id} descargado exitosamente",
            channel="email"
        )

# Configuración
event_bus = EventBus()
download_handler = DatasetDownloadedEventHandler(analytics_service, notification_service)
event_bus.subscribe(DatasetDownloadedEvent, download_handler.handle)
```

---

## 3. Optimizaciones de Performance

### 3.1 Caching Strategy

**Multi-Level Caching:**

```python
# infrastructure/cache/multi_level.py
class MultiLevelCache:
    """Cache multinivel: L1 (in-memory) -> L2 (Redis) -> L3 (DB)"""
    
    def __init__(
        self,
        l1_cache: TTLCache,      # Memoria local
        l2_cache: RedisCache,    # Redis distribuido
        l3_store: Repository     # Base de datos
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.l3 = l3_store
    
    async def get(self, key: str) -> Optional[Any]:
        # L1: Memoria local (más rápido)
        if value := self.l1.get(key):
            return value
        
        # L2: Redis (rápido, distribuido)
        if value := await self.l2.get(key):
            self.l1.set(key, value)  # Promover a L1
            return value
        
        # L3: Base de datos (más lento)
        if value := await self.l3.find_by_key(key):
            await self.l2.set(key, value, ttl=3600)  # Promover a L2
            self.l1.set(key, value)  # Promover a L1
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        # Escribir en todos los niveles
        self.l1.set(key, value)
        await self.l2.set(key, value, ttl=ttl)
        await self.l3.save(key, value)
```

### 3.2 Connection Pooling

```python
# infrastructure/database/pool.py
class DatabaseConnectionPool:
    """Pool de conexiones para PostgreSQL"""
    
    def __init__(
        self,
        dsn: str,
        min_size: int = 10,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0
    ):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            max_queries=self.max_queries,
            max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            command_timeout=60
        )
    
    async def acquire(self):
        return self._pool.acquire()
    
    async def close(self):
        await self._pool.close()
```

### 3.3 Async I/O Optimization

```python
# infrastructure/download/async_downloader.py
class AsyncDownloader:
    """Descargador asíncrono con control de concurrencia"""
    
    def __init__(self, max_concurrent: int = 5, chunk_size: int = 8192):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.chunk_size = chunk_size
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=3600, connect=60)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def download_file(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable] = None
    ) -> DownloadResult:
        async with self.semaphore:
            return await self._download_with_progress(
                url, destination, progress_callback
            )
    
    async def _download_with_progress(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable]
    ) -> DownloadResult:
        async with self.session.get(url) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            async with aio files.open(destination, 'wb') as f:
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback:
                        await progress_callback(downloaded, total_size)
        
        return DownloadResult(
            path=destination,
            size=downloaded,
            success=True
        )
    
    async def download_many(
        self,
        urls: List[str],
        destinations: List[Path]
    ) -> List[DownloadResult]:
        """Descarga múltiple con paralelización controlada"""
        tasks = [
            self.download_file(url, dest)
            for url, dest in zip(urls, destinations)
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 4. Observabilidad y Monitoring

### 4.1 Structured Logging

```python
# infrastructure/logging/structured.py
import structlog

def configure_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False
    )

logger = structlog.get_logger()

# Uso con contexto
logger.info(
    "dataset_download_started",
    dataset_id="deepmind/code_contests",
    user_id="user_123",
    provider="huggingface"
)
```

### 4.2 Metrics con Prometheus

```python
# infrastructure/metrics/prometheus.py
from prometheus_client import Counter, Histogram, Gauge

# Métricas de negocio
dataset_downloads_total = Counter(
    'dataset_downloads_total',
    'Total de descargas de datasets',
    ['provider', 'dataset_id']
)

download_duration_seconds = Histogram(
    'download_duration_seconds',
    'Duración de descargas en segundos',
    ['provider'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
)

active_downloads = Gauge(
    'active_downloads',
    'Número de descargas activas'
)

search_requests_total = Counter(
    'search_requests_total',
    'Total de búsquedas',
    ['ranking_mode']
)

# Instrumentación
class MetricsMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        download_duration_seconds.labels(
            provider=request.state.provider
        ).observe(duration)
        
        return response
```

### 4.3 Distributed Tracing

```python
# infrastructure/tracing/opentelemetry.py
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("download_dataset")
async def download_dataset(dataset_id: str, provider: str):
    span = trace.get_current_span()
    span.set_attribute("dataset.id", dataset_id)
    span.set_attribute("dataset.provider", provider)
    
    try:
        result = await downloader.download(dataset_id)
        span.set_attribute("download.size", result.size)
        span.set_status(trace.Status(trace.StatusCode.OK))
        return result
    except Exception as e:
        span.set_status(trace.Status(trace.StatusCode.ERROR))
        span.record_exception(e)
        raise
```

---

## 5. Seguridad y Compliance

### 5.1 Autenticación JWT

```python
# infrastructure/security/jwt.py
from jose import jwt, JWTError
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except JWTError:
            raise InvalidTokenError()
```

### 5.2 Rate Limiting

```python
# infrastructure/security/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

# En FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/datasets/search")
@limiter.limit("100/minute")
async def search_datasets(request: Request, query: SearchRequest):
    ...
```

---

## 6. Testing Strategy

### 6.1 Pirámide de Testing

```
                 ┌─────────────┐
                 │     E2E     │  ← 10% (Cypress, Playwright)
                 └─────────────┘
              ┌──────────────────┐
              │   Integration    │  ← 20% (Pytest + TestContainers)
              └──────────────────┘
          ┌────────────────────────┐
          │     Unit Tests         │  ← 70% (Pytest + Mocks)
          └────────────────────────┘
```

### 6.2 Ejemplos de Tests

```python
# tests/unit/domain/test_ranking.py
import pytest
from domain.services.ranking import BM25RankingStrategy

class TestBM25Ranking:
    @pytest.fixture
    def strategy(self):
        return BM25RankingStrategy(k1=1.5, b=0.75)
    
    def test_exact_match_scores_highest(self, strategy):
        results = [
            SearchResult(id="1", text="python code generation"),
            SearchResult(id="2", text="machine learning algorithms"),
            SearchResult(id="3", text="code generation python advanced")
        ]
        
        ranked = strategy.rank(results, "python code generation")
        
        assert ranked[0].result.id == "1"
        assert ranked[0].score > ranked[1].score

# tests/integration/test_download_pipeline.py
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="module")
def postgres():
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture(scope="module")
def redis():
    with RedisContainer("redis:7") as redis:
        yield redis

@pytest.mark.integration
async def test_full_download_pipeline(postgres, redis):
    # Setup
    db = await setup_database(postgres.get_connection_url())
    cache = await setup_redis(redis.get_connection_url())
    
    downloader = DownloadOrchestrator(db, cache)
    
    # Execute
    result = await downloader.download(
        provider="huggingface",
        dataset_id="test/dataset",
        options=DownloadOptions()
    )
    
    # Assert
    assert result.success
    assert os.path.exists(result.path)
```

---

Este documento constituye la base arquitectónica para un sistema de gestión de datasets de nivel empresarial, implementando patrones probados y prácticas de ingeniería de elite.
