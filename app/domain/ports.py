"""
DataHub - Domain Ports
Interfaces abstractas siguiendo Hexagonal Architecture (Ports & Adapters Pattern)
Los puertos definen contratos que la capa de infraestructura debe implementar
"""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator, Dict, Any
from pathlib import Path

from app.domain.entities import (
    DatasetMetadata,
    SearchCriteria,
    SearchResults,
    DownloadJob,
    ProviderType,
    User
)


# ============================================================================
# PORTS DE SALIDA (Output Ports) - Infraestructura
# ============================================================================

class DatasetRepository(ABC):
    """
    Puerto de repositorio para persistencia de datasets
    Implementa el patrón Repository del DDD
    """
    
    @abstractmethod
    async def find_by_id(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Buscar dataset por ID único"""
        pass
    
    @abstractmethod
    async def find_by_provider_id(
        self,
        provider: ProviderType,
        provider_id: str
    ) -> Optional[DatasetMetadata]:
        """Buscar dataset por ID del proveedor"""
        pass
    
    @abstractmethod
    async def save(self, dataset: DatasetMetadata) -> DatasetMetadata:
        """Persistir dataset (insert o update)"""
        pass
    
    @abstractmethod
    async def delete(self, dataset_id: str) -> bool:
        """Eliminar dataset"""
        pass
    
    @abstractmethod
    async def search(self, criteria: SearchCriteria) -> List[DatasetMetadata]:
        """Búsqueda con criterios complejos"""
        pass
    
    @abstractmethod
    async def search_full_text(
        self,
        query: str,
        limit: int = 50
    ) -> List[DatasetMetadata]:
        """Búsqueda full-text en PostgreSQL"""
        pass
    
    @abstractmethod
    async def list_popular(
        self,
        limit: int = 20,
        provider: Optional[ProviderType] = None
    ) -> List[DatasetMetadata]:
        """Listar datasets más populares"""
        pass
    
    @abstractmethod
    async def count(self, criteria: Optional[SearchCriteria] = None) -> int:
        """Contar datasets que cumplen criterios"""
        pass


class DatasetProvider(ABC):
    """
    Puerto para proveedores externos de datasets
    Cada adaptador (HuggingFace, Kaggle, etc.) implementa esta interfaz
    """
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Tipo de proveedor"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[DatasetMetadata]:
        """Búsqueda en el proveedor externo"""
        pass
    
    @abstractmethod
    async def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Obtener metadata completa de un dataset"""
        pass
    
    @abstractmethod
    async def download(
        self,
        dataset_id: str,
        output_path: Path,
        splits: Optional[List[str]] = None,
        format: str = "parquet"
    ) -> Path:
        """
        Descargar dataset
        
        Returns:
            Path al dataset descargado
        """
        pass
    
    @abstractmethod
    async def list_recent(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[DatasetMetadata]:
        """Listar datasets recientes para indexación"""
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validar credenciales del proveedor"""
        pass


class CacheService(ABC):
    """Puerto para servicio de caché (Redis, Memcached, etc.)"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor de caché"""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Guardar en caché con TTL opcional"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar entrada de caché"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verificar si existe una key"""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Eliminar todas las keys que coincidan con patrón"""
        pass


class StorageBackend(ABC):
    """
    Puerto para almacenamiento de datasets
    Puede ser implementado con S3, MinIO, sistema de archivos local, etc.
    """
    
    @abstractmethod
    async def save_file(
        self,
        file_path: Path,
        destination: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Guardar archivo en storage
        
        Returns:
            URL o path del archivo guardado
        """
        pass
    
    @abstractmethod
    async def get_file(self, file_key: str, destination: Path) -> Path:
        """Descargar archivo del storage"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_key: str) -> bool:
        """Eliminar archivo del storage"""
        pass
    
    @abstractmethod
    async def file_exists(self, file_key: str) -> bool:
        """Verificar si existe un archivo"""
        pass
    
    @abstractmethod
    async def generate_presigned_url(
        self,
        file_key: str,
        expiration: int = 3600
    ) -> str:
        """Generar URL pre-firmada para descarga"""
        pass


class EmbeddingService(ABC):
    """
    Puerto para servicio de embeddings semánticos
    Usado para búsqueda semántica y ranking
    """
    
    @abstractmethod
    async def encode_text(self, text: str) -> List[float]:
        """Generar embedding de texto"""
        pass
    
    @abstractmethod
    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generar embeddings de múltiples textos"""
        pass
    
    @abstractmethod
    async def similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calcular similitud entre embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Obtener dimensión de los embeddings"""
        pass


class EventBus(ABC):
    """
    Puerto para bus de eventos (Event-Driven Architecture)
    Implementa patrón Publisher-Subscriber
    """
    
    @abstractmethod
    async def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publicar evento"""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: callable
    ):
        """Suscribirse a un tipo de evento"""
        pass
    
    @abstractmethod
    async def unsubscribe(
        self,
        event_type: str,
        handler: callable
    ):
        """Desuscribirse de un evento"""
        pass


class DownloadJobRepository(ABC):
    """Repositorio para trabajos de descarga"""
    
    @abstractmethod
    async def create(self, job: DownloadJob) -> DownloadJob:
        """Crear nuevo trabajo de descarga"""
        pass
    
    @abstractmethod
    async def update(self, job: DownloadJob) -> DownloadJob:
        """Actualizar trabajo existente"""
        pass
    
    @abstractmethod
    async def find_by_id(self, job_id: str) -> Optional[DownloadJob]:
        """Buscar trabajo por ID"""
        pass
    
    @abstractmethod
    async def find_by_user(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[DownloadJob]:
        """Listar trabajos de un usuario"""
        pass
    
    @abstractmethod
    async def find_pending(self, limit: int = 10) -> List[DownloadJob]:
        """Encontrar trabajos pendientes para procesamiento"""
        pass
    
    @abstractmethod
    async def delete_old_completed(self, days: int = 7) -> int:
        """Eliminar trabajos completados antiguos"""
        pass


class UserRepository(ABC):
    """Repositorio para usuarios"""
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """Crear nuevo usuario"""
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """Actualizar usuario"""
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: str) -> Optional[User]:
        """Buscar usuario por ID"""
        pass
    
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        """Buscar usuario por email"""
        pass
    
    @abstractmethod
    async def find_by_username(self, username: str) -> Optional[User]:
        """Buscar usuario por username"""
        pass
    
    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """Eliminar usuario"""
        pass


class NotificationService(ABC):
    """Puerto para servicio de notificaciones"""
    
    @abstractmethod
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False
    ) -> bool:
        """Enviar email"""
        pass
    
    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Enviar webhook notification"""
        pass


class MetricsCollector(ABC):
    """
    Puerto para colección de métricas (Prometheus, Datadog, etc.)
    """
    
    @abstractmethod
    def increment_counter(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Incrementar contador"""
        pass
    
    @abstractmethod
    def record_histogram(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Registrar valor en histograma"""
        pass
    
    @abstractmethod
    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Establecer gauge"""
        pass


# ============================================================================
# PORTS DE ENTRADA (Input Ports) - Casos de Uso
# ============================================================================

class SearchDatasetsUseCase(ABC):
    """Caso de uso: Búsqueda de datasets"""
    
    @abstractmethod
    async def execute(self, criteria: SearchCriteria) -> SearchResults:
        """Ejecutar búsqueda de datasets"""
        pass


class DownloadDatasetUseCase(ABC):
    """Caso de uso: Descarga de dataset"""
    
    @abstractmethod
    async def execute(
        self,
        dataset_id: str,
        user_id: str,
        provider: ProviderType,
        splits: Optional[List[str]] = None,
        format: str = "parquet"
    ) -> DownloadJob:
        """Iniciar descarga de dataset"""
        pass


class GetDatasetDetailsUseCase(ABC):
    """Caso de uso: Obtener detalles de dataset"""
    
    @abstractmethod
    async def execute(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Obtener detalles completos de un dataset"""
        pass


class ListPopularDatasetsUseCase(ABC):
    """Caso de uso: Listar datasets populares"""
    
    @abstractmethod
    async def execute(
        self,
        limit: int = 20,
        provider: Optional[ProviderType] = None
    ) -> List[DatasetMetadata]:
        """Listar datasets más descargados/populares"""
        pass


class MonitorDownloadProgressUseCase(ABC):
    """Caso de uso: Monitorear progreso de descarga"""
    
    @abstractmethod
    async def execute(self, job_id: str) -> AsyncIterator[DownloadJob]:
        """Stream de actualizaciones de progreso"""
        pass


# ============================================================================
# FACTORY PATTERN PARA PROVIDERS
# ============================================================================

class ProviderFactory(ABC):
    """
    Factory para crear instancias de proveedores
    Implementa Abstract Factory Pattern
    """
    
    @abstractmethod
    def create_provider(
        self,
        provider_type: ProviderType,
        config: Dict[str, Any]
    ) -> DatasetProvider:
        """Crear instancia de proveedor configurado"""
        pass
    
    @abstractmethod
    def get_supported_providers(self) -> List[ProviderType]:
        """Obtener lista de proveedores soportados"""
        pass


# ============================================================================
# SPECIFICATION PATTERN PARA QUERIES COMPLEJAS
# ============================================================================

class Specification(ABC):
    """
    Patrón Specification para queries complejas
    Permite componer criterios de búsqueda de manera declarativa
    """
    
    @abstractmethod
    def is_satisfied_by(self, dataset: DatasetMetadata) -> bool:
        """Verificar si dataset satisface la especificación"""
        pass
    
    def and_(self, other: 'Specification') -> 'Specification':
        """Combinación AND"""
        return AndSpecification(self, other)
    
    def or_(self, other: 'Specification') -> 'Specification':
        """Combinación OR"""
        return OrSpecification(self, other)
    
    def not_(self) -> 'Specification':
        """Negación"""
        return NotSpecification(self)


class AndSpecification(Specification):
    """Especificación AND"""
    
    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, dataset: DatasetMetadata) -> bool:
        return (
            self.spec1.is_satisfied_by(dataset) and
            self.spec2.is_satisfied_by(dataset)
        )


class OrSpecification(Specification):
    """Especificación OR"""
    
    def __init__(self, spec1: Specification, spec2: Specification):
        self.spec1 = spec1
        self.spec2 = spec2
    
    def is_satisfied_by(self, dataset: DatasetMetadata) -> bool:
        return (
            self.spec1.is_satisfied_by(dataset) or
            self.spec2.is_satisfied_by(dataset)
        )


class NotSpecification(Specification):
    """Especificación NOT"""
    
    def __init__(self, spec: Specification):
        self.spec = spec
    
    def is_satisfied_by(self, dataset: DatasetMetadata) -> bool:
        return not self.spec.is_satisfied_by(dataset)
