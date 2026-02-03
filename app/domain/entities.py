"""
DataHub - Domain Entities
Entidades de dominio siguiendo Domain-Driven Design (DDD)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4


class DomainTag(str, Enum):
    """Tags de dominio para categorización de datasets"""
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    GRAPH = "graph"
    CODE = "code"
    MULTIMODAL = "multimodal"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class TaskType(str, Enum):
    """Tipos de tareas de Machine Learning"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    NAMED_ENTITY_RECOGNITION = "ner"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    SPEECH_RECOGNITION = "speech_recognition"
    TEXT_TO_SPEECH = "text_to_speech"
    CODE_GENERATION = "code_generation"


class Modality(str, Enum):
    """Modalidades de datos"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    CODE = "code"


class LicenseType(str, Enum):
    """Tipos de licencias comunes"""
    MIT = "mit"
    APACHE_2_0 = "apache-2.0"
    GPL_3_0 = "gpl-3.0"
    BSD_3_CLAUSE = "bsd-3-clause"
    CC_BY_4_0 = "cc-by-4.0"
    CC_BY_SA_4_0 = "cc-by-sa-4.0"
    CC_BY_NC_4_0 = "cc-by-nc-4.0"
    PROPRIETARY = "proprietary"
    OTHER = "other"


class ProviderType(str, Enum):
    """Proveedores de datasets soportados"""
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    GITHUB = "github"
    GOOGLE_DRIVE = "google_drive"
    PAPERS_WITH_CODE = "papers_with_code"
    CUSTOM = "custom"


@dataclass
class DatasetMetadata:
    """
    Metadata completa de un dataset
    Agregado raíz en el dominio
    """
    id: UUID = field(default_factory=uuid4)
    provider: ProviderType
    provider_id: str  # ID en el sistema del proveedor
    name: str
    description: Optional[str] = None
    
    # Categorización
    domains: List[DomainTag] = field(default_factory=list)
    tasks: List[TaskType] = field(default_factory=list)
    modalities: List[Modality] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)  # ISO 639-1 codes
    
    # Licencia y legal
    license: Optional[LicenseType] = None
    license_url: Optional[str] = None
    citation: Optional[str] = None
    
    # Estadísticas
    num_samples: Optional[int] = None
    file_size: Optional[int] = None  # bytes
    downloads_count: int = 0
    likes_count: int = 0
    
    # Estructura
    splits: List[str] = field(default_factory=list)  # train, validation, test
    features: Dict[str, Any] = field(default_factory=dict)
    
    # URLs
    url: Optional[str] = None
    paper_url: Optional[str] = None
    github_url: Optional[str] = None
    
    # Metadata temporal
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: Optional[datetime] = None
    
    # Checksum para integridad
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        
        if not self.provider_id:
            raise ValueError("Provider ID cannot be empty")
    
    def update_metadata(self, **kwargs):
        """Actualizar metadata manteniendo auditoría"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.utcnow()
    
    def add_download(self):
        """Incrementar contador de descargas"""
        self.downloads_count += 1
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización a diccionario"""
        return {
            "id": str(self.id),
            "provider": self.provider.value,
            "provider_id": self.provider_id,
            "name": self.name,
            "description": self.description,
            "domains": [d.value for d in self.domains],
            "tasks": [t.value for t in self.tasks],
            "modalities": [m.value for m in self.modalities],
            "languages": self.languages,
            "license": self.license.value if self.license else None,
            "num_samples": self.num_samples,
            "file_size": self.file_size,
            "downloads_count": self.downloads_count,
            "url": self.url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class SearchCriteria:
    """
    Value Object para criterios de búsqueda
    Inmutable y validado
    """
    text: str
    domains: List[DomainTag] = field(default_factory=list)
    tasks: List[TaskType] = field(default_factory=list)
    modalities: List[Modality] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    licenses: List[LicenseType] = field(default_factory=list)
    
    # Filtros numéricos
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None
    min_file_size: Optional[int] = None
    max_file_size: Optional[int] = None
    min_downloads: Optional[int] = None
    
    # Filtros booleanos
    has_splits: Optional[bool] = None
    has_paper: Optional[bool] = None
    
    # Filtros temporales
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Paginación
    limit: int = 20
    offset: int = 0
    
    def __post_init__(self):
        """Validación de criterios"""
        if self.limit <= 0 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        
        if self.offset < 0:
            raise ValueError("Offset cannot be negative")
        
        if self.min_samples is not None and self.min_samples < 0:
            raise ValueError("min_samples cannot be negative")
        
        if self.max_samples is not None and self.min_samples is not None:
            if self.max_samples < self.min_samples:
                raise ValueError("max_samples cannot be less than min_samples")


class DownloadStatus(str, Enum):
    """Estados de descarga"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadJob:
    """
    Entidad que representa un trabajo de descarga
    Agregado raíz con ciclo de vida propio
    """
    id: UUID = field(default_factory=uuid4)
    dataset_id: UUID
    user_id: UUID
    provider: ProviderType
    
    status: DownloadStatus = DownloadStatus.PENDING
    
    # Configuración de descarga
    splits: List[str] = field(default_factory=list)
    output_format: str = "parquet"
    output_path: Optional[str] = None
    
    # Progreso
    total_bytes: int = 0
    downloaded_bytes: int = 0
    progress_percentage: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def start(self, total_bytes: int):
        """Iniciar descarga"""
        self.status = DownloadStatus.DOWNLOADING
        self.total_bytes = total_bytes
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_progress(self, downloaded_bytes: int):
        """Actualizar progreso"""
        self.downloaded_bytes = downloaded_bytes
        
        if self.total_bytes > 0:
            self.progress_percentage = (downloaded_bytes / self.total_bytes) * 100
        
        self.updated_at = datetime.utcnow()
        
        # Estimar tiempo de completado
        if self.started_at and self.progress_percentage > 0:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            total_estimated = elapsed / (self.progress_percentage / 100)
            remaining = total_estimated - elapsed
            self.estimated_completion = datetime.utcnow() + timedelta(seconds=remaining)
    
    def complete(self, output_path: str):
        """Marcar como completado"""
        self.status = DownloadStatus.COMPLETED
        self.output_path = output_path
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.progress_percentage = 100.0
    
    def fail(self, error_message: str):
        """Marcar como fallido"""
        self.status = DownloadStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
    
    def can_retry(self) -> bool:
        """Verificar si se puede reintentar"""
        return (
            self.status == DownloadStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def retry(self):
        """Reintentar descarga"""
        if not self.can_retry():
            raise ValueError("Cannot retry this download job")
        
        self.retry_count += 1
        self.status = DownloadStatus.PENDING
        self.error_message = None
        self.updated_at = datetime.utcnow()


@dataclass
class RankedResult:
    """
    Value Object para resultados rankeados
    """
    dataset: DatasetMetadata
    score: float
    ranking_method: str  # bm25, semantic, hybrid
    explanation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validación"""
        if not 0 <= self.score <= 1:
            # Normalizar score si está fuera de rango
            self.score = max(0.0, min(1.0, self.score))


@dataclass
class SearchResults:
    """
    Agregado de resultados de búsqueda
    """
    results: List[RankedResult]
    total_count: int
    query: SearchCriteria
    execution_time_ms: float
    
    # Metadata de búsqueda
    providers_searched: List[ProviderType] = field(default_factory=list)
    ranking_method: str = "hybrid"
    
    @property
    def total_pages(self) -> int:
        """Calcular total de páginas"""
        return (self.total_count + self.query.limit - 1) // self.query.limit
    
    @property
    def current_page(self) -> int:
        """Página actual"""
        return (self.query.offset // self.query.limit) + 1
    
    @property
    def has_next_page(self) -> bool:
        """Verificar si hay siguiente página"""
        return self.current_page < self.total_pages
    
    @property
    def has_previous_page(self) -> bool:
        """Verificar si hay página anterior"""
        return self.current_page > 1


@dataclass
class User:
    """
    Entidad de usuario
    """
    id: UUID = field(default_factory=uuid4)
    email: str
    username: str
    hashed_password: str
    
    # Preferencias
    preferred_providers: List[ProviderType] = field(default_factory=list)
    api_keys: Dict[str, str] = field(default_factory=dict)  # provider -> api_key
    
    # Quota management
    daily_download_quota: int = 100
    downloads_today: int = 0
    
    # Metadata
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    def can_download(self) -> bool:
        """Verificar si puede descargar"""
        return (
            self.is_active and
            self.downloads_today < self.daily_download_quota
        )
    
    def increment_downloads(self):
        """Incrementar contador de descargas"""
        if not self.can_download():
            raise ValueError("Download quota exceeded")
        
        self.downloads_today += 1
    
    def reset_daily_quota(self):
        """Reset quota diaria"""
        self.downloads_today = 0
