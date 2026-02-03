"""
DataHub - Sistema Unificado de Gestión de Datasets para IA
Version: 1.0.0
License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "DataHub Engineering Team"
__email__ = "engineering@datahub.ai"

from app.domain.entities import (
    DatasetMetadata,
    SearchCriteria,
    DownloadJob,
    DomainTag,
    TaskType,
    Modality,
    LicenseType,
    ProviderType,
)

__all__ = [
    "__version__",
    "DatasetMetadata",
    "SearchCriteria",
    "DownloadJob",
    "DomainTag",
    "TaskType",
    "Modality",
    "LicenseType",
    "ProviderType",
]
