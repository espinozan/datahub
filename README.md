## En Desarrollo no esta listo para prod. 

# DataHub: Sistema Unificado de Gestión de Datasets para Investigación en IA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Visión General

**DataHub** es una plataforma de código abierto diseñada específicamente para científicos e investigadores en Inteligencia Artificial que necesitan descubrir, filtrar, descargar y gestionar datasets de manera eficiente. El sistema unifica múltiples fuentes de datos bajo una arquitectura REST API basada en FastAPI.

### Características Principales

- 🔍 **Búsqueda Unificada**: Integración con Hugging Face, Kaggle, GitHub y Google Drive
- 🧠 **Búsqueda Semántica**: Ranking híbrido con BM25 y embeddings
- ⚡ **Descarga Paralela**: Orquestación eficiente con retry automático
- 📊 **Analytics Avanzados**: Métricas de calidad y distribución de datos
- 🔌 **Sistema de Plugins**: Arquitectura extensible para nuevos proveedores
- 🎨 **Interface Web**: React + TypeScript con UI moderna
- 📈 **Observabilidad**: Monitoring completo con Prometheus y Grafana

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     CAPA DE PRESENTACIÓN                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Web UI      │  │  REST API    │  │  CLI Tool    │      │
│  │  (React)     │  │  (FastAPI)   │  │  (Typer)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE LÓGICA DE NEGOCIO                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         Dataset Discovery & Management Engine         │   │
│  │  • Semantic Search    • Filter Pipeline              │   │
│  │  • Metadata Indexing  • Download Orchestrator        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   CAPA DE ADAPTADORES (Plugins)              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ HuggingFace │ │   Kaggle    │ │   GitHub    │          │
│  │  Adapter    │ │   Adapter   │ │   Adapter   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PERSISTENCIA                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  PostgreSQL  │  │   Redis      │  │  MinIO/S3    │      │
│  │  (Metadata)  │  │  (Cache)     │  │  (Storage)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisitos

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (opcional)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/espinozan/datahub.git
cd datahub

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Ejecutar migraciones
alembic upgrade head

# Iniciar servicios
docker-compose up -d postgres redis minio

# Iniciar servidor
uvicorn app.main:app --reload
```

### Uso Rápido

```python
from datahub import DataHub

# Inicializar cliente
hub = DataHub(api_key="your_api_key")

# Búsqueda de datasets
results = hub.search(
    query="code generation python",
    filters={
        "domain": ["code", "nlp"],
        "size_min": 10000,
        "license": ["mit", "apache-2.0"]
    }
)

# Descargar dataset
job = hub.download(
    dataset_id="deepmind/code_contests",
    provider="huggingface",
    splits=["train", "validation"]
)

# Monitorear progreso
for update in job.progress():
    print(f"Progress: {update.percent}%")
```

## 📚 Documentación Completa

- [Guía de Arquitectura](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Guía de Desarrollo](docs/DEVELOPMENT.md)
- [Sistema de Plugins](docs/PLUGINS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con coverage
pytest --cov=app --cov-report=html

# Tests de integración
pytest tests/integration/

# Tests de performance
pytest tests/performance/ --benchmark
```

## 🤝 Contribución

Contribuciones son bienvenidas! Por favor lee nuestra [Guía de Contribución](CONTRIBUTING.md) para detalles sobre nuestro código de conducta y el proceso para enviar pull requests.

## 📄 Licencia

Este proyecto está licenciado bajo Apache License 2.0 - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Hugging Face por su excelente ecosistema de datasets
- FastAPI por el framework web de alto rendimiento
- La comunidad de investigación en IA

## 📞 Contacto

- Email: contact@datahub.ai
- Discord: [DataHub Community](https://discord.gg/datahub)
- Twitter: [@DataHubAI](https://twitter.com/DataHubAI)

---

**Desarrollado con ❤️ por el equipo de Ainsophic - DataHub Engineering**
