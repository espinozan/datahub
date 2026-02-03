"""
DataHub - FastAPI Application Entry Point
Sistema Unificado de Gestión de Datasets para Investigación en IA

Arquitectura: Clean Architecture + Hexagonal (Ports & Adapters)
Framework: FastAPI 0.109+
Python: 3.11+
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.infrastructure.config import get_settings
from app.infrastructure.database import DatabaseManager
from app.infrastructure.logging import configure_logging
from app.api.routes import api_router


# Configurar logging estructurado
configure_logging()
logger = structlog.get_logger()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifecycle manager para FastAPI
    Gestiona inicialización y limpieza de recursos
    """
    logger.info(
        "application_startup",
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION
    )
    
    # Inicializar conexión a base de datos
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Inicializar caché Redis
    # await cache_manager.initialize()
    
    # Inicializar proveedores de datasets
    # await provider_registry.initialize_all()
    
    logger.info("application_ready", message="All systems operational")
    
    yield  # Aplicación en ejecución
    
    # Cleanup
    logger.info("application_shutdown", message="Graceful shutdown initiated")
    
    await db_manager.close()
    # await cache_manager.close()
    
    logger.info("application_shutdown_complete")


# ============================================================================
# FastAPI Application Factory
# ============================================================================
def create_application() -> FastAPI:
    """
    Factory pattern para crear instancia de FastAPI configurada
    Permite múltiples configuraciones (testing, staging, production)
    """
    
    app = FastAPI(
        title="DataHub API",
        description="""
        **DataHub**: Sistema Unificado de Gestión de Datasets para Investigación en IA
        
        ## Características
        
        * 🔍 **Búsqueda Unificada**: Integración con múltiples proveedores de datasets
        * 🧠 **Búsqueda Semántica**: Ranking híbrido con BM25 + embeddings
        * ⚡ **Descarga Eficiente**: Orquestación paralela con retry automático
        * 📊 **Analytics**: Métricas detalladas de calidad y distribución
        * 🔌 **Extensible**: Sistema de plugins para nuevos proveedores
        
        ## Proveedores Soportados
        
        - Hugging Face Hub
        - Kaggle Datasets
        - GitHub Repositories
        - Google Drive
        
        ## Autenticación
        
        Utiliza JWT (JSON Web Tokens) para autenticación.
        Incluir header: `Authorization: Bearer <token>`
        """,
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # ========================================================================
    # Middleware Configuration
    # ========================================================================
    
    # CORS - Cross Origin Resource Sharing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip compression para responses grandes
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request ID middleware para tracing
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        
        # Bind request_id al contexto de logging
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    # ========================================================================
    # Exception Handlers
    # ========================================================================
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handler global para excepciones no capturadas"""
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request.headers.get("X-Request-ID")
            }
        )
    
    # ========================================================================
    # Routes
    # ========================================================================
    
    # Health check endpoint (sin autenticación)
    @app.get("/health", tags=["System"])
    async def health_check():
        """
        Health check endpoint para load balancers y monitoring
        """
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        }
    
    # API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # ========================================================================
    # Prometheus Metrics
    # ========================================================================
    
    if settings.PROMETHEUS_ENABLED:
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="http_requests_inprogress",
            inprogress_labels=True,
        )
        
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    logger.info(
        "application_configured",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG
    )
    
    return app


# ============================================================================
# Application Instance
# ============================================================================
app = create_application()


# ============================================================================
# Development Server (no usar en producción)
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
