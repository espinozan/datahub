# DataHub - Production Dockerfile
# Multi-stage build para optimizar tamaño de imagen

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Instalar dependencias del sistema para compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Crear virtual environment y instalar dependencias
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="DataHub Engineering <engineering@datahub.ai>"
LABEL version="1.0.0"
LABEL description="DataHub - Sistema Unificado de Gestión de Datasets para IA"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app

# Instalar dependencias runtime
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r datahub && useradd -r -g datahub datahub

# Copiar virtual environment desde builder
COPY --from=builder /opt/venv /opt/venv

# Configurar directorio de trabajo
WORKDIR $APP_HOME

# Copiar código de aplicación
COPY --chown=datahub:datahub . .

# Cambiar a usuario no-root
USER datahub

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Exponer puerto
EXPOSE 8000

# Comando por defecto (puede ser sobreescrito)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
