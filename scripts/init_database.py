#!/usr/bin/env python3
"""
DataHub - Database Initialization Script
Inicializa la base de datos con schema y data esencial
"""

import asyncio
import asyncpg
from app.infrastructure.config import get_settings

settings = get_settings()


async def create_database():
    """Crear base de datos si no existe"""
    print("🔧 Verificando base de datos...")
    
    # Conectar a postgres default database
    conn = await asyncpg.connect(
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432,
        database="postgres"
    )
    
    # Verificar si existe
    exists = await conn.fetchval(
        "SELECT 1 FROM pg_database WHERE datname = 'datahub'"
    )
    
    if not exists:
        print("📦 Creando base de datos 'datahub'...")
        await conn.execute("CREATE DATABASE datahub")
        print("✅ Base de datos creada")
    else:
        print("✅ Base de datos ya existe")
    
    await conn.close()


async def create_extensions():
    """Crear extensiones necesarias"""
    print("\n🔌 Instalando extensiones de PostgreSQL...")
    
    conn = await asyncpg.connect(settings.DATABASE_URL)
    
    # pgvector para embeddings
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("  ✓ pgvector instalado")
    
    # UUID generation
    await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
    print("  ✓ uuid-ossp instalado")
    
    # Full-text search
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    print("  ✓ pg_trgm instalado")
    
    await conn.close()


async def main():
    print("=" * 60)
    print("DataHub - Inicialización de Base de Datos")
    print("=" * 60)
    
    try:
        await create_database()
        await create_extensions()
        
        print("\n🎉 Inicialización completada exitosamente!")
        print("\nPróximos pasos:")
        print("  1. Ejecutar migraciones: alembic upgrade head")
        print("  2. Cargar datos de ejemplo: python scripts/seed_data.py")
        
    except Exception as e:
        print(f"\n❌ Error durante inicialización: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
