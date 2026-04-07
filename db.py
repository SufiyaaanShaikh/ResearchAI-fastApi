from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/researchai"


def _to_async_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


def _to_sync_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg2://", 1)
    return database_url


DATABASE_URL = _to_async_database_url(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
SYNC_DATABASE_URL = _to_sync_database_url(DATABASE_URL)

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# This engine is intended for one-off synchronous setup tasks such as init_db.py.
sync_engine = create_engine(SYNC_DATABASE_URL, pool_pre_ping=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


__all__ = [
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "sync_engine",
]
