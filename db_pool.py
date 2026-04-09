from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from config import DATABASE_URL

_pool: asyncpg.Pool | None = None


def _get_asyncpg_database_url() -> str:
    if DATABASE_URL.startswith("postgresql+asyncpg://"):
        return DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
    if DATABASE_URL.startswith("postgresql+psycopg2://"):
        return DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://", 1)
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL
    if DATABASE_URL.startswith("postgres://"):
        return DATABASE_URL.replace("postgres://", "postgresql://", 1)
    return DATABASE_URL


async def init_pool() -> None:
    global _pool
    if _pool is not None:
        return

    _pool = await asyncpg.create_pool(
        dsn=_get_asyncpg_database_url(),
        min_size=2,
        max_size=10,
        init=register_vector,
    )


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool
