from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("researchai")


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


_raw_database_url = os.environ.get("DATABASE_URL")
if not _raw_database_url:
    raise RuntimeError("DATABASE_URL is not set")

_raw_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _raw_groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

DATABASE_URL = _normalize_database_url(_raw_database_url)
GROQ_API_KEY = _raw_groq_api_key

logger.info(
    f"GROQ_API_KEY loaded: {'YES (len=' + str(len(GROQ_API_KEY)) + ')' if GROQ_API_KEY else 'NO - THIS WILL FAIL'}"
)
