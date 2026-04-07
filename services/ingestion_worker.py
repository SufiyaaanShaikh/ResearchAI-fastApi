from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path
from uuid import uuid4

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

from models.embedding_model import get_embedding
from services.chunker import chunk_document
from services.pdf_parser import parse_from_local, parse_from_url
from services.section_detector import detect_sections

load_dotenv()

DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:admin123@localhost:5432/researchai"
BATCH_SIZE = 32


def _get_asyncpg_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


async def _connect() -> asyncpg.Connection:
    connection = await asyncpg.connect(_get_asyncpg_database_url())
    await register_vector(connection)
    return connection


async def _set_status(connection: asyncpg.Connection, paper_id: str, status: str) -> None:
    await connection.execute(
        """
        UPDATE papers
        SET status = $1, updated_at = NOW()
        WHERE id = $2
        """,
        status,
        paper_id,
    )


async def _mark_failed(paper_id: str) -> None:
    connection = await _connect()
    try:
        await _set_status(connection, paper_id, "failed")
    finally:
        await connection.close()


async def run_ingestion(paper_id: str) -> None:
    """
    Full ingestion pipeline. Catches all exceptions and marks paper as 'failed' on error.
    """
    connection: asyncpg.Connection | None = None
    try:
        connection = await _connect()

        await _set_status(connection, paper_id, "downloading")
        paper_row = await connection.fetchrow(
            """
            SELECT id, source_type, pdf_url, local_pdf_path
            FROM papers
            WHERE id = $1
            """,
            paper_id,
        )
        if paper_row is None:
            raise ValueError(f"Paper not found: {paper_id}")

        pdf_url = paper_row["pdf_url"]
        local_pdf_path = paper_row["local_pdf_path"]

        await _set_status(connection, paper_id, "parsing")
        if local_pdf_path and Path(local_pdf_path).exists():
            parsed = await asyncio.to_thread(parse_from_local, local_pdf_path)
        elif pdf_url:
            parsed = await asyncio.to_thread(parse_from_url, pdf_url)
        else:
            raise ValueError(f"No PDF source available for paper {paper_id}")

        await connection.execute(
            """
            UPDATE papers
            SET total_pages = $1, updated_at = NOW()
            WHERE id = $2
            """,
            parsed.total_pages,
            paper_id,
        )

        await _set_status(connection, paper_id, "chunking")
        sections = detect_sections(parsed.pages)
        chunks = chunk_document(parsed.pages)

        section_id_map: dict[str, str] = {}
        if sections:
            section_rows = []
            for section in sections:
                section_id = str(uuid4())
                section_id_map[section.section_name] = section_id
                section_rows.append(
                    (
                        section_id,
                        paper_id,
                        section.section_name,
                        section.page_start,
                        section.page_end,
                        section.section_order,
                    )
                )

            await connection.executemany(
                """
                INSERT INTO paper_sections (
                    id,
                    paper_id,
                    section_name,
                    page_start,
                    page_end,
                    section_order
                )
                VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
                """,
                section_rows,
            )

        await _set_status(connection, paper_id, "embedding")
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            chunk_rows = []
            for chunk in batch:
                embedding = await asyncio.to_thread(get_embedding, chunk.content)
                chunk_rows.append(
                    (
                        str(uuid4()),
                        paper_id,
                        section_id_map.get(chunk.section_name),
                        chunk.chunk_index,
                        chunk.chunk_type,
                        chunk.page_start,
                        chunk.page_start,
                        chunk.content,
                        chunk.token_count,
                        embedding.tolist(),
                    )
                )

            await connection.executemany(
                """
                INSERT INTO paper_chunks (
                    id,
                    paper_id,
                    section_id,
                    chunk_index,
                    chunk_type,
                    page_start,
                    page_end,
                    content,
                    token_count,
                    embedding
                )
                VALUES (
                    $1::uuid,
                    $2::uuid,
                    $3::uuid,
                    $4,
                    $5,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10::vector
                )
                """,
                chunk_rows,
            )

        await _set_status(connection, paper_id, "ready")
    except Exception:
        traceback.print_exc()
        try:
            if connection is not None and not connection.is_closed():
                await _set_status(connection, paper_id, "failed")
            else:
                await _mark_failed(paper_id)
        except Exception:
            traceback.print_exc()
    finally:
        if connection is not None and not connection.is_closed():
            await connection.close()


async def trigger_ingestion(paper_id: str) -> None:
    """Entry point called by routers. Wraps run_ingestion with error guard."""
    try:
        await run_ingestion(paper_id)
    except Exception as e:
        print(f"[INGESTION ERROR] paper_id={paper_id}: {e}")
