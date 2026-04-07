from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


async def _get_asyncpg_connection(db: AsyncSession):
    connection = await db.connection()
    raw_connection = await connection.get_raw_connection()
    return raw_connection.driver_connection


async def create_paper_record(
    db: AsyncSession,
    title: str,
    authors: list[str] | None,
    year: int | None,
    abstract: str | None,
    source_type: str,
    arxiv_id: str | None = None,
    pdf_url: str | None = None,
    local_pdf_path: str | None = None,
    *,
    paper_id: str | None = None,
) -> dict[str, Any]:
    connection = await _get_asyncpg_connection(db)
    try:
        row = await connection.fetchrow(
            """
            INSERT INTO papers (
                id,
                title,
                authors,
                year,
                abstract,
                source_type,
                arxiv_id,
                pdf_url,
                local_pdf_path
            )
            VALUES (
                COALESCE($1::uuid, gen_random_uuid()),
                $2,
                $3::text[],
                $4,
                $5,
                $6,
                $7,
                $8,
                $9
            )
            RETURNING
                id,
                title,
                authors,
                year,
                abstract,
                source_type,
                arxiv_id,
                pdf_url,
                local_pdf_path,
                total_pages,
                status,
                created_at,
                updated_at
            """,
            paper_id,
            title,
            authors,
            year,
            abstract,
            source_type,
            arxiv_id,
            pdf_url,
            local_pdf_path,
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    return dict(row) if row else {}


async def get_paper_status(db: AsyncSession, paper_id: str) -> dict[str, Any] | None:
    connection = await _get_asyncpg_connection(db)
    row = await connection.fetchrow(
        """
        SELECT id, title, status, created_at
        FROM papers
        WHERE id = $1
        """,
        paper_id,
    )
    return dict(row) if row else None


async def update_paper_status(db: AsyncSession, paper_id: str, status: str) -> None:
    connection = await _get_asyncpg_connection(db)
    try:
        await connection.execute(
            """
            UPDATE papers
            SET status = $1, updated_at = NOW()
            WHERE id = $2
            """,
            status,
            paper_id,
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
