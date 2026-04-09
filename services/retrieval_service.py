from __future__ import annotations

import asyncio
from dataclasses import dataclass

import asyncpg
import numpy as np

from db_pool import get_pool
from services.embedding_service import embed_query
from services.rag_pipeline import _SECTION_LABEL_MAP, _detect_section_focus


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    content: str
    section_name: str
    page_start: int
    chunk_type: str
    vector_score: float
    keyword_score: float
    combined_score: float
    chunk_index: int


def _row_to_chunk(
    row: asyncpg.Record,
    *,
    vector_score: float = 0.0,
    keyword_score: float = 0.0,
    combined_score: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row["id"]),
        paper_id=str(row["paper_id"]),
        content=str(row["content"]),
        section_name=str(row["section_name"] or "Unknown"),
        page_start=int(row["page_start"] or 0),
        chunk_type=str(row["chunk_type"] or "paragraph"),
        vector_score=float(vector_score),
        keyword_score=float(keyword_score),
        combined_score=float(combined_score),
        chunk_index=int(row["chunk_index"] or 0),
    )


async def vector_search(paper_id: str, query_embedding: np.ndarray, top_k: int = 40) -> list[RetrievedChunk]:
    """
    pgvector cosine similarity search scoped to one paper.
    """
    pool = get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                pc.id,
                pc.paper_id,
                pc.content,
                pc.chunk_type,
                pc.page_start,
                pc.chunk_index,
                ps.section_name,
                1 - (pc.embedding <=> $1::vector) AS score
            FROM paper_chunks pc
            LEFT JOIN paper_sections ps ON pc.section_id = ps.id
            WHERE pc.paper_id = $2::uuid
            ORDER BY pc.embedding <=> $1::vector
            LIMIT $3
            """,
            query_embedding.tolist(),
            paper_id,
            top_k,
        )
        return [_row_to_chunk(row, vector_score=float(row["score"] or 0.0)) for row in rows]


async def keyword_search(paper_id: str, question: str, top_k: int = 40) -> list[RetrievedChunk]:
    """
    PostgreSQL full-text search scoped to one paper.
    """
    pool = get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                pc.id,
                pc.paper_id,
                pc.content,
                pc.chunk_type,
                pc.page_start,
                pc.chunk_index,
                ps.section_name,
                ts_rank(
                    to_tsvector('english', pc.content),
                    plainto_tsquery('english', $1)
                ) AS score
            FROM paper_chunks pc
            LEFT JOIN paper_sections ps ON pc.section_id = ps.id
            WHERE pc.paper_id = $2::uuid
              AND to_tsvector('english', pc.content) @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $3
            """,
            question,
            paper_id,
            top_k,
        )
        return [_row_to_chunk(row, keyword_score=float(row["score"] or 0.0)) for row in rows]


async def section_search(
    paper_id: str,
    question: str,
    top_k: int = 40,
) -> list[RetrievedChunk]:
    """
    Retrieve all chunks whose section_name matches the detected section intent.
    This is the primary fix for queries like 'What is related work?' where
    chunk body text does not contain section-specific keywords.
    """
    section_focus = _detect_section_focus(question.lower())
    if section_focus is None:
        return []

    # Build ILIKE patterns from the label map for this focus.
    section_terms = _SECTION_LABEL_MAP.get(section_focus, [section_focus])
    # Convert to SQL ILIKE patterns: 'related work' -> '%related work%'
    patterns = [f"%{term}%" for term in section_terms]

    pool = get_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                pc.id,
                pc.paper_id,
                pc.content,
                pc.chunk_type,
                pc.page_start,
                pc.chunk_index,
                ps.section_name
            FROM paper_chunks pc
            LEFT JOIN paper_sections ps ON pc.section_id = ps.id
            WHERE pc.paper_id = $1::uuid
              AND ps.section_name ILIKE ANY($2::text[])
            ORDER BY pc.chunk_index ASC
            LIMIT $3
            """,
            paper_id,
            patterns,
            top_k,
        )
        # Assign a fixed high score since section match is high-confidence.
        return [
            _row_to_chunk(row, vector_score=0.6, keyword_score=0.6, combined_score=0.6)
            for row in rows
        ]


async def hybrid_retrieve(paper_id: str, question: str, top_k: int = 40) -> list[RetrievedChunk]:
    """
    Run vector and keyword search in parallel and merge their scores.
    """
    query_embedding = embed_query(question)
    vector_results, keyword_results, section_results = await asyncio.gather(
        vector_search(paper_id, query_embedding, top_k=top_k),
        keyword_search(paper_id, question, top_k=top_k),
        section_search(paper_id, question, top_k=top_k),
    )

    merged: dict[str, RetrievedChunk] = {}

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    for chunk in vector_results:
        # Vector-only: use full vector_score — do not penalise for keyword absence.
        chunk.combined_score = chunk.vector_score
        merged[chunk.chunk_id] = chunk

    for chunk in keyword_results:
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            # Keyword-only: use weighted keyword score.
            chunk.combined_score = KEYWORD_WEIGHT * chunk.keyword_score
            merged[chunk.chunk_id] = chunk
            continue

        # Both signals: weighted combination.
        existing.keyword_score = chunk.keyword_score
        existing.combined_score = (
            VECTOR_WEIGHT * existing.vector_score
            + KEYWORD_WEIGHT * existing.keyword_score
        )

    # Merge section_results: if already present, bump score; else insert.
    for chunk in section_results:
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            merged[chunk.chunk_id] = chunk
        else:
            # Existing chunk from vector/keyword: boost its score since it also
            # matches the target section — strong convergent signal.
            existing.combined_score = min(1.0, existing.combined_score + 0.15)

    results = sorted(merged.values(), key=lambda chunk: chunk.combined_score, reverse=True)
    return results[:top_k]
