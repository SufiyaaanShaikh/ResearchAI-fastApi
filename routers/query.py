from __future__ import annotations

import os
from uuid import UUID

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from schemas.paper_schema import PaperQueryRequest, PaperQueryResponse
from services.context_builder import build_context
from services.paper_service import get_paper_status
from services.reranker import rerank_chunks_with_all
from services.retrieval_service import hybrid_retrieve

load_dotenv()

router = APIRouter(prefix="/papers", tags=["query"])

DEFAULT_DATABASE_URL = "postgresql+asyncpg://user:admin123@localhost:5432/researchai"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


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
    return await asyncpg.connect(_get_asyncpg_database_url())


async def call_groq_llm(system_prompt: str, context_text: str, question: str) -> str:
    """
    Call Groq API and return the assistant content string.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured.")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_text}\n\nQUESTION: {question}"},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Groq API request failed.") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Groq API returned an invalid response.") from exc


async def fetch_chat_history(paper_id: str, limit: int = 3) -> list[dict]:
    """
    Return chat history in chronological order.
    """
    connection = await _connect()
    try:
        rows = await connection.fetch(
            """
            SELECT question, answer
            FROM chat_history
            WHERE paper_id = $1::uuid
            ORDER BY created_at DESC
            LIMIT $2
            """,
            paper_id,
            limit,
        )
        return [{"question": row["question"], "answer": row["answer"]} for row in reversed(rows)]
    finally:
        await connection.close()


async def save_chat_history(paper_id: str, question: str, answer: str, chunk_ids: list[str]) -> None:
    """
    Persist the latest question/answer and retrieved chunk ids.
    """
    connection = await _connect()
    try:
        await connection.execute(
            """
            INSERT INTO chat_history (paper_id, question, answer, retrieved_chunk_ids)
            VALUES ($1::uuid, $2, $3, $4::uuid[])
            """,
            paper_id,
            question,
            answer,
            [UUID(chunk_id) for chunk_id in chunk_ids],
        )
    finally:
        await connection.close()


async def fetch_paper_metadata(paper_id: str) -> dict:
    """
    Fetch paper metadata for context construction.
    """
    connection = await _connect()
    try:
        row = await connection.fetchrow(
            """
            SELECT title, authors, year, abstract
            FROM papers
            WHERE id = $1::uuid
            """,
            paper_id,
        )
    finally:
        await connection.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    return {
        "title": row["title"],
        "authors": list(row["authors"] or []),
        "year": row["year"],
        "abstract": row["abstract"],
    }


@router.post("/query", response_model=PaperQueryResponse)
async def query_paper(
    payload: PaperQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> PaperQueryResponse:
    """
    Full RAG pipeline for one paper.
    """
    paper_metadata = await fetch_paper_metadata(payload.paper_id)
    paper_status = await get_paper_status(db, payload.paper_id)
    if paper_status is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper_status["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Paper is still being processed. Status: {paper_status['status']}",
        )

    candidates = await hybrid_retrieve(payload.paper_id, payload.question, top_k=payload.top_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="No content found for this paper.")

    reranked = rerank_chunks_with_all(payload.question, candidates, candidates, top_n=payload.top_n)
    history = await fetch_chat_history(payload.paper_id) if payload.include_history else None
    built = build_context(paper_metadata, reranked, payload.question, history)
    answer = await call_groq_llm(built.system_prompt, built.context_text, payload.question)
    await save_chat_history(payload.paper_id, payload.question, answer, [chunk.chunk_id for chunk in reranked])

    return PaperQueryResponse(
        paper_id=payload.paper_id,
        question=payload.question,
        answer=answer,
        citations=built.citations,
        chunks_used=len(reranked),
        model=GROQ_MODEL,
    )
