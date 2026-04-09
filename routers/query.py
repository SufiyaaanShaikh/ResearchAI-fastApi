from __future__ import annotations

from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import json

from config import GROQ_API_KEY
from db import get_db
from db_pool import get_pool
from schemas.paper_schema import PaperQueryRequest, PaperQueryResponse
from services.context_builder import build_context
from services.paper_service import get_paper_status
from services.reranker import rerank_chunks_with_all
from services.retrieval_service import hybrid_retrieve

router = APIRouter(prefix="/papers", tags=["query"])

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_OUTPUT_TOKENS = 2048
MAX_CONTEXT_TOKENS = 5000


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def call_groq_llm(system_prompt: str, context_text: str, question: str) -> str:
    """
    Call Groq API and return the assistant content string.
    """
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_text}\n\nQUESTION: {question}"},
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise exc

    data = response.json()
    if not data.get("choices") or not data["choices"][0].get("message") or "content" not in data["choices"][0]["message"]:
        raise HTTPException(status_code=503, detail="Groq API returned an invalid response.")
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="Groq API returned an invalid response.") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def stream_groq_llm(system_prompt: str, context_text: str, question: str):
    """
    Stream Groq tokens as server-sent events.
    """
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context_text}\n\nQUESTION: {question}"},
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", GROQ_API_URL, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue

                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
    except httpx.HTTPError:
        yield "event: error\ndata: Groq API unavailable after retries.\n\n"

    yield "event: done\ndata: end\n\n"


async def fetch_chat_history(paper_id: str, limit: int = 3) -> list[dict]:
    """
    Return chat history in chronological order.
    """
    pool = get_pool()
    async with pool.acquire() as connection:
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


async def save_chat_history(paper_id: str, question: str, answer: str, chunk_ids: list[str]) -> None:
    """
    Persist the latest question/answer and retrieved chunk ids.
    """
    pool = get_pool()
    async with pool.acquire() as connection:
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


async def fetch_paper_metadata(paper_id: str) -> dict:
    """
    Fetch paper metadata for context construction.
    """
    pool = get_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT title, authors, year, abstract
            FROM papers
            WHERE id = $1::uuid
            """,
            paper_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    return {
        "title": row["title"],
        "authors": list(row["authors"] or []),
        "year": row["year"],
        "abstract": row["abstract"],
    }


async def fetch_neighbor_chunks(paper_id: str, chunk_indices: list[int]):
    """
    Fetch only adjacent neighbor chunks for the provided chunk indices.
    """
    from services.retrieval_service import RetrievedChunk

    normalized_indices = sorted({index for index in chunk_indices if index >= 0})
    if not normalized_indices:
        return []

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
              AND pc.chunk_index = ANY($2::int[])
            ORDER BY pc.chunk_index ASC
            """,
            paper_id,
            normalized_indices,
        )
        return [
            RetrievedChunk(
                chunk_id=str(row["id"]),
                paper_id=str(row["paper_id"]),
                content=str(row["content"]),
                section_name=str(row["section_name"] or "Unknown"),
                page_start=int(row["page_start"] or 0),
                chunk_type=str(row["chunk_type"] or "paragraph"),
                vector_score=0.0,
                keyword_score=0.0,
                combined_score=0.0,
                chunk_index=int(row["chunk_index"] or 0),
            )
            for row in rows
        ]


async def _prepare_query_context(
    payload: PaperQueryRequest,
    db: AsyncSession,
):
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

    # --- Debug logging (stdout only, not returned to caller) ---
    from services.rag_pipeline import _detect_section_focus
    section_focus = _detect_section_focus(payload.question.lower())
    print(f"\n[RAG DEBUG] question='{payload.question}'")
    print(f"[RAG DEBUG] section_focus={section_focus}")
    print(f"[RAG DEBUG] candidates retrieved: {len(candidates)}")
    if candidates:
        top5_sections = [(c.section_name, round(c.combined_score, 3)) for c in candidates[:5]]
        print(f"[RAG DEBUG] top-5 candidate sections: {top5_sections}")

    candidate_indices = [chunk.chunk_index for chunk in candidates]
    neighbor_indices: list[int] = []
    for idx in candidate_indices:
        neighbor_indices.append(idx - 1)
        neighbor_indices.append(idx + 1)

    neighbor_chunks = await fetch_neighbor_chunks(payload.paper_id, neighbor_indices)
    reranked = rerank_chunks_with_all(
        payload.question, candidates, neighbor_chunks, top_n=payload.top_n
    )
    print(f"[RAG DEBUG] reranked chunks: {len(reranked)}")
    if reranked:
        top_reranked = [(r.section_name, round(r.combined_score, 3)) for r in reranked[:5]]
        print(f"[RAG DEBUG] top-5 reranked: {top_reranked}")

    history = await fetch_chat_history(payload.paper_id) if payload.include_history else None
    built = build_context(
        paper_metadata,
        reranked,
        payload.question,
        history,
        max_context_tokens=MAX_CONTEXT_TOKENS,
    )
    print(f"[RAG DEBUG] context chunks included: {len(built.citations)}")
    print(f"[RAG DEBUG] context tokens (approx): {built.total_tokens}")
    print(f"[RAG DEBUG] truncated context tokens: {built.total_tokens}")
    included = [(c["section"], c["page"]) for c in built.citations]
    print(f"[RAG DEBUG] included sections+pages: {included}")

    return reranked, built


@router.post("/query", response_model=PaperQueryResponse)
async def query_paper(
    payload: PaperQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> PaperQueryResponse:
    """
    Full RAG pipeline for one paper.
    """
    reranked, built = await _prepare_query_context(payload, db)
    try:
        answer = await call_groq_llm(built.system_prompt, built.context_text, payload.question)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail="Groq API unavailable after retries."
        ) from exc
    await save_chat_history(payload.paper_id, payload.question, answer, [chunk.chunk_id for chunk in reranked])

    return PaperQueryResponse(
        paper_id=payload.paper_id,
        question=payload.question,
        answer=answer,
        citations=built.citations,
        chunks_used=len(reranked),
        model=GROQ_MODEL,
    )


@router.post("/query/stream")
async def query_paper_stream(
    payload: PaperQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    reranked, built = await _prepare_query_context(payload, db)
    _ = reranked
    return StreamingResponse(
        stream_groq_llm(built.system_prompt, built.context_text, payload.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
