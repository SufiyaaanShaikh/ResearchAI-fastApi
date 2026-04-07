from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from services.arxiv_service import download_pdf, fetch_arxiv_metadata
from services.ingestion_worker import trigger_ingestion
from services.paper_service import create_paper_record, get_paper_status


router = APIRouter(prefix="/papers", tags=["papers"])

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "papers"


class AddPaperRequest(BaseModel):
    arxiv_id: str | None = None
    pdf_url: str | None = None


def _paper_file_path(paper_id: str) -> Path:
    return STORAGE_DIR / f"{paper_id}.pdf"


def _parse_authors(authors: str | None) -> list[str] | None:
    if not authors:
        return None

    parsed_authors = [author.strip() for author in authors.split(",") if author.strip()]
    return parsed_authors or None


def _title_from_pdf_url(pdf_url: str) -> str:
    parsed = urlparse(pdf_url)
    filename = Path(unquote(parsed.path)).stem
    cleaned = filename.replace("-", " ").replace("_", " ").strip()
    return cleaned or "Imported PDF"


async def _save_upload(file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)


async def _get_asyncpg_connection(db: AsyncSession):
    connection = await db.connection()
    raw_connection = await connection.get_raw_connection()
    return raw_connection.driver_connection


@router.post("/upload")
async def upload_paper(
    title: str = Form(...),
    authors: str | None = Form(None),
    year: int | None = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    paper_id = str(uuid4())
    file_path = _paper_file_path(paper_id)

    try:
        await _save_upload(file, file_path)
    finally:
        await file.close()

    record = await create_paper_record(
        db,
        title=title,
        authors=_parse_authors(authors),
        year=year,
        abstract=None,
        source_type="upload",
        pdf_url=None,
        local_pdf_path=str(file_path),
        paper_id=paper_id,
    )

    asyncio.create_task(trigger_ingestion(paper_id))

    return {
        "paper_id": str(record["id"]),
        "title": record["title"],
        "status": record["status"],
        "message": "Ingestion started",
    }


@router.post("/add")
async def add_paper(
    payload: AddPaperRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    if not payload.arxiv_id and not payload.pdf_url:
        raise HTTPException(status_code=400, detail="Provide at least arxiv_id or pdf_url.")

    title = _title_from_pdf_url(payload.pdf_url) if payload.pdf_url else "Imported PDF"
    authors = None
    year = None
    abstract = None
    pdf_url = payload.pdf_url

    if payload.arxiv_id:
        metadata = await fetch_arxiv_metadata(payload.arxiv_id)
        title = metadata["title"]
        authors = metadata["authors"]
        year = metadata["year"]
        abstract = metadata["abstract"]
        pdf_url = metadata["pdf_url"]

    if not pdf_url:
        raise HTTPException(status_code=400, detail="Unable to determine PDF URL.")

    paper_id = str(uuid4())
    file_path = _paper_file_path(paper_id)
    await download_pdf(pdf_url, str(file_path))

    record = await create_paper_record(
        db,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        source_type="arxiv",
        arxiv_id=payload.arxiv_id,
        pdf_url=pdf_url,
        local_pdf_path=str(file_path),
        paper_id=paper_id,
    )

    asyncio.create_task(trigger_ingestion(paper_id))

    return {
        "paper_id": str(record["id"]),
        "title": record["title"],
        "status": record["status"],
        "message": "Ingestion started",
    }


@router.get("/status/{paper_id}")
async def paper_status(
    paper_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    paper = await get_paper_status(db, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return {
        "paper_id": str(paper["id"]),
        "title": paper["title"],
        "status": paper["status"],
        "created_at": paper["created_at"],
    }


@router.get("/list")
async def list_papers(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    connection = await _get_asyncpg_connection(db)
    rows = await connection.fetch(
        """
        SELECT id, title, status, source_type, year, created_at
        FROM papers
        ORDER BY created_at DESC
        LIMIT 50
        """
    )
    return [dict(row) for row in rows]
