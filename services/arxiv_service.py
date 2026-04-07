from __future__ import annotations

import html
import re
from pathlib import Path

import httpx
from fastapi import HTTPException

ARXIV_BASE_URL = "https://export.arxiv.org/abs"
ARXIV_PDF_URL = "https://arxiv.org/pdf"
REQUEST_HEADERS = {"User-Agent": "ResearchAI/1.0"}


def _extract_meta_values(page: str, name: str) -> list[str]:
    pattern = re.compile(
        rf'<meta\s+name="{re.escape(name)}"\s+content="([^"]*)"',
        re.IGNORECASE,
    )
    return [html.unescape(value).strip() for value in pattern.findall(page) if value.strip()]


def _extract_first_meta_value(page: str, name: str) -> str | None:
    values = _extract_meta_values(page, name)
    return values[0] if values else None


def _extract_abstract(page: str) -> str | None:
    meta_abstract = _extract_first_meta_value(page, "citation_abstract")
    if meta_abstract:
        return meta_abstract

    blockquote_match = re.search(
        r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
        page,
        re.IGNORECASE | re.DOTALL,
    )
    if not blockquote_match:
        return None

    abstract_text = re.sub(r"<[^>]+>", " ", blockquote_match.group(1))
    return " ".join(html.unescape(abstract_text).split())


def _extract_year(page: str) -> int | None:
    citation_date = _extract_first_meta_value(page, "citation_date")
    if citation_date:
        year_match = re.search(r"(19|20)\d{2}", citation_date)
        if year_match:
            return int(year_match.group(0))

    submitted_match = re.search(
        r"Submitted on\s+\d+\s+\w+\s+((?:19|20)\d{2})",
        page,
        re.IGNORECASE,
    )
    if submitted_match:
        return int(submitted_match.group(1))

    return None


async def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    url = f"{ARXIV_BASE_URL}/{arxiv_id}"
    try:
        async with httpx.AsyncClient(headers=REQUEST_HEADERS, follow_redirects=True, timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=404, detail="Failed to fetch arXiv metadata.") from exc

    page = response.text
    title = _extract_first_meta_value(page, "citation_title")
    authors = _extract_meta_values(page, "citation_author")
    year = _extract_year(page)
    abstract = _extract_abstract(page)

    if not title or not authors or year is None or not abstract:
        raise HTTPException(status_code=404, detail="Failed to parse arXiv metadata.")

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "pdf_url": f"{ARXIV_PDF_URL}/{arxiv_id}.pdf",
    }


async def download_pdf(pdf_url: str, save_path: str) -> str:
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(headers=REQUEST_HEADERS, follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", pdf_url) as response:
                response.raise_for_status()
                with destination.open("wb") as file_obj:
                    async for chunk in response.aiter_bytes():
                        file_obj.write(chunk)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=404, detail="Failed to download PDF.") from exc

    return str(destination)
