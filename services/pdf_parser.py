from __future__ import annotations

import io
from dataclasses import dataclass

import fitz

from services.pdf_text import _clean_page_text, _extract_page_text_blocks, extract_pdf_pages


@dataclass
class ParsedPage:
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    total_pages: int


def parse_from_url(pdf_url: str) -> ParsedDocument:
    """Calls existing extract_pdf_pages(pdf_url), wraps result in ParsedDocument."""
    pages_tuple = extract_pdf_pages(pdf_url)
    pages = [ParsedPage(page_number=i + 1, text=text) for i, text in enumerate(pages_tuple)]
    return ParsedDocument(pages=pages, total_pages=len(pages))


def parse_from_local(local_path: str) -> ParsedDocument:
    """Same logic as extract_pdf_pages but reads from local file path."""
    document = fitz.open(local_path)
    try:
        raw_pages = [_extract_page_text_blocks(page) for page in document]
    finally:
        document.close()

    parsed_pages: list[ParsedPage] = []
    references_started = False
    for page_number, page_text in enumerate(raw_pages, start=1):
        cleaned, references_started = _clean_page_text(page_text, references_started)
        if cleaned:
            parsed_pages.append(ParsedPage(page_number=page_number, text=cleaned))
        if references_started:
            break

    return ParsedDocument(pages=parsed_pages, total_pages=len(parsed_pages))
