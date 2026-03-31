from __future__ import annotations

import io
from functools import lru_cache
import re
from urllib.request import urlopen

import fitz


_REFERENCES_HEADER = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_FIGURE_CAPTION = re.compile(r"^\s*(fig\.?|figure)\b", re.IGNORECASE)
_TABLE_WORD = re.compile(r"\btable\b", re.IGNORECASE)
_NUMERIC_TOKEN = re.compile(r"^[\d.,%+\-]+$")
_MULTISPACE = re.compile(r"\s{2,}")
_MULTIBLANKLINES = re.compile(r"\n{3,}")


def _is_table_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if _TABLE_WORD.search(stripped):
        return True

    tokens = stripped.split()
    if len(tokens) < 4:
        return False

    numeric_tokens = sum(1 for token in tokens if _NUMERIC_TOKEN.match(token))
    numeric_ratio = numeric_tokens / max(len(tokens), 1)
    many_columns = stripped.count("  ") >= 3
    return numeric_ratio >= 0.5 or (numeric_ratio >= 0.35 and many_columns)


def _clean_page_text(page_text: str, references_started: bool) -> tuple[str, bool]:
    if references_started:
        return "", True

    kept_lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            kept_lines.append("")
            continue

        if _REFERENCES_HEADER.match(line):
            return "", True

        if _FIGURE_CAPTION.match(line):
            continue

        normalized_line = _MULTISPACE.sub(" ", line)
        kept_lines.append(normalized_line)

    cleaned = "\n".join(kept_lines)
    cleaned = _MULTIBLANKLINES.sub("\n\n", cleaned).strip()
    return cleaned, False


@lru_cache(maxsize=64)
def extract_pdf_pages(pdf_url: str) -> tuple[str, ...]:
    with urlopen(pdf_url) as response:
        pdf_bytes = response.read()

    document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        raw_pages = [page.get_text("text") for page in document]
    finally:
        document.close()

    cleaned_pages: list[str] = []
    references_started = False
    for page_text in raw_pages:
        cleaned, references_started = _clean_page_text(page_text, references_started)
        if cleaned:
            cleaned_pages.append(cleaned)
        if references_started:
            break

    return tuple(cleaned_pages)


@lru_cache(maxsize=64)
def extract_pdf_text(pdf_url: str) -> str:
    pages = extract_pdf_pages(pdf_url)
    return "\n\n".join(pages).strip()
