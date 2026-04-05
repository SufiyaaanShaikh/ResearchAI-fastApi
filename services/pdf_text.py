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
_TABLE_SPLIT = re.compile(r"\t+|\s{2,}")

# FIX 6 (from previous session): Only skip SHORT standalone figure caption
# lines, not every line that mentions "figure".
_MAX_CAPTION_LINE_LENGTH = 120


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


def _looks_like_table_data_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    separators = re.findall(r"\t+|\s{2,}", stripped)
    if len(separators) < 2:
        return False

    columns = [part.strip() for part in _TABLE_SPLIT.split(stripped) if part.strip()]
    if len(columns) < 3:
        return False

    numeric_columns = sum(1 for column in columns if re.search(r"\d", column))
    return numeric_columns >= 2


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

        # Keep full captions because they often contain the experiment takeaway.
        # Only drop pure labels like "Fig. 1" with no descriptive body.
        if _FIGURE_CAPTION.match(line) and len(line) < 15:
            continue

        normalized_line = _MULTISPACE.sub(" ", line)
        if _FIGURE_CAPTION.match(line) and len(line) < _MAX_CAPTION_LINE_LENGTH:
            normalized_line = f"[FIGURE] {normalized_line}"
        kept_lines.append(normalized_line)

    cleaned = "\n".join(kept_lines)
    cleaned = _MULTIBLANKLINES.sub("\n\n", cleaned).strip()
    return cleaned, False


# ---------------------------------------------------------------------------
# FIX 5: Block-aware page extraction
# ---------------------------------------------------------------------------
#
# The original code called page.get_text("text") which concatenates all text
# spans in the order PyMuPDF encounters them internally.  For single-column
# papers this is fine.  For two-column IEEE / ACM papers it is wrong:
# PyMuPDF returns left-column text interleaved with right-column text because
# the internal span order follows PDF object IDs, not visual reading order.
#
# The fix uses page.get_text("blocks", sort=True):
#   "blocks"  — returns one entry per text block (paragraph / table row / caption)
#               rather than one giant string, giving us structural boundaries.
#   sort=True — re-sorts blocks by their top-left corner (y then x), which
#               matches visual reading order: top-to-bottom, left col before right.
#
# Each block entry is a tuple:
#   (x0, y0, x1, y1, text, block_no, block_type)
# block_type == 0 = text block; block_type == 1 = image block (skip).
#
# We insert explicit blank lines between blocks so that the downstream chunker
# correctly detects paragraph boundaries and standalone table-header lines.

def _extract_page_text_blocks(page: fitz.Page) -> str:
    blocks = page.get_text("blocks", sort=True)   # sort=True = reading order
    lines: list[str] = []
    for block in blocks:
        # block_type: 0 = text, 1 = image — skip images
        if block[6] != 0:
            continue
        raw: str = block[4]
        stripped = raw.strip()
        if stripped:
            block_lines: list[str] = []
            for block_line in stripped.splitlines():
                cleaned_line = block_line.strip()
                if not cleaned_line:
                    continue
                if _looks_like_table_data_line(cleaned_line):
                    block_lines.append(f"[TABLE DATA] {cleaned_line}")
                else:
                    block_lines.append(cleaned_line)
            if block_lines:
                lines.append("\n".join(block_lines))
            lines.append("")   # blank line = paragraph boundary for the chunker
    return "\n".join(lines)


@lru_cache(maxsize=64)
def extract_pdf_pages(pdf_url: str) -> tuple[str, ...]:
    with urlopen(pdf_url) as response:
        pdf_bytes = response.read()

    document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        # FIX 5: Use block-aware extraction instead of plain get_text("text")
        raw_pages = [_extract_page_text_blocks(page) for page in document]
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


@lru_cache(maxsize=64)
def extract_pdf_section(pdf_url: str, section_name: str) -> str:
    """Extract all text from chunks belonging to a specific section."""
    pages = extract_pdf_pages(pdf_url)
    full_text = "\n\n".join(pages)
    pattern = re.compile(
        rf"^(?:[IVX]{{1,6}}\.\s+)?{re.escape(section_name)}\b",
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(full_text)
    if not match:
        return ""
    start = match.start()
    next_section = re.compile(
        r"^(?:[IVX]{1,6}\.\s+)?[A-Z][A-Za-z\s\-]{2,60}$",
        re.MULTILINE
    )
    remaining = full_text[start + len(match.group()):]
    next_match = next_section.search(remaining)
    end = start + len(match.group()) + (next_match.start() if next_match else len(remaining))
    return full_text[start:end].strip()
