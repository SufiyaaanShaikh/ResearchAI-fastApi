from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from services.rag_pipeline import _semantic_chunk_pages


@dataclass
class ChunkRecord:
    chunk_index: int
    section_name: str
    chunk_type: str
    page_start: int
    content: str
    token_count: int


def _infer_chunk_type(content: str) -> str:
    if content.startswith("[TABLE DATA]"):
        return "table"
    if content.startswith("[FIGURE]"):
        return "figure_caption"
    return "paragraph"


def _chunk_index_from_meta(chunk: dict[str, Any], fallback_index: int) -> int:
    chunk_id = str(chunk.get("chunk_id", ""))
    match = re.search(r"(\d+)$", chunk_id)
    if match:
        return int(match.group(1))
    return fallback_index


def chunk_document(pages: list[Any]) -> list[ChunkRecord]:
    """
    Call existing _semantic_chunk_pages([p.text for p in pages]).
    Convert the returned list[ChunkMeta] dicts into ChunkRecord dataclass objects.
    Infer chunk_type from content prefix:
      starts with "[TABLE DATA]" -> "table"
      starts with "[FIGURE]" -> "figure_caption"
      else -> "paragraph"
    Count tokens as: len(content.split())
    """
    chunk_meta = _semantic_chunk_pages([page.text for page in pages])

    return [
        ChunkRecord(
            chunk_index=_chunk_index_from_meta(chunk, fallback_index=index),
            section_name=str(chunk["section"]),
            chunk_type=_infer_chunk_type(str(chunk["text"])),
            page_start=int(chunk["page"]),
            content=str(chunk["text"]),
            token_count=len(str(chunk["text"]).split()),
        )
        for index, chunk in enumerate(chunk_meta, start=1)
    ]
