from __future__ import annotations

from dataclasses import dataclass

from services.retrieval_service import RetrievedChunk

MAX_CONTEXT_TOKENS = 6000
TOKENS_PER_CHAR = 0.25


@dataclass
class BuiltContext:
    system_prompt: str
    context_text: str
    citations: list[dict]
    total_tokens: int


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) * TOKENS_PER_CHAR))


def build_context(
    paper_metadata: dict,
    chunks: list[RetrievedChunk],
    question: str,
    chat_history: list[dict] | None = None,
) -> BuiltContext:
    """
    Build a structured context string respecting MAX_CONTEXT_TOKENS.
    """
    system_prompt = (
        "You are a research assistant. Answer questions using ONLY the provided "
        "evidence. Always cite page numbers. If information is not in the "
        "evidence, say 'Not found in this paper.'"
    )

    title = paper_metadata.get("title") or "Unknown Title"
    year = paper_metadata.get("year") or "Unknown Year"
    authors = paper_metadata.get("authors") or []
    authors_text = ", ".join(authors) if authors else "Unknown Authors"

    header = f"PAPER: {title} ({year})\nAUTHORS: {authors_text}\n"
    history_lines: list[str] = []
    if chat_history:
        for item in chat_history[-3:]:
            history_lines.append(f"Q: {item['question']}")
            history_lines.append(f"A: {item['answer']}")

    history_block = ""
    if history_lines:
        history_block = "PREVIOUS CONVERSATION:\n" + "\n".join(history_lines) + "\n\n"

    base_text = f"{header}\n{history_block}RETRIEVED EVIDENCE:\n"
    base_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(header)
    history_tokens = min(200, _estimate_tokens(history_block)) if history_block else 0
    used_tokens = base_tokens + history_tokens

    chunk_budget = MAX_CONTEXT_TOKENS - used_tokens
    citations: list[dict] = []
    evidence_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[{index}] [Page {chunk.page_start} | {chunk.section_name}]\n"
            f"{chunk.content}\n"
        )
        block_tokens = _estimate_tokens(block)
        if block_tokens > chunk_budget:
            break

        evidence_blocks.append(block)
        chunk_budget -= block_tokens
        citations.append(
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "section": chunk.section_name,
                "snippet": chunk.content[:100],
            }
        )

    context_text = base_text + ("\n".join(evidence_blocks).strip() if evidence_blocks else "")
    total_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(context_text)
    return BuiltContext(
        system_prompt=system_prompt,
        context_text=context_text,
        citations=citations,
        total_tokens=total_tokens,
    )
