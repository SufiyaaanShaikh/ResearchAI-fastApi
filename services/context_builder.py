from __future__ import annotations

from dataclasses import dataclass

from services.retrieval_service import RetrievedChunk

MAX_CONTEXT_TOKENS = 12000
# Word-count based estimation is more accurate for mixed scientific text.
# 1 token ~= 0.75 words (GPT standard), so words * 1.35 ~= tokens.
_WORDS_PER_TOKEN_RATIO = 1.35


@dataclass
class BuiltContext:
    system_prompt: str
    context_text: str
    citations: list[dict]
    total_tokens: int


def _estimate_tokens(text: str) -> int:
    """Estimate tokens using word count - more accurate for scientific text."""
    return max(1, int(len(text.split()) * _WORDS_PER_TOKEN_RATIO))


def build_context(
    paper_metadata: dict,
    chunks: list[RetrievedChunk],
    question: str,
    chat_history: list[dict] | None = None,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
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

    chunk_budget = max_context_tokens - used_tokens
    citations: list[dict] = []
    evidence_blocks: list[str] = []
    token_total = used_tokens

    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[{index}] [Page {chunk.page_start} | {chunk.section_name}]\n"
            f"{chunk.content}\n"
        )
        chunk_tokens = getattr(chunk, "token_count", None) or len(chunk.content.split())
        block_tokens = max(chunk_tokens, _estimate_tokens(block))

        # Changed from break to continue:
        # A single oversized chunk should not prevent smaller subsequent chunks
        # from being included. Only stop if even a minimal chunk won't fit.
        if block_tokens > chunk_budget:
            if chunk_budget < 100:  # Truly exhausted - stop.
                break
            continue  # This chunk is too large; try the next one.

        evidence_blocks.append(block)
        chunk_budget -= block_tokens
        token_total += block_tokens
        citations.append(
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_start,
                "section": chunk.section_name,
                "snippet": chunk.content[:100],
            }
        )

    if not evidence_blocks:
        # Nothing fit - include first chunk truncated to 800 words as fallback.
        if chunks:
            fallback_content = " ".join(chunks[0].content.split()[:800])
            evidence_blocks.append(
                f"[1] [Page {chunks[0].page_start} | {chunks[0].section_name}]\n"
                f"{fallback_content}\n"
            )
            citations.append({
                "chunk_id": chunks[0].chunk_id,
                "page": chunks[0].page_start,
                "section": chunks[0].section_name,
                "snippet": chunks[0].content[:100],
            })

    context_text = base_text + ("\n".join(evidence_blocks).strip() if evidence_blocks else "")
    total_tokens = min(token_total, _estimate_tokens(system_prompt) + _estimate_tokens(context_text))
    return BuiltContext(
        system_prompt=system_prompt,
        context_text=context_text,
        citations=citations,
        total_tokens=total_tokens,
    )
