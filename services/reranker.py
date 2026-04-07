from __future__ import annotations

from services.rag_pipeline import _SECTION_BOOST, _detect_section_focus, _get_reranker_model, _section_match
from services.retrieval_service import RetrievedChunk


def get_reranker():
    return _get_reranker_model()


def rerank_chunks(question: str, chunks: list[RetrievedChunk], top_n: int = 12) -> list[RetrievedChunk]:
    """
    Rerank chunks with the cross-encoder and apply section boosting.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(question, chunk.content) for chunk in chunks]
    raw_scores = reranker.predict(pairs).tolist()
    section_focus = _detect_section_focus(question.lower())

    rescored: list[RetrievedChunk] = []
    for chunk, raw_score in zip(chunks, raw_scores):
        score = float(raw_score)
        if section_focus and section_focus.lower() in chunk.section_name.lower():
            score *= _SECTION_BOOST
        chunk.combined_score = score
        rescored.append(chunk)

    rescored.sort(key=lambda chunk: chunk.combined_score, reverse=True)
    return rescored[:top_n]


def rerank_chunks_with_all(
    question: str,
    candidates: list[RetrievedChunk],
    all_chunks: list[RetrievedChunk],
    top_n: int = 12,
) -> list[RetrievedChunk]:
    """Calls rerank_chunks with neighbor expansion support."""
    if not candidates:
        return []

    reranked = rerank_chunks(question, candidates, top_n=len(candidates))
    all_by_index = {chunk.chunk_index: chunk for chunk in sorted(all_chunks, key=lambda item: item.chunk_index)}
    selected: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in reranked}

    for chunk in reranked[:5]:
        for neighbor_index in (chunk.chunk_index - 1, chunk.chunk_index + 1):
            neighbor = all_by_index.get(neighbor_index)
            if neighbor is None or neighbor.chunk_id in selected:
                continue

            neighbor.vector_score = chunk.vector_score
            neighbor.keyword_score = chunk.keyword_score
            neighbor.combined_score = chunk.combined_score * 0.8
            selected[neighbor.chunk_id] = neighbor

    expanded = sorted(selected.values(), key=lambda chunk: chunk.combined_score, reverse=True)
    return expanded[:top_n]
