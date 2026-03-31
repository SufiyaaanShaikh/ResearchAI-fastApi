from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, TypedDict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from services.pdf_text import extract_pdf_pages


class RetrievedChunk(TypedDict):
    text: str
    section: str
    page: int
    score: float


class ChunkMeta(TypedDict):
    text: str
    section: str
    page: int
    chunk_id: str


class RagCacheEntry(TypedDict):
    chunks: List[ChunkMeta]
    embeddings: np.ndarray
    index: faiss.IndexFlatL2
    bm25: BM25Okapi
    tokenized_chunks: List[List[str]]


_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None
_pdf_processing_cache: Dict[str, List[str]] = {}
_embedding_cache: Dict[str, RagCacheEntry] = {}

_SECTION_HEADER_REGEX = re.compile(
    r"^\s*(?:[IVX]{1,6}\.\s+)?("
    r"Introduction|Related Work|Method|Methodology|Approach|Experiment|Dataset|Results|Discussion|Conclusion|Future Work"
    r")\s*$",
    re.IGNORECASE,
)
_ROMAN_SECTION_REGEX = re.compile(r"^\s*[IVX]{1,6}\.\s+[A-Za-z][A-Za-z\s\-]{2,80}\s*$", re.IGNORECASE)
_WORD_REGEX = re.compile(r"\b\w+\b")
_TOKEN_SPLIT_REGEX = re.compile(r"\W+")

_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "experiment": ["experiment", "experiment design", "experiment 1", "experiment 2"],
    "methodology": ["method", "methodology", "approach", "model"],
    "dataset": ["dataset", "data used"],
    "results": ["result", "table", "benchmark", "accuracy", "success rate", "performance"],
    "evaluation": ["evaluation", "metric", "result", "analysis"],
    "conclusion": ["conclusion", "discussion", "limitation"],
}
_SECTION_LABEL_MAP: Dict[str, List[str]] = {
    "experiment": ["experiment", "evaluation"],
    "methodology": ["method", "methodology", "approach", "model"],
    "dataset": ["dataset", "data"],
    "results": ["result", "table", "benchmark", "accuracy", "performance"],
    "evaluation": ["evaluation", "result", "analysis", "metric", "table"],
    "conclusion": ["conclusion", "discussion", "limitation"],
}


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _embedding_model


def _get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-large")
    return _reranker_model


def _paper_cache_key(pdf_url: str) -> str:
    match = re.search(r"/([^/]+?)(?:\.pdf)?(?:\?.*)?$", pdf_url)
    if not match:
        return pdf_url

    paper_id = match.group(1)
    paper_id = re.sub(r"v\d+$", "", paper_id)
    return paper_id


def _tokenize(text: str) -> List[str]:
    return [token for token in _TOKEN_SPLIT_REGEX.split(text.lower()) if token]


def _extract_section_title(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    keyword_match = _SECTION_HEADER_REGEX.match(stripped)
    if keyword_match:
        return keyword_match.group(1).title()

    if _ROMAN_SECTION_REGEX.match(stripped):
        normalized = re.sub(r"\s+", " ", stripped)
        return normalized

    return None


def _words_to_chunks(
    words: List[str],
    section: str,
    page: int,
    chunk_counter: List[int],
    chunk_size: int = 1500,
    overlap: int = 250,
) -> List[ChunkMeta]:
    if not words:
        return []

    if chunk_size <= overlap:
        overlap = 0

    step = chunk_size - overlap
    chunks: List[ChunkMeta] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunk_counter[0] += 1
            chunks.append(
                {
                    "text": chunk_text,
                    "section": section,
                    "page": page,
                    "chunk_id": f"chunk_{chunk_counter[0]}",
                }
            )
        if end == len(words):
            break
        start += step
    return chunks


def _semantic_chunk_pages(pages: List[str]) -> List[ChunkMeta]:
    chunks: List[ChunkMeta] = []
    chunk_counter = [0]
    active_section = "Unknown"

    for page_number, page_text in enumerate(pages, start=1):
        lines = [line.rstrip() for line in page_text.splitlines()]
        section_paragraphs: Dict[str, List[str]] = defaultdict(list)

        paragraph_lines: List[str] = []
        for line in lines + [""]:
            section_title = _extract_section_title(line)
            if section_title:
                if paragraph_lines:
                    section_paragraphs[active_section].append(" ".join(paragraph_lines))
                    paragraph_lines = []
                active_section = section_title
                continue

            if not line.strip():
                if paragraph_lines:
                    section_paragraphs[active_section].append(" ".join(paragraph_lines))
                    paragraph_lines = []
                continue

            paragraph_lines.append(line.strip())

        for section, paragraphs in section_paragraphs.items():
            section_text = "\n\n".join(paragraphs).strip()
            section_words = _WORD_REGEX.findall(section_text)
            chunks.extend(
                _words_to_chunks(
                    words=section_words,
                    section=section,
                    page=page_number,
                    chunk_counter=chunk_counter,
                    chunk_size=1500,
                    overlap=250,
                )
            )

    return chunks


def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    model = _get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def _get_or_build_entry(pdf_url: str) -> RagCacheEntry:
    paper_id = _paper_cache_key(pdf_url)
    cached = _embedding_cache.get(paper_id)
    if cached:
        return cached

    pages = _pdf_processing_cache.get(paper_id)
    if pages is None:
        pages = list(extract_pdf_pages(pdf_url))
        _pdf_processing_cache[paper_id] = pages

    chunks = _semantic_chunk_pages(pages)
    if not chunks:
        entry: RagCacheEntry = {
            "chunks": [],
            "embeddings": np.zeros((0, 768), dtype=np.float32),
            "index": faiss.IndexFlatL2(768),
            "bm25": BM25Okapi([["empty"]]),
            "tokenized_chunks": [],
        }
        _embedding_cache[paper_id] = entry
        return entry

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = _embed_texts(chunk_texts)
    index = _build_faiss_index(embeddings)
    tokenized_chunks = [_tokenize(text) for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    entry = {
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "bm25": bm25,
        "tokenized_chunks": tokenized_chunks,
    }
    _embedding_cache[paper_id] = entry
    return entry


def _detect_section_focus(question: str) -> str | None:
    normalized = question.lower()
    for section_name, keywords in _SECTION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return section_name
    return None


def _section_match(section_name: str, chunk_section: str) -> bool:
    terms = _SECTION_LABEL_MAP.get(section_name, [section_name])
    chunk_section_lower = chunk_section.lower()
    return any(term in chunk_section_lower for term in terms)


def retrieve_relevant_chunks(pdf_url: str, question: str, top_k: int = 40) -> List[RetrievedChunk]:
    entry = _get_or_build_entry(pdf_url)
    if not entry["chunks"]:
        return []

    query_embedding = _embed_texts([question])
    _, faiss_indices = entry["index"].search(query_embedding, min(top_k, len(entry["chunks"])))
    semantic_indices = [idx for idx in faiss_indices[0].tolist() if 0 <= idx < len(entry["chunks"])]

    question_tokens = _tokenize(question)
    bm25_scores = entry["bm25"].get_scores(question_tokens)
    top_bm25 = np.argsort(bm25_scores)[::-1][: min(top_k, len(entry["chunks"]))].tolist()

    candidate_indices: List[int] = []
    seen: set[int] = set()
    for idx in semantic_indices + [int(i) for i in top_bm25]:
        if idx not in seen and 0 <= idx < len(entry["chunks"]):
            seen.add(idx)
            candidate_indices.append(idx)

    retrieved_chunks = [entry["chunks"][idx] for idx in candidate_indices]

    section_focus = _detect_section_focus(question)
    if section_focus:
        focused_chunks = [
            chunk for chunk in retrieved_chunks if _section_match(section_focus, chunk["section"])
        ]
        if focused_chunks:
            retrieved_chunks = focused_chunks

    if not retrieved_chunks:
        return []

    reranker = _get_reranker_model()
    rerank_pairs = [(question, chunk["text"]) for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(rerank_pairs)

    scored_chunks = list(zip(retrieved_chunks, rerank_scores))
    scored_chunks.sort(key=lambda item: float(item[1]), reverse=True)
    reranked_chunks = scored_chunks[:10]

    final_chunks: List[RetrievedChunk] = []
    for chunk, score in reranked_chunks:
        final_chunks.append(
            {
                "text": chunk["text"],
                "section": chunk["section"],
                "page": chunk["page"],
                "score": float(score),
            }
        )

    print("FAISS results:", len(semantic_indices))
    print("BM25 results:", len(top_bm25))
    print("Final reranked chunks:", len(reranked_chunks))
    return final_chunks
