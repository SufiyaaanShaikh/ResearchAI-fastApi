from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple, TypedDict

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

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SECTION_HEADER_REGEX = re.compile(
    r"^\s*(?:[IVX]{1,6}\.\s+)?("
    r"Introduction|Related Work|Method|Methodology|Approach|Experiment|Dataset|Results|Discussion|Conclusion|Future Work|"
    r"Background|Preliminaries|Notation|Overview|Framework|Architecture|System|Training|Formulation|Objective|Loss|"
    r"Evaluation|Baseline|Ablation|Analysis|Limitation|Appendix|Supplement|Contribution|Motivation|Problem Statement|"
    r"Related|Prior Work"
    r")\s*$",
    re.IGNORECASE,
)
_ROMAN_SECTION_REGEX = re.compile(
    r"^\s*[IVX]{1,6}\.\s+[A-Za-z][A-Za-z\s\-]{2,80}\s*$",
    re.IGNORECASE,
)

# FIX B-1: Detect "TABLE I", "TABLE II", "TABLE 1" etc. as their own section
# boundary so table rows are not buried inside a giant surrounding prose chunk.
_TABLE_HEADER_REGEX = re.compile(
    r"^\s*(TABLE\s+(?:[IVXLCDM]+|\d+))\s*$",
    re.IGNORECASE,
)

# FIX C-1: Keep numeric tokens (0.3, 0.54, m3/h) intact instead of splitting on
# every non-word character.  The original _WORD_REGEX r"\b\w+\b" turned "0.3"
# into ["0", "3"] and stripped units like "m3/h" to ["m3", "h"].
# The new pattern matches sequences of word chars plus dots/slashes/percent so
# that "0.54", "m3/h", "48%", "1,350" are kept as single tokens.
_CHUNK_TOKEN_REGEX = re.compile(r"[\w][\w./,%+-]*")

# BM25 tokeniser stays simple (split on whitespace/punctuation) so lookup stays fast.
_TOKEN_SPLIT_REGEX = re.compile(r"\W+")

# ---------------------------------------------------------------------------
# FIX B-2: Expanded _SECTION_KEYWORDS and _SECTION_LABEL_MAP
#
# Problems with the original:
#   • "asymptomatic rate" — not in any keyword list → section_focus=None
#     (harmless because filter was skipped, but no boosting either)
#   • "TABLE I" / "basic parameters" — mapped to "results" via the "table"
#     keyword, then the section match looked for chunks labelled "result/table/
#     benchmark/accuracy/performance" — none existed in this paper → the filter
#     was skipped and boosting missed the right chunks
#   • "ventilation rate", "quanta", "filtration efficiency" — not in any list
#
# Fix: add a "parameters" focus that matches the experiment-design / basic-
# parameters sections that IEEE papers typically have.
# ---------------------------------------------------------------------------
_SECTION_KEYWORDS: Dict[str, List[str]] = {
    "background": [
        "background", "preliminar", "notation", "overview", "framework",
        "architecture", "formulation", "prior work", "related work", "motivation",
    ],
    "experiment": ["experiment", "experiment design", "experiment 1", "experiment 2"],
    "methodology": ["method", "methodology", "approach", "model"],
    "dataset": ["dataset", "data used"],
    "results": ["result", "benchmark", "accuracy", "success rate", "performance", "figure"],
    "evaluation": ["evaluation", "metric", "analysis"],
    "conclusion": ["conclusion", "discussion", "limitation"],
    # NEW — catches table lookups and parameter lookups
    "parameters": [
        "table", "table i", "table ii", "table iii", "table 1", "table 2",
        "table 3", "basic parameter", "asymptomatic", "ventilation", "quanta",
        "filtration", "pulmonary", "infectability", "infection probability",
        "exposed days", "infectious days", "asymptomatic days",
    ],
}

_SECTION_LABEL_MAP: Dict[str, List[str]] = {
    "background": [
        "background", "preliminar", "notation", "overview", "framework",
        "architecture", "formulation", "prior work", "related work", "motivation",
    ],
    "experiment": ["experiment", "evaluation"],
    "methodology": ["method", "methodology", "approach", "model"],
    "dataset": ["dataset", "data"],
    "results": ["result", "benchmark", "accuracy", "performance"],
    "evaluation": ["evaluation", "result", "analysis", "metric"],
    "conclusion": ["conclusion", "discussion", "limitation"],
    # NEW — matches the chunks that were labelled "Table I", "Table Ii",
    # "Experiment Design", "Basic Parameters" etc.
    "parameters": [
        "table", "parameter", "experiment design", "basic", "infection model",
        "school virus", "simulator",
    ],
}

# How much to boost reranker scores for chunks in the focused section.
# 1.25 = +25 % — enough to pull a correct chunk past a generic one without
# completely overriding the reranker's quality signal.
_SECTION_BOOST = 1.25


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section / header detection
# ---------------------------------------------------------------------------

def _extract_section_title(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    # Standard named sections ("Introduction", "III. Results" …)
    keyword_match = _SECTION_HEADER_REGEX.match(stripped)
    if keyword_match:
        return keyword_match.group(1).title()

    # Roman-numeral sections of arbitrary name ("IV. Experiment Design")
    if _ROMAN_SECTION_REGEX.match(stripped):
        return re.sub(r"\s+", " ", stripped)

    # FIX B-1: Table headers ("TABLE I", "TABLE II", "TABLE 2" …)
    # These are isolated on their own line in IEEE / ACM papers and should start
    # a fresh mini-section so the table rows are not diluted by surrounding prose.
    table_match = _TABLE_HEADER_REGEX.match(stripped)
    if table_match:
        return table_match.group(1).title()  # → "Table I", "Table Ii" etc.

    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _words_to_chunks(
    words: List[str],
    section: str,
    page: int,
    chunk_counter: List[int],
    chunk_size: int = 600,
    overlap: int = 150,
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
            # FIX C-1: Use _CHUNK_TOKEN_REGEX instead of _WORD_REGEX so numeric
            # values like "0.3", "0.54(m3/h)", "1,350" are preserved as tokens.
            section_words = _CHUNK_TOKEN_REGEX.findall(section_text)
            chunks.extend(
                _words_to_chunks(
                    words=section_words,
                    section=section,
                    page=page_number,
                    chunk_counter=chunk_counter,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Embedding / FAISS helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Section focus helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    pdf_url: str,
    question: str,
    top_k: int = 60,
    top_n: int = 20,
) -> List[RetrievedChunk]:
    entry = _get_or_build_entry(pdf_url)
    if not entry["chunks"]:
        return []

    # --- Semantic search (FAISS) ---
    query_embedding = _embed_texts([question])
    _, faiss_indices = entry["index"].search(query_embedding, min(top_k, len(entry["chunks"])))
    semantic_indices = [idx for idx in faiss_indices[0].tolist() if 0 <= idx < len(entry["chunks"])]

    # --- Keyword search (BM25) ---
    question_tokens = _tokenize(question)
    bm25_scores = entry["bm25"].get_scores(question_tokens)
    top_bm25 = np.argsort(bm25_scores)[::-1][: min(top_k, len(entry["chunks"]))].tolist()

    # --- Merge candidates (deduped) ---
    candidate_indices: List[int] = []
    seen: set[int] = set()
    for idx in semantic_indices + [int(i) for i in top_bm25]:
        if idx not in seen and 0 <= idx < len(entry["chunks"]):
            seen.add(idx)
            candidate_indices.append(idx)

    retrieved_chunks = [entry["chunks"][idx] for idx in candidate_indices]

    if not retrieved_chunks:
        return []

    # --- Cross-encoder reranking ---
    reranker = _get_reranker_model()
    rerank_pairs = [(question, chunk["text"]) for chunk in retrieved_chunks]
    rerank_scores = reranker.predict(rerank_pairs).tolist()

    # FIX D: Replace the hard section filter with a scoring BOOST.
    #
    # The old code did:
    #   focused_chunks = [c for c if section_match(c)]
    #   if focused_chunks: retrieved_chunks = focused_chunks   ← hard discard
    #
    # The problem: if section detection is wrong (e.g. "table" → "results"
    # but this paper labels the section "Table I" / "Experiment Design"), the
    # hard filter throws away the correct chunks entirely.
    #
    # The new approach: let the cross-encoder score stand as the quality signal,
    # and apply a 1.25× multiplicative boost to chunks that ARE in the focused
    # section.  Wrong-section-but-correct-content chunks still survive.
    section_focus = _detect_section_focus(question)

    scored: List[Tuple[ChunkMeta, float]] = []
    for chunk, raw_score in zip(retrieved_chunks, rerank_scores):
        score = float(raw_score)
        if section_focus and _section_match(section_focus, chunk["section"]):
            score *= _SECTION_BOOST
        scored.append((chunk, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    chunk_positions = {
        chunk["chunk_id"]: idx
        for idx, chunk in enumerate(entry["chunks"])
    }
    scored_ids = {chunk["chunk_id"] for chunk, _ in scored}
    neighbor_candidates: List[Tuple[ChunkMeta, float]] = []
    for chunk, score in scored[:5]:
        chunk_index = chunk_positions.get(chunk["chunk_id"])
        if chunk_index is None:
            continue
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            if 0 <= neighbor_index < len(entry["chunks"]):
                neighbor_chunk = entry["chunks"][neighbor_index]
                neighbor_id = neighbor_chunk["chunk_id"]
                if neighbor_id in scored_ids:
                    continue
                scored_ids.add(neighbor_id)
                neighbor_candidates.append((neighbor_chunk, score * 0.8))

    if neighbor_candidates:
        scored.extend(neighbor_candidates)
        scored.sort(key=lambda pair: pair[1], reverse=True)

    top_scored = scored[:top_n]

    final_chunks: List[RetrievedChunk] = [
        {
            "text": chunk["text"],
            "section": chunk["section"],
            "page": chunk["page"],
            "score": score,
        }
        for chunk, score in top_scored
    ]

    print("FAISS results:", len(semantic_indices))
    print("BM25 results:", len(top_bm25))
    print("Section focus:", section_focus)
    print("Final reranked chunks:", len(final_chunks))
    for i, c in enumerate(final_chunks):
        print(f"  [{i+1}] section={c['section']!r:30s} page={c['page']} score={c['score']:.3f}")

    return final_chunks
