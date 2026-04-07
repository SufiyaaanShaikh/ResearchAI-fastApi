from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.embedding_model import load_model
from schemas.paper_schema import (
    ClusterPapersResponse,
    KeywordsRequest,
    KeywordsResponse,
    PDFTextRequest,
    PDFTextResponse,
    PaperList,
    RagQueryRequest,
    RagQueryResponse,
    SimilarPapersRequest,
    SimilarPapersResponse,
)
from services.clustering import cluster_papers
from services.keywords import extract_keywords
from services.pdf_text import extract_pdf_text
from services.rag_pipeline import retrieve_relevant_chunks
from services.similarity import find_similar_papers

app = FastAPI(title="ResearchAI FastAPI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.post("/similar-papers", response_model=SimilarPapersResponse)
def similar_papers(payload: SimilarPapersRequest) -> SimilarPapersResponse:
    similar = find_similar_papers(payload.target_paper, payload.papers)
    return SimilarPapersResponse(similar=similar)


@app.post("/keywords", response_model=KeywordsResponse)
def keywords(payload: KeywordsRequest) -> KeywordsResponse:
    keywords_list = extract_keywords(payload.text)
    return KeywordsResponse(keywords=keywords_list)


@app.post("/cluster-papers", response_model=ClusterPapersResponse)
def cluster(payload: PaperList) -> ClusterPapersResponse:
    clusters = cluster_papers(payload.papers)
    return ClusterPapersResponse(clusters=clusters)


@app.post("/extract-pdf-text", response_model=PDFTextResponse)
def extract_full_paper_text(payload: PDFTextRequest) -> PDFTextResponse:
    text = extract_pdf_text(payload.pdf_url)
    return PDFTextResponse(text=text)


@app.get("/extract-pdf-sections")
def extract_sections(pdf_url: str) -> dict:
    from services.pdf_text import extract_pdf_pages
    from services.rag_pipeline import _extract_section_title

    pages = extract_pdf_pages(pdf_url)
    sections = []
    seen = set()
    for page in pages:
        for line in page.splitlines():
            title = _extract_section_title(line)
            if title and title not in seen:
                seen.add(title)
                sections.append(title)
    return {"sections": sections}


@app.post("/rag-query", response_model=RagQueryResponse)
def rag_query(payload: RagQueryRequest) -> RagQueryResponse:
    results = retrieve_relevant_chunks(payload.pdf_url, payload.question, top_k=60, top_n=payload.top_n)
    return RagQueryResponse(context_chunks=results)


from routers.papers import router as papers_router

app.include_router(papers_router)


from routers.query import router as query_router

app.include_router(query_router)
