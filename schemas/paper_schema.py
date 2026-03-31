from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class Paper(BaseModel):
    id: str
    title: str
    summary: str


class PaperList(BaseModel):
    papers: List[Paper]


class SimilarPapersRequest(BaseModel):
    target_paper: Paper
    papers: List[Paper]


class SimilarPapersResponse(BaseModel):
    similar: List[Paper]


class KeywordsRequest(BaseModel):
    text: str


class KeywordsResponse(BaseModel):
    keywords: List[str]


class ClusterPapersResponse(BaseModel):
    clusters: Dict[str, List[Paper]]


class PDFTextRequest(BaseModel):
    pdf_url: str


class PDFTextResponse(BaseModel):
    text: str


class RagQueryRequest(BaseModel):
    pdf_url: str
    question: str


class RagChunk(BaseModel):
    text: str
    section: str
    page: int
    score: float | None = None


class RagQueryResponse(BaseModel):
    context_chunks: List[RagChunk]
