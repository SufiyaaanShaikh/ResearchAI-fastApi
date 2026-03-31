from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.embedding_model import get_embedding
from schemas.paper_schema import Paper


def _paper_text(paper: Paper) -> str:
    return f"{paper.title}. {paper.summary}"


def find_similar_papers(target_paper: Paper, papers: List[Paper]) -> List[Paper]:
    if not papers:
        return []

    candidates = [paper for paper in papers if paper.id != target_paper.id]
    if not candidates:
        return []

    target_embedding = get_embedding(_paper_text(target_paper)).reshape(1, -1)
    candidate_embeddings = np.vstack([get_embedding(_paper_text(paper)) for paper in candidates])

    scores = cosine_similarity(target_embedding, candidate_embeddings).flatten()
    ranked_indices = np.argsort(scores)[::-1]

    top_indices = ranked_indices[:3]
    return [candidates[index] for index in top_indices]
