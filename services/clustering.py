from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from models.embedding_model import get_embedding
from schemas.paper_schema import Paper


def _paper_text(paper: Paper) -> str:
    return f"{paper.title}. {paper.summary}"


def cluster_papers(papers: List[Paper]) -> Dict[str, List[Paper]]:
    if not papers:
        return {"cluster_1": [], "cluster_2": [], "cluster_3": []}

    embeddings = np.vstack([get_embedding(_paper_text(paper)) for paper in papers])
    n_clusters = min(3, len(papers))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    grouped: Dict[str, List[Paper]] = defaultdict(list)
    for paper, label in zip(papers, labels):
        grouped[f"cluster_{int(label) + 1}"].append(paper)

    for cluster_id in range(1, 4):
        grouped.setdefault(f"cluster_{cluster_id}", [])

    return dict(grouped)
