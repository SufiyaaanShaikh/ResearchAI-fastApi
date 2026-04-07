from __future__ import annotations

import numpy as np

from models.embedding_model import load_model


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.
    Use the already-loaded SentenceTransformer (all-mpnet-base-v2, 768-dim).
    normalize_embeddings=True for cosine similarity.
    Returns np.ndarray shape (768,) dtype float32.
    """
    model = load_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float32)


def embed_batch(texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
    """
    Embed a batch of texts in chunks of batch_size.
    Returns list of np.ndarray.
    """
    if not texts:
        return []

    model = load_model()
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings.extend(np.asarray(batch_embeddings, dtype=np.float32))
    return embeddings
