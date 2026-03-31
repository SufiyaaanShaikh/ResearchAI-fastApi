from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def get_embedding(text: str) -> np.ndarray:
    model = load_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return np.asarray(embedding, dtype=np.float32)
