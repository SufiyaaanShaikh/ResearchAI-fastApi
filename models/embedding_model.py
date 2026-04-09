from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR = "./hf_cache"
_model: Optional[SentenceTransformer] = None

os.makedirs(_CACHE_DIR, exist_ok=True)


def load_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    try:
        _model = SentenceTransformer(
            _MODEL_NAME,
            cache_folder=_CACHE_DIR,
        )
    except Exception as exc:
        print("Primary embedding model unavailable:", exc)
        print("Falling back to lightweight model")
        _model = SentenceTransformer(
            _FALLBACK_MODEL_NAME,
            cache_folder=_CACHE_DIR,
        )

    return _model


def get_embedding(text: str) -> np.ndarray:
    model = load_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float32)
