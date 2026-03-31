from __future__ import annotations

from typing import List, Optional

from keybert import KeyBERT

from models.embedding_model import load_model

_kw_model: Optional[KeyBERT] = None


def _get_kw_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model=load_model())
    return _kw_model


def extract_keywords(text: str) -> List[str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    kw_model = _get_kw_model()
    keywords = kw_model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5,
    )
    return [keyword for keyword, _score in keywords]
