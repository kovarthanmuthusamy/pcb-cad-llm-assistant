"""Semantic fallback for naming the analysis type when the text has no explicit SI/FD/TDR term.

Uses a Sentence-BERT model that is loaded the first time it is needed, not at import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

SBERT_MODEL = "all-MiniLM-L6-v2"

# Canonical analysis type -> phrases that should map to it.
SIM_TYPES: dict[str, list[str]] = {
    "FD": ["FD", "Frequency Domain", "Frequency Analysis"],
    "SI": ["SI", "Signal Integrity"],
    "TDR": ["TDR", "Time-Domain Reflectometry"],
}

# (phrase, canonical type) in a fixed order, so an index into the embeddings maps back to a type.
SYNONYMS: list[tuple[str, str]] = [
    (phrase, analysis_type) for analysis_type, phrases in SIM_TYPES.items() for phrase in phrases
]


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode([phrase for phrase, _ in SYNONYMS], convert_to_tensor=True)
    return model, embeddings


def classify_analysis_type(text: str) -> tuple[str, float]:
    """Return the closest analysis type for ``text`` and its cosine similarity."""

    from sentence_transformers import util

    model, embeddings = _load()
    scores = util.cos_sim(model.encode(text, convert_to_tensor=True), embeddings)[0]
    best = int(scores.argmax().item())
    return SYNONYMS[best][1], float(scores[best].item())
