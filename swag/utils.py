"""Utility helpers for the SWAG framework."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any


def generate_chunk_id(source_filename: str, index: int, text_prefix: str = "") -> str:
    """Generate a deterministic chunk ID from source and index."""
    prefix = hashlib.md5(text_prefix[:50].encode()).hexdigest()[:8]
    base = Path(source_filename).stem
    return f"{base}_{index:04d}_{prefix}"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_overlap(text: str) -> list[str]:
    """Simple word tokenization for overlap detection."""
    return re.findall(r"\b\w+\b", text.lower())


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def fuzzy_term_match(query_term: str, candidate_term: str, threshold: float = 0.6) -> bool:
    """
    Check if two terms match using token overlap.
    Avoids expanding isolated ambiguous words — requires multi-token overlap
    or exact substring match for single tokens.
    """
    q = query_term.lower().strip()
    c = candidate_term.lower().strip()

    if q == c:
        return True
    if q in c or c in q:
        return True

    q_tokens = set(tokenize_for_overlap(q))
    c_tokens = set(tokenize_for_overlap(c))

    # Single ambiguous words require exact match
    if len(q_tokens) == 1 and len(c_tokens) == 1:
        return q_tokens == c_tokens

    return jaccard_similarity(q_tokens, c_tokens) >= threshold


def save_json(path: Path, data: Any) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load JSON from path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def supported_extensions() -> set[str]:
    return {".txt", ".md", ".pdf", ".docx"}
