"""Utility helpers for the SWAG framework."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from swag.settings import settings

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def generate_chunk_id(source_filename: str, index: int, text_prefix: str = "") -> str:
    prefix = hashlib.md5(text_prefix[:50].encode()).hexdigest()[:8]
    base = Path(source_filename).stem
    return f"{base}_{index:04d}_{prefix}"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def normalize_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_overlap(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def term_grounded_in_text(term: str, text: str, min_ratio: float | None = None) -> bool:
    ratio = min_ratio if min_ratio is not None else settings.swag_grounding_min_token_ratio
    term_tokens = tokenize_for_overlap(term)
    if not term_tokens or not text.strip():
        return False
    text_tokens = set(tokenize_for_overlap(text))
    hits = sum(1 for token in term_tokens if token in text_tokens)
    return hits / len(term_tokens) >= ratio


_EXCLUSION_MARKERS = frozenset(
    {
        "without",
        "only",
        "no",
        "not",
        "non",
        "pure",
        "exclude",
        "excluding",
        "instead",
        "rather",
        "versus",
        "vs",
    }
)

_CONTRASTIVE_WINDOW_RE = re.compile(
    r"(?:\bvs\.?\b|\bversus\b|\bcompared to\b|\bunlike\b|\binstead of\b|"
    r"\brather than\b|\balternative to\b|\bin contrast\b|\bas opposed to\b|"
    r"\bdifferent from\b|\bwithout\b|\bnot only\b)",
    re.IGNORECASE,
)


def term_mentioned_contrastively(term: str, chunk_text: str) -> bool:
    if not term.strip() or not chunk_text.strip():
        return False
    text = chunk_text.lower()
    term_l = term.lower().strip()
    if term_l not in text:
        return False
    start = 0
    while True:
        pos = text.find(term_l, start)
        if pos == -1:
            break
        window = text[max(0, pos - 90) : pos + len(term_l) + 90]
        if _CONTRASTIVE_WINDOW_RE.search(window):
            return True
        start = pos + max(len(term_l), 1)
    return False


def negative_penalty_match(negative_phrase: str, chunk_term: str) -> bool:
    negative = negative_phrase.lower().strip()
    chunk = chunk_term.lower().strip()
    if not negative or not chunk:
        return False
    if negative == chunk:
        return True

    negative_tokens = set(tokenize_for_overlap(negative))
    chunk_tokens = set(tokenize_for_overlap(chunk))
    if not chunk_tokens or not chunk_tokens.issubset(negative_tokens):
        return False

    overlap = jaccard_similarity(negative_tokens, chunk_tokens)

    if chunk_tokens != negative_tokens:
        if len(chunk_tokens) <= 2 and len(negative_tokens) >= 4:
            return False
        if len(chunk_tokens) < len(negative_tokens) and overlap < 0.85:
            return False

    if len(negative_tokens) <= 3:
        return True

    if negative_tokens & _EXCLUSION_MARKERS:
        return len(negative_tokens) <= 5 and overlap >= 0.5

    return overlap >= 0.85


def fuzzy_term_match(query_term: str, candidate_term: str, threshold: float = 0.6) -> bool:
    q = query_term.lower().strip()
    c = candidate_term.lower().strip()
    if not q or not c:
        return False
    if q == c:
        return True

    min_len = settings.swag_fuzzy_min_token_len
    q_tokens = set(tokenize_for_overlap(q))
    c_tokens = set(tokenize_for_overlap(c))

    if len(q_tokens) == 1 and len(c_tokens) == 1:
        return q_tokens == c_tokens

    if len(q) >= min_len and len(c) >= min_len:
        if q in c or c in q:
            return True

    return jaccard_similarity(q_tokens, c_tokens) >= threshold


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def supported_extensions() -> set[str]:
    return {".txt", ".md", ".pdf", ".docx"}
