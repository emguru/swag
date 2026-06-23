"""Chunk-type signals for reranking — derived from chunk text only (no source mutation)."""

from __future__ import annotations

import re

from swag.settings import settings
from swag.schemas import SWAGProfile

_HTML_MARKERS = ("<!doctype", "<html", "<head", "<meta", "<script")
_LINK_MARKERS = ("](http", "![", "http://", "https://")
_HEADER_RE = re.compile(r"^#{1,4}\s+\S")
_SENTENCE_RE = re.compile(r"[.!?]\s+\S")


def classify_chunk_kind(text: str) -> str:
    """Classify chunk shape: body, title, nav, boilerplate, or empty."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "empty"

    lower = cleaned.lower()
    if any(marker in lower for marker in _HTML_MARKERS) or lower.count("<") >= 4:
        return "boilerplate"

    link_signals = sum(cleaned.count(marker) for marker in _LINK_MARKERS)
    sentence_count = len(_SENTENCE_RE.findall(cleaned)) + (1 if cleaned.endswith((".", "!", "?")) else 0)
    if link_signals >= 2 and len(cleaned) < 260 and sentence_count <= 1:
        return "nav"

    if _HEADER_RE.match(cleaned) and len(cleaned) < 160:
        return "title"
    if len(cleaned) < 90 and sentence_count == 0:
        return "title"

    return "body"


def chunk_kind_from_profile(profile: SWAGProfile | None, *, text: str = "") -> str:
    if profile and profile.chunk_kind and profile.chunk_kind != "body":
        return profile.chunk_kind
    return classify_chunk_kind(text)


def chunk_quality_multiplier(
    kind: str,
    *,
    evaluative_query: bool = False,
) -> float:
    """Scale SWAG contribution by chunk type (retrieval hints, not source edits)."""
    multipliers = {
        "body": 1.0,
        "title": settings.swag_chunk_title_multiplier,
        "nav": settings.swag_chunk_nav_multiplier,
        "boilerplate": settings.swag_chunk_boilerplate_multiplier,
        "empty": 0.0,
    }
    base = multipliers.get(kind, 1.0)
    if evaluative_query and kind in {"title", "nav"}:
        base *= settings.swag_evaluative_title_penalty
    return max(0.0, min(1.0, base))
