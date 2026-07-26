"""Chunk-type signals for reranking — derived from chunk text only (no source mutation)."""

from __future__ import annotations

import re

from swag.settings import settings
from swag.schemas import SWAGProfile

_HTML_MARKERS = ("<!doctype", "<html", "<head", "<meta", "<script")
_LINK_MARKERS = ("](http", "![", "http://", "https://")
_BADGE_MARKERS = ("img.shields.io", "shields.io", "badge/")
_HEADER_RE = re.compile(r"^#{1,4}\s+\S")
_SENTENCE_RE = re.compile(r"[.!?]\s+\S")
_ANCHOR_BULLET_RE = re.compile(r"[-*]\s*\[[^\]]+\]\(#")
_STRIP_LINKS_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)|\[[^\]]+\]\([^)]+\)")


def _substantive_sentence_count(cleaned: str) -> int:
    """Count sentences that look like explanatory prose, not nav labels."""
    plain = _STRIP_LINKS_RE.sub(" ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+", plain)
    substantive = 0
    for part in parts:
        words = [w for w in part.split() if w]
        if len(words) >= 12 or len(part) >= 80:
            substantive += 1
    return substantive


def classify_chunk_kind(text: str) -> str:
    """Classify chunk shape: body, title, nav, boilerplate, or empty."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "empty"

    lower = cleaned.lower()
    if any(marker in lower for marker in _HTML_MARKERS) or lower.count("<") >= 4:
        return "boilerplate"

    link_signals = sum(cleaned.count(marker) for marker in _LINK_MARKERS)
    anchor_bullets = len(_ANCHOR_BULLET_RE.findall(cleaned))
    sentence_count = len(_SENTENCE_RE.findall(cleaned)) + (1 if cleaned.endswith((".", "!", "?")) else 0)
    substantive_sentences = _substantive_sentence_count(cleaned)

    if "table of contents" in lower or anchor_bullets >= 2:
        return "nav"
    if any(marker in lower for marker in _BADGE_MARKERS) and link_signals >= 2:
        return "nav"
    if link_signals >= 3 and len(cleaned) < 450 and substantive_sentences <= 1:
        return "nav"
    if link_signals >= 2 and len(cleaned) < 320 and sentence_count <= 1:
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
