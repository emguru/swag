"""Structure-aware markdown segmentation — split nav/title from body before chunking."""

from __future__ import annotations

import re

from swag.chunk_quality import classify_chunk_kind

_HEADING_SPLIT = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)


def split_markdown_segments(text: str) -> list[tuple[str, str]]:
    """
    Split markdown into segments tagged by chunk kind (nav, title, body, …).

    Preserves line breaks so link/nav heuristics can classify segments.
    """
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    parts = [part.strip() for part in _HEADING_SPLIT.split(cleaned) if part.strip()]
    if not parts:
        return [(cleaned, classify_chunk_kind(cleaned))]

    segments: list[tuple[str, str]] = []
    for part in parts:
        segments.append((part, classify_chunk_kind(part)))
    return segments
