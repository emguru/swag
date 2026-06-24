"""Confidence-gated SWAG rerank — skip when baseline vector search is already decisive."""

from __future__ import annotations

from swag.chunk_quality import chunk_kind_from_profile, classify_chunk_kind
from swag.profile import profile_from_payload
from swag.settings import settings


def should_rerank_with_swag(hits: list[dict], *, query: str) -> tuple[bool, str]:
    """
    Decide whether SWAG rerank is worth the cost for this candidate pool.

    Rerank when top-1 is low-quality chrome or vector scores are ambiguous.
    Skip when a body chunk clearly leads the pool.
    """
    del query  # reserved for future query-shape gates (definitional vs navigational)
    if not hits:
        return False, "no_candidates"
    if len(hits) == 1:
        kind = _top_kind(hits[0])
        if kind in {"nav", "title", "boilerplate"}:
            return True, f"single_candidate_kind={kind}"
        return False, "single_body_candidate"

    top_score = float(hits[0].get("score") or 0.0)
    second_score = float(hits[1].get("score") or 0.0)
    margin = top_score - second_score
    epsilon = settings.swag_adaptive_vector_epsilon
    top_kind = _top_kind(hits[0])

    if top_kind in {"nav", "title", "boilerplate"}:
        return True, f"top1_kind={top_kind}"

    if margin <= epsilon:
        return True, "vector_margin_ambiguous"

    if top_kind == "body" and margin > epsilon:
        return False, "clear_body_winner"

    return True, "default_rerank"


def _top_kind(hit: dict) -> str:
    profile = profile_from_payload(hit.get("swag_profile"))
    text = str(hit.get("text") or "")
    metadata_kind = (hit.get("metadata") or {}).get("chunk_kind")
    if metadata_kind:
        return str(metadata_kind)
    explicit = hit.get("chunk_kind")
    if explicit:
        return str(explicit)
    return chunk_kind_from_profile(profile, text=text) or classify_chunk_kind(text)
