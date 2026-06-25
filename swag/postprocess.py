"""Post-ranking filters: dedupe, diversity, noise floor, adaptive blend weights."""

from __future__ import annotations

from swag.settings import settings
from swag.utils import jaccard_similarity, tokenize_for_overlap


def source_group_key(hit: dict) -> str:
    """Group chunks by document (not individual chunk)."""
    doc_id = hit.get("document_id")
    if doc_id:
        return str(doc_id)
    source = str(hit.get("source") or "")
    if source:
        parts = source.rstrip("/").split("/")
        if len(parts) >= 2:
            return "/".join(parts[:-1])
        return source
    return str(hit.get("id") or "")


def _text_similarity(a: str, b: str) -> float:
    ta = set(tokenize_for_overlap(a))
    tb = set(tokenize_for_overlap(b))
    return jaccard_similarity(ta, tb)


def collapse_near_duplicate_hits(hits: list[dict], *, threshold: float) -> tuple[list[dict], int]:
    """Drop later hits whose text closely matches an earlier candidate."""
    if threshold <= 0 or len(hits) < 2:
        return hits, 0
    kept: list[dict] = []
    removed = 0
    for hit in hits:
        text = str(hit.get("text") or "")
        if not text.strip():
            kept.append(hit)
            continue
        if any(_text_similarity(text, str(k.get("text") or "")) >= threshold for k in kept):
            removed += 1
            continue
        kept.append(hit)
    return kept, removed


def adaptive_blend_weights(
    vector_scores: list[float],
    *,
    evaluative_query: bool = False,
) -> tuple[float, float, float]:
    """
    When top vector scores cluster — or the query is evaluative — lean on SWAG.
    """
    if not settings.swag_adaptive_blend or len(vector_scores) < 2:
        return (
            settings.swag_vector_weight,
            settings.swag_weight,
            settings.swag_metadata_weight,
        )

    ordered = sorted(vector_scores, reverse=True)
    spread = ordered[0] - ordered[min(2, len(ordered) - 1)]
    clustered = spread <= settings.swag_adaptive_vector_epsilon
    if not clustered and not evaluative_query:
        return (
            settings.swag_vector_weight,
            settings.swag_weight,
            settings.swag_metadata_weight,
        )

    boost = settings.swag_adaptive_swag_boost
    if evaluative_query:
        boost += settings.swag_evaluative_blend_boost
    vector_w = max(0.40, settings.swag_vector_weight - boost)
    swag_w = min(0.55, settings.swag_weight + boost)
    return vector_w, swag_w, settings.swag_metadata_weight


def passes_noise_gate(
    hit: dict,
    *,
    top_vector: float,
    relative_ratio: float | None = None,
) -> bool:
    """Drop weak semantic fits unless vector score is near the leader."""
    floor = settings.swag_min_score_threshold
    if floor <= 0:
        return True
    swag = float(hit.get("swag_score") or 0.0)
    vector = float(hit.get("vector_score") or hit.get("score") or 0.0)
    if swag >= floor:
        return True
    if top_vector <= 0:
        return False
    ratio_cutoff = (
        relative_ratio
        if relative_ratio is not None
        else settings.swag_noise_relative_vector_ratio
    )
    relative = vector / top_vector
    if relative >= ratio_cutoff:
        return True
    return vector >= top_vector - settings.swag_noise_vector_margin


def _hard_noise_kind(hit: dict) -> bool:
    return str(hit.get("chunk_kind") or "body") in {"boilerplate", "empty"}


def backfill_gated_hits(
    scored: list[dict],
    gated: list[dict],
    *,
    top_vector: float,
    limit: int,
) -> tuple[list[dict], int]:
    """Relax noise ratio slightly to reach showcase top-k without admitting junk."""
    if len(gated) >= limit:
        return gated, 0
    seen = {id(h) for h in gated}
    backfilled = 0
    relaxed = settings.swag_noise_backfill_ratio
    for hit in scored:
        if len(gated) >= limit:
            break
        if id(hit) in seen or _hard_noise_kind(hit):
            continue
        if passes_noise_gate(hit, top_vector=top_vector, relative_ratio=relaxed):
            gated.append(hit)
            seen.add(id(hit))
            backfilled += 1
    return gated, backfilled


def _substance_sort_key(hit: dict) -> tuple:
    """Tie-break: higher SWAG, substantive body chunks, longer grounded text."""
    kind = str(hit.get("chunk_kind") or "body")
    kind_rank = {"body": 0, "title": 1, "nav": 2, "boilerplate": 3, "empty": 4}.get(kind, 1)
    text_len = len(str(hit.get("text") or ""))
    return (
        -float(hit.get("score") or 0.0),
        -float(hit.get("swag_score") or 0.0),
        kind_rank,
        -text_len,
    )


def apply_source_diversity(hits: list[dict], *, limit: int) -> tuple[list[dict], int]:
    """Cap how many chunks from the same document appear in top-k."""
    max_per = settings.swag_max_chunks_per_source
    if max_per <= 0:
        return hits[:limit], 0
    selected: list[dict] = []
    counts: dict[str, int] = {}
    skipped = 0
    for hit in hits:
        if len(selected) >= limit:
            break
        key = source_group_key(hit)
        if counts.get(key, 0) >= max_per:
            skipped += 1
            continue
        selected.append(hit)
        counts[key] = counts.get(key, 0) + 1
    return selected, skipped


def finalize_swag_ranking(scored: list[dict], *, limit: int) -> tuple[list[dict], dict]:
    """Sort, noise-gate, diversify, and trim to top-k."""
    scored.sort(key=_substance_sort_key)
    top_vector = max((float(h.get("vector_score") or 0.0) for h in scored), default=0.0)

    gated = [h for h in scored if passes_noise_gate(h, top_vector=top_vector)]
    noise_dropped = len(scored) - len(gated)
    noise_backfilled = 0
    if settings.swag_noise_backfill_enabled:
        gated, noise_backfilled = backfill_gated_hits(
            scored, gated, top_vector=top_vector, limit=limit
        )

    diversified, diversity_skipped = apply_source_diversity(gated, limit=limit)
    reranked = diversified[:limit]

    for rank, hit in enumerate(reranked, start=1):
        hit["swag_rank"] = rank

    return reranked, {
        "noise_dropped": noise_dropped,
        "noise_backfilled": noise_backfilled,
        "diversity_skipped": diversity_skipped,
    }
