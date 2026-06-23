from __future__ import annotations

from swag.settings import settings
from swag.chunk_quality import chunk_kind_from_profile, chunk_quality_multiplier
from swag.heuristic import is_evaluative_query
from swag.postprocess import (
    adaptive_blend_weights,
    collapse_near_duplicate_hits,
    finalize_swag_ranking,
)
from swag.profiles import ensure_swag_profiles
from swag.profile import SWAGProfileGenerator, profile_from_payload
from swag.schemas import QueryProfile
from swag.scoring import (
    build_selection_explanation,
    compute_final_score,
    compute_metadata_boost,
    score_swag_alignment,
)


def _hit_key(hit: dict) -> str:
    return str(
        hit.get("id")
        or hit.get("document_id")
        or f"{hit.get('source', '')}:{hit.get('page', '')}:{hit.get('chunk_index', '')}"
    )


def rerank_hits_with_swag(
    query: str,
    hits: list[dict],
    *,
    collection: str = "",
    top_k: int | None = None,
    vector_store=None,
) -> tuple[list[dict], dict]:
    """Rerank vector hits using SWAG semantic profile alignment."""
    limit = top_k or settings.rag_top_k
    if not hits:
        return [], {
            "enabled": True,
            "candidates": 0,
            "profiles_available": 0,
            "profiles_generated": 0,
            "profiles_cached": 0,
            "profiling_ran": False,
            "query_domain": None,
        }

    hits, profile_meta = ensure_swag_profiles(
        hits, collection=collection, vector_store=vector_store
    )
    candidate_count = len(hits)

    dedupe_threshold = settings.swag_deduplicate_similarity
    hits, deduped = collapse_near_duplicate_hits(hits, threshold=dedupe_threshold)

    generator = SWAGProfileGenerator()
    query_profile = generator.generate_query_profile(query)
    query_profile_obj = QueryProfile(original_query=query, swag_profile=query_profile)
    evaluative = is_evaluative_query(query)

    vector_scores = [float(h.get("score") or 0.0) for h in hits]
    vector_w, swag_w, meta_w = adaptive_blend_weights(vector_scores, evaluative_query=evaluative)

    scored: list[dict] = []
    profiles_available = 0

    for hit in hits:
        vector_score = float(hit.get("score") or 0.0)
        chunk_text = str(hit.get("text") or "")
        chunk_profile = profile_from_payload(hit.get("swag_profile"))
        if chunk_profile and chunk_profile.confidence > 0:
            profiles_available += 1

        chunk_kind = chunk_kind_from_profile(chunk_profile, text=chunk_text)
        quality = chunk_quality_multiplier(chunk_kind, evaluative_query=evaluative)

        if chunk_profile and chunk_profile.confidence > 0:
            swag_score, swag_explanations = score_swag_alignment(
                query_profile_obj.swag_profile,
                chunk_profile,
                chunk_text=chunk_text,
                evaluative_query=evaluative,
            )
            if quality < 1.0:
                swag_explanations.append(
                    f"Chunk kind '{chunk_kind}' quality multiplier {quality:.2f}"
                )
            metadata_boost = compute_metadata_boost(
                chunk_profile=chunk_profile,
                query_profile=query_profile_obj.swag_profile,
                source=str(hit.get("source") or ""),
                original_query=query,
            )
        else:
            swag_score = 0.0
            swag_explanations = ["Semantic profile unavailable for chunk"]
            metadata_boost = 0.0

        base_final = compute_final_score(
            vector_score,
            swag_score,
            metadata_boost,
            vector_weight=vector_w,
            swag_weight=swag_w,
            metadata_weight=meta_w,
        )
        final_score = base_final * quality
        explanation = build_selection_explanation(
            vector_score,
            swag_score,
            metadata_boost,
            final_score,
            swag_explanations,
        )
        if quality < 1.0:
            explanation += f" Final score scaled by chunk quality ({quality:.2f})."

        enriched = dict(hit)
        enriched["vector_score"] = vector_score
        enriched["swag_score"] = swag_score
        enriched["metadata_boost"] = metadata_boost
        enriched["chunk_kind"] = chunk_kind
        enriched["score"] = final_score
        enriched["swag_explanation"] = explanation
        scored.append(enriched)

    reranked, post_meta = finalize_swag_ranking(scored, limit=limit)

    meta = {
        "enabled": True,
        "candidates": candidate_count,
        "profiles_available": profiles_available,
        "profiles_generated": profile_meta.get("profiles_generated", 0),
        "profiles_cached": profile_meta.get("profiles_cached", 0),
        "profiling_ran": profile_meta.get("profiling_ran", False),
        "query_domain": query_profile.detected_domain,
        "evaluative_query": evaluative,
        "vector_weight": vector_w,
        "swag_weight": swag_w,
        "metadata_weight": meta_w,
        "deduped": deduped,
        "noise_dropped": post_meta.get("noise_dropped", 0),
        "noise_backfilled": post_meta.get("noise_backfilled", 0),
        "diversity_skipped": post_meta.get("diversity_skipped", 0),
    }
    return reranked, meta


def compute_rank_changes(baseline_hits: list[dict], augmented_hits: list[dict]) -> list[dict]:
    baseline_rank = {_hit_key(hit): idx + 1 for idx, hit in enumerate(baseline_hits)}
    augmented_rank = {_hit_key(hit): idx + 1 for idx, hit in enumerate(augmented_hits)}

    changes: list[dict] = []
    for key in set(baseline_rank) | set(augmented_rank):
        before = baseline_rank.get(key)
        after = augmented_rank.get(key)
        delta = None
        if before is not None and after is not None:
            delta = before - after
        changes.append(
            {
                "key": key,
                "baseline_rank": before,
                "augmented_rank": after,
                "rank_delta": delta,
            }
        )

    changes.sort(
        key=lambda item: (
            item["augmented_rank"] if item["augmented_rank"] is not None else 999,
            item["baseline_rank"] if item["baseline_rank"] is not None else 999,
        )
    )
    return changes
