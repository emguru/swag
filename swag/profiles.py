from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from swag.settings import settings
from swag.profile import SWAGProfileGenerator, profile_from_payload
from swag.schemas import SWAGProfile

if TYPE_CHECKING:
    from swag.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


def _profile_ready(profile: SWAGProfile | None) -> bool:
    return profile is not None and profile.confidence > 0


def ensure_swag_profiles(
    hits: list[dict],
    *,
    collection: str = "",
    vector_store: BaseVectorStore | None = None,
) -> tuple[list[dict], dict]:
    """Ensure candidate chunks have SWAG profiles (generate inline if missing)."""
    if not hits:
        return [], {"profiles_generated": 0, "profiles_cached": 0, "profiling_ran": False}

    generator = SWAGProfileGenerator()
    enriched: list[dict] = []
    generated = 0
    cached = 0

    for hit in hits:
        profile = profile_from_payload(hit.get("swag_profile"))
        if _profile_ready(profile):
            cached += 1
            enriched.append(hit)
            continue

        source_name = str(hit.get("source", "unknown")).split("/")[-1]
        page = hit.get("page")
        page_number = int(page) if isinstance(page, int) else None
        profile = generator.generate_chunk_profile(
            str(hit.get("text") or ""),
            source_filename=source_name,
            page_number=page_number,
        )
        updated = dict(hit)
        updated["swag_profile"] = profile.model_dump()
        point_id = updated.get("id")
        if point_id and settings.swag_persist_profiles and vector_store is not None:
            try:
                vector_store.update_chunk_profile(str(point_id), profile)
            except Exception as exc:
                logger.warning("Failed to persist SWAG profile for %s: %s", point_id, exc)
        generated += 1
        enriched.append(updated)

    return enriched, {
        "profiles_generated": generated,
        "profiles_cached": cached,
        "profiling_ran": generated > 0,
    }
