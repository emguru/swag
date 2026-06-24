"""Runtime settings bridge — kube-compatible attribute names for ported SWAG modules."""

from __future__ import annotations

from swag.config import SWAGConfig, config


class SettingsBridge:
    """Expose SWAG tuning knobs under the same names used in kube_rag_langg."""

    def __init__(self, cfg: SWAGConfig) -> None:
        self._cfg = cfg

    @property
    def rag_top_k(self) -> int:
        return self._cfg.top_k

    @property
    def swag_enabled(self) -> bool:
        return self._cfg.swag.swag_enabled

    @property
    def swag_ingest_profiles(self) -> bool:
        return self._cfg.swag.swag_ingest_profiles

    @property
    def swag_profile_mode(self) -> str:
        return self._cfg.swag.swag_profile_mode

    @property
    def swag_grader_model(self) -> str:
        return self._cfg.swag.swag_grader_model

    @property
    def swag_retrieval_candidates(self) -> int:
        return self._cfg.retrieval_candidates

    @property
    def swag_vector_weight(self) -> float:
        return self._cfg.scoring.vector_weight

    @property
    def swag_weight(self) -> float:
        return self._cfg.scoring.swag_weight

    @property
    def swag_metadata_weight(self) -> float:
        return self._cfg.scoring.metadata_weight

    @property
    def swag_min_score_threshold(self) -> float:
        return self._cfg.swag.swag_min_score_threshold

    @property
    def swag_noise_vector_margin(self) -> float:
        return self._cfg.swag.swag_noise_vector_margin

    @property
    def swag_noise_relative_vector_ratio(self) -> float:
        return self._cfg.swag.swag_noise_relative_vector_ratio

    @property
    def swag_noise_backfill_ratio(self) -> float:
        return self._cfg.swag.swag_noise_backfill_ratio

    @property
    def swag_evaluative_blend_boost(self) -> float:
        return self._cfg.swag.swag_evaluative_blend_boost

    @property
    def swag_evaluative_title_penalty(self) -> float:
        return self._cfg.swag.swag_evaluative_title_penalty

    @property
    def swag_chunk_title_multiplier(self) -> float:
        return self._cfg.swag.swag_chunk_title_multiplier

    @property
    def swag_chunk_nav_multiplier(self) -> float:
        return self._cfg.swag.swag_chunk_nav_multiplier

    @property
    def swag_chunk_boilerplate_multiplier(self) -> float:
        return self._cfg.swag.swag_chunk_boilerplate_multiplier

    @property
    def swag_intent_category_boost(self) -> float:
        return self._cfg.swag.swag_intent_category_boost

    @property
    def swag_max_chunks_per_source(self) -> int:
        return self._cfg.swag.swag_max_chunks_per_source

    @property
    def swag_deduplicate_similarity(self) -> float:
        return self._cfg.swag.swag_deduplicate_similarity

    @property
    def swag_adaptive_blend(self) -> bool:
        return self._cfg.swag.swag_adaptive_blend

    @property
    def swag_adaptive_vector_epsilon(self) -> float:
        return self._cfg.swag.swag_adaptive_vector_epsilon

    @property
    def swag_adaptive_swag_boost(self) -> float:
        return self._cfg.swag.swag_adaptive_swag_boost

    @property
    def swag_persist_profiles(self) -> bool:
        return self._cfg.swag.swag_persist_profiles

    @property
    def swag_max_terms_per_list(self) -> int:
        return self._cfg.swag.swag_max_terms_per_list

    @property
    def swag_max_weighted_terms(self) -> int:
        return self._cfg.swag.swag_max_weighted_terms

    @property
    def swag_max_term_chars(self) -> int:
        return self._cfg.swag.swag_max_term_chars

    @property
    def swag_max_reason_chars(self) -> int:
        return self._cfg.swag.swag_max_reason_chars

    @property
    def swag_max_domain_chars(self) -> int:
        return self._cfg.swag.swag_max_domain_chars

    @property
    def swag_ungrounded_term_weight_multiplier(self) -> float:
        return self._cfg.swag.swag_ungrounded_term_weight_multiplier

    @property
    def swag_grounding_min_token_ratio(self) -> float:
        return self._cfg.swag.swag_grounding_min_token_ratio

    @property
    def swag_fuzzy_min_token_len(self) -> int:
        return self._cfg.swag.swag_fuzzy_min_token_len

    @property
    def swag_gated_rerank(self) -> bool:
        return self._cfg.swag.swag_gated_rerank

    def __setattr__(self, name: str, value) -> None:
        if name == "_cfg":
            object.__setattr__(self, name, value)
            return
        if name == "rag_top_k":
            self._cfg.top_k = int(value)
            return
        if name in {"swag_vector_weight", "swag_weight", "swag_metadata_weight"}:
            field = name.replace("swag_", "").replace("vector_", "vector_")
            mapping = {
                "swag_vector_weight": "vector_weight",
                "swag_weight": "swag_weight",
                "swag_metadata_weight": "metadata_weight",
            }
            setattr(self._cfg.scoring, mapping[name], value)
            return
        if hasattr(self._cfg.swag, name):
            setattr(self._cfg.swag, name, value)
            return
        raise AttributeError(name)


def reload_settings(cfg: SWAGConfig | None = None) -> SettingsBridge:
    global settings
    settings = SettingsBridge(cfg or SWAGConfig.from_env())
    return settings


settings = SettingsBridge(config)
