from swag.postprocess import (
    apply_source_diversity,
    collapse_near_duplicate_hits,
    passes_noise_gate,
    source_group_key,
)


def test_source_group_key_groups_by_document():
    a = {"document_id": "doc-1", "source": "s3://b/t/doc-1/a.md", "chunk_index": 0}
    b = {"document_id": "doc-1", "source": "s3://b/t/doc-1/b.md", "chunk_index": 1}
    assert source_group_key(a) == source_group_key(b)


def test_collapse_near_duplicate_drops_similar_text():
    hits = [
        {"text": "Retrieval augmented generation improves LLM answers with context."},
        {"text": "Retrieval augmented generation improves LLM answers with context and citations."},
        {"text": "SMTP mail relay configuration for outbound email."},
    ]
    kept, removed = collapse_near_duplicate_hits(hits, threshold=0.6)
    assert removed >= 1
    assert len(kept) == 2


def test_source_diversity_caps_per_document(monkeypatch):
    from swag.settings import settings as swag_settings

    monkeypatch.setattr(swag_settings, "swag_max_chunks_per_source", 3)
    hits = [
        {"document_id": "d1", "score": 0.9 - i * 0.01, "source": f"s/doc{i}"}
        for i in range(5)
    ]
    selected, skipped = apply_source_diversity(hits, limit=5)
    assert len(selected) <= 3
    assert skipped >= 2


def test_backfill_adds_hits_when_strict_gate_is_sparse(monkeypatch):
    from swag.postprocess import finalize_swag_ranking
    from swag.settings import settings as swag_settings

    monkeypatch.setattr(swag_settings, "swag_noise_backfill_enabled", True)
    monkeypatch.setattr(swag_settings, "swag_noise_relative_vector_ratio", 0.65)
    monkeypatch.setattr(swag_settings, "swag_max_chunks_per_source", 5)

    scored = [
        {"vector_score": 0.6, "swag_score": 0.2, "score": 0.5, "chunk_kind": "body", "text": "a" * 100},
        {"vector_score": 0.35, "swag_score": 0.05, "score": 0.3, "chunk_kind": "body", "text": "b" * 100},
        {"vector_score": 0.34, "swag_score": 0.04, "score": 0.28, "chunk_kind": "body", "text": "c" * 100},
        {"vector_score": 0.20, "swag_score": 0.0, "score": 0.1, "chunk_kind": "body", "text": "d" * 100},
        {"vector_score": 0.10, "swag_score": 0.0, "score": 0.05, "chunk_kind": "boilerplate", "text": "<html>"},
    ]
    reranked, meta = finalize_swag_ranking(scored, limit=4)
    assert len(reranked) >= 2
    assert meta.get("noise_backfilled", 0) >= 1
    assert all(h.get("chunk_kind") != "boilerplate" for h in reranked)


def test_noise_gate_keeps_near_top_vector():
    assert passes_noise_gate({"swag_score": 0.02, "vector_score": 0.59}, top_vector=0.60)
    assert not passes_noise_gate({"swag_score": 0.02, "vector_score": 0.38}, top_vector=0.60)
