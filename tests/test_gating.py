"""Tests for confidence-gated SWAG rerank."""

from swag.gating import should_rerank_with_swag


def test_gate_skips_clear_body_winner():
    hits = [
        {"score": 0.82, "text": "Body paragraph with several sentences. It explains RAG in detail.", "chunk_kind": "body"},
        {"score": 0.55, "text": "Nav links only", "chunk_kind": "nav"},
    ]
    should, reason = should_rerank_with_swag(hits, query="what is rag")
    assert should is False
    assert reason == "clear_body_winner"


def test_gate_reranks_nav_top1():
    hits = [
        {"score": 0.80, "text": "[TOC](#overview) [Install](#install)", "chunk_kind": "nav"},
        {"score": 0.78, "text": "LLMs are trained on a large but fixed corpus of data.", "chunk_kind": "body"},
    ]
    should, reason = should_rerank_with_swag(hits, query="what is rag")
    assert should is True
    assert reason == "top1_kind=nav"


def test_gate_reranks_ambiguous_margin():
    hits = [
        {"score": 0.71, "text": "Paragraph about chunking boundaries and overlap.", "chunk_kind": "body"},
        {"score": 0.70, "text": "Another body paragraph about indexing strategies.", "chunk_kind": "body"},
    ]
    should, reason = should_rerank_with_swag(hits, query="chunking quality")
    assert should is True
    assert reason == "vector_margin_ambiguous"
