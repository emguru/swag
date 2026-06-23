from swag.schemas import SemanticCategory, SWAGProfile, WeightedSemanticTerm
from swag.scoring import score_swag_alignment
from swag.utils import (
    fuzzy_term_match,
    negative_penalty_match,
    term_grounded_in_text,
    term_mentioned_contrastively,
)
from swag.validation import sanitize_profile_data


def test_fuzzy_rejects_short_substring_match():
    assert not fuzzy_term_match("ai", "mail")
    assert not fuzzy_term_match("it", "kubernetes")
    assert fuzzy_term_match("apple", "apple")


def test_query_negative_excluded_from_positive_matching():
    query = SWAGProfile(
        primary_concepts=["invoice"],
        negative_contexts=["fruit"],
    )
    chunk = SWAGProfile(
        primary_concepts=["fruit"],
        domain_terms=["orchard"],
        confidence=0.8,
    )
    score, explanations = score_swag_alignment(
        query,
        chunk,
        chunk_text="The orchard ships fresh fruit weekly.",
    )
    assert score < 0.5
    assert any("query negative context" in e for e in explanations)


def test_ungrounded_chunk_term_discounted():
    query = SWAGProfile(primary_concepts=["quantum computing"])
    chunk = SWAGProfile(primary_concepts=["quantum computing"], confidence=0.8)
    grounded_score, _ = score_swag_alignment(
        query,
        chunk,
        chunk_text="This document discusses quantum computing research.",
    )
    ungrounded_score, ungrounded_expl = score_swag_alignment(
        query,
        chunk,
        chunk_text="Completely unrelated content about payroll.",
    )
    assert grounded_score > ungrounded_score
    assert any("ungrounded" in e for e in ungrounded_expl)


def test_sanitize_caps_profile_size():
    data = {
        "detected_domain": "x" * 200,
        "primary_concepts": [f"term-{i}" for i in range(50)],
        "weighted_terms": [
            {"term": f"w-{i}", "category": "primary_concept", "weight": 1.0, "reason": "r"}
            for i in range(50)
        ],
    }
    sanitized = sanitize_profile_data(data)
    assert len(sanitized["primary_concepts"]) <= 12
    assert len(sanitized["weighted_terms"]) <= 24
    assert len(sanitized["detected_domain"]) <= 64


def test_term_grounding_requires_token_overlap():
    assert term_grounded_in_text("invoice matching", "Vendor invoice matching rules apply.")
    assert not term_grounded_in_text("blockchain", "Vendor invoice matching rules apply.")


def test_negative_penalty_skips_incidental_token_overlap():
    assert not negative_penalty_match(
        "pure generative models without retrieval",
        "retrieval",
    )
    assert not negative_penalty_match("fine-tuning only approaches", "fine-tuning")
    assert not negative_penalty_match("fine-tuning methodology", "fine-tuning")
    assert not negative_penalty_match(
        "traditional search engines without generation",
        "generation",
    )
    assert negative_penalty_match("fruit", "fruit")
    assert negative_penalty_match("payroll", "payroll")


def test_contrastive_mention_skips_penalty_in_scoring():
    query = SWAGProfile(
        primary_concepts=["retrieval augmented generation"],
        negative_contexts=["fine-tuning methodology"],
    )
    chunk = SWAGProfile(
        primary_concepts=["retrieval augmented generation"],
        domain_terms=["fine-tuning"],
        confidence=0.8,
    )
    contrast_text = (
        "RAG augments LLMs with retrieval instead of relying on fine-tuning alone."
    )
    score, explanations = score_swag_alignment(query, chunk, chunk_text=contrast_text)
    assert score > 0.0
    assert not any(
        e.startswith("Penalty: query negative context") for e in explanations
    )


def test_term_mentioned_contrastively():
    text = "We compare RAG versus fine-tuning-only approaches in this guide."
    assert term_mentioned_contrastively("fine-tuning", text)
    assert not term_mentioned_contrastively("blockchain", text)
