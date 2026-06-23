from swag.heuristic import heuristic_chunk_profile, heuristic_query_profile
from swag.profile import SWAGProfileGenerator
from swag.scoring import score_swag_alignment


def test_heuristic_query_profile_has_concepts():
    profile = heuristic_query_profile("What is retrieval augmented generation?")
    assert profile.confidence > 0
    assert profile.primary_concepts
    assert any("retrieval" in t.lower() for t in profile.domain_terms + profile.primary_concepts)


def test_heuristic_chunk_profile_from_text():
    text = "RAG augments LLMs with retrieval instead of fine-tuning alone."
    profile = heuristic_chunk_profile(text, source_filename="rag.md")
    assert profile.confidence > 0
    assert profile.primary_concepts


def test_generator_uses_heuristic_when_configured(monkeypatch):
    from swag.settings import settings

    monkeypatch.setattr(settings, "swag_profile_mode", "heuristic")
    gen = SWAGProfileGenerator()
    profile = gen.generate_query_profile("invoice matching rules")
    assert profile.confidence >= 0.7
    assert "invoice" in " ".join(profile.domain_terms + profile.primary_concepts).lower()


def test_heuristic_rag_beats_mail_trap():
    query = heuristic_query_profile("What is retrieval augmented generation?")
    rag_chunk = heuristic_chunk_profile(
        "Retrieval augmented generation combines vector search with LLM generation.",
        source_filename="rag.md",
    )
    mail_chunk = heuristic_chunk_profile(
        "Configure SMTP mail relay for outbound email.",
        source_filename="mail.md",
    )
    rag_score, _ = score_swag_alignment(
        query,
        rag_chunk,
        chunk_text="Retrieval augmented generation combines vector search with LLM generation.",
    )
    mail_score, _ = score_swag_alignment(
        query,
        mail_chunk,
        chunk_text="Configure SMTP mail relay for outbound email.",
    )
    assert rag_score > mail_score
