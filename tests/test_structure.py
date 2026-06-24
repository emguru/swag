"""Tests for markdown structure splitting."""

from swag.structure import split_markdown_segments


COMBINED = """# RAG From Scratch

[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue)](https://github.com/langchain-ai/rag-from-scratch)

## Table of Contents

- [Overview](#overview)

Navigation and badges only — not substantive RAG explanation.

## What is retrieval augmented generation?

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information.
"""


def test_split_combined_markdown_tags_nav_and_body():
    segments = split_markdown_segments(COMBINED)
    kinds = [kind for _, kind in segments]
    assert "nav" in kinds or "title" in kinds
    assert "body" in kinds
    body_text = " ".join(text for text, kind in segments if kind == "body")
    assert "LLMs are trained" in body_text
