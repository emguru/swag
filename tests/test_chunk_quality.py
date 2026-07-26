"""Tests for chunk-kind classification (README chrome → nav)."""

from swag.chunk_quality import classify_chunk_kind


def test_classify_chunk_kinds():
    assert classify_chunk_kind("# RAG From Scratch") == "title"
    assert classify_chunk_kind("<!DOCTYPE html><html><head>") == "boilerplate"
    assert (
        classify_chunk_kind(
            "RAG has emerged as a popular mechanism to expand an LLM knowledge base."
        )
        == "body"
    )


def test_classify_readme_chrome_as_nav():
    chrome = """# RAG From Scratch

[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue)](https://github.com/langchain-ai/rag-from-scratch)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)

Navigation and badges only — not substantive RAG explanation.
"""
    py_chrome = """# CPython

![Python logo](https://www.python.org/static/img/python-logo.png)

## Documentation

- [Tutorial](https://docs.python.org/3/tutorial/)
- [Library reference](https://docs.python.org/3/library/)

## Contributing

See the developer guide. Navigation chrome only.
"""
    body = (
        "RAG has emerged as a popular and powerful mechanism to expand an LLM's "
        "knowledge base, using documents retrieved from an external data source "
        "to ground the LLM generation via in-context learning."
    )
    assert classify_chunk_kind(chrome) == "nav"
    assert classify_chunk_kind(py_chrome) == "nav"
    assert classify_chunk_kind(body) == "body"
