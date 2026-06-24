# RAG From Scratch

[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue)](https://github.com/langchain-ai/rag-from-scratch)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)

Navigation and badges only — not substantive RAG explanation.

## What is retrieval augmented generation?

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Retrieval augmented generation (RAG) addresses this by retrieving relevant documents at query time and conditioning the model on that context.

RAG pipelines typically include: document loading, chunking, embedding, vector storage, retrieval, and generation. Chunk quality and metadata filtering strongly affect answer accuracy.

## Chunking and indexing

Documents are split into overlapping chunks before embedding. Poor chunk boundaries can split tables and code blocks, reducing retrieval precision. Production systems often combine semantic chunking with metadata such as section titles and source URLs.

## Limitations

RAG cannot fix a model that lacks reasoning ability, and stale indexes require re-ingestion. Evaluation should measure both retrieval precision and answer faithfulness.
