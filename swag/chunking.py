"""Document chunking with configurable size and overlap."""

from __future__ import annotations

import logging
from pathlib import Path

from swag.config import ChunkingConfig
from swag.schemas import DocumentChunk
from swag.utils import generate_chunk_id, normalize_text

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    source_filename: str,
    config: ChunkingConfig,
    page_number: int | None = None,
) -> list[DocumentChunk]:
    """
    Split text into overlapping chunks preserving metadata.

    Uses character-based chunking with word-boundary awareness
    to avoid splitting mid-word when possible.
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    chunk_size = config.chunk_size
    overlap = config.chunk_overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Snap to word boundary unless at document end
        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx > start + chunk_size // 2:
                end = space_idx

        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append(
                DocumentChunk(
                    chunk_id=generate_chunk_id(source_filename, index, chunk_text_slice),
                    source_filename=source_filename,
                    page_number=page_number,
                    raw_text=chunk_text_slice,
                    char_start=start,
                    char_end=end,
                    metadata={
                        "document_id": source_filename,
                        "chunk_index": index,
                    },
                )
            )
            index += 1

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    logger.debug("Chunked %s into %d chunks", source_filename, len(chunks))
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source_filename: str,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    """Chunk multi-page content, preserving page numbers."""
    all_chunks: list[DocumentChunk] = []
    global_index = 0

    for page_num, page_text in pages:
        page_text = normalize_text(page_text)
        if not page_text:
            continue

        page_chunks = chunk_text(page_text, source_filename, config, page_number=page_num)
        for chunk in page_chunks:
            chunk.chunk_id = generate_chunk_id(
                source_filename, global_index, chunk.raw_text
            )
            global_index += 1
            all_chunks.append(chunk)

    return all_chunks


def merge_small_chunks(
    chunks: list[DocumentChunk],
    min_size: int = 100,
) -> list[DocumentChunk]:
    """Merge chunks smaller than min_size with the previous chunk."""
    if not chunks:
        return []

    merged: list[DocumentChunk] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk.raw_text) < min_size and merged:
            prev = merged[-1]
            merged[-1] = DocumentChunk(
                chunk_id=prev.chunk_id,
                source_filename=prev.source_filename,
                page_number=prev.page_number,
                raw_text=prev.raw_text + " " + chunk.raw_text,
                char_start=prev.char_start,
                char_end=chunk.char_end,
                metadata=prev.metadata,
            )
        else:
            merged.append(chunk)
    return merged
