"""Document ingestion pipeline — load, chunk, embed, index, generate SWAG profiles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from swag.chunking import chunk_markdown_or_text, chunk_pages, chunk_text
from swag.config import SWAGConfig
from swag.embeddings import BaseEmbedder, create_embedder
from swag.schemas import IngestResult
from swag.profile import SWAGProfileGenerator
from swag.utils import supported_extensions
from swag.vector_store import BaseVectorStore, create_vector_store

logger = logging.getLogger(__name__)


def load_text_file(path: Path) -> list[tuple[int | None, str]]:
    """Load .txt or .md file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(None, text)]


def load_pdf_file(path: Path) -> list[tuple[int, str]]:
    """Load PDF with page numbers."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def load_docx_file(path: Path) -> list[tuple[int | None, str]]:
    """Load DOCX file."""
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("Install python-docx: pip install python-docx") from exc

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return [(None, "\n\n".join(paragraphs))]


def load_document(path: Path) -> list[tuple[int | None, str]]:
    """Load document based on file extension."""
    ext = path.suffix.lower()
    loaders = {
        ".txt": load_text_file,
        ".md": load_text_file,
        ".pdf": load_pdf_file,
        ".docx": load_docx_file,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {supported_extensions()}")
    return loaders[ext](path)


class IngestionPipeline:
    """End-to-end document ingestion with SWAG profile generation."""

    def __init__(
        self,
        config: Optional[SWAGConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        profile_generator: Optional[SWAGProfileGenerator] = None,
    ):
        self.config = config or SWAGConfig.from_env()
        self.config.ensure_dirs()
        self.embedder = embedder or create_embedder(self.config)
        self.vector_store = vector_store or create_vector_store(
            self.config, dimension=self.embedder.dimension
        )
        self.profile_generator = profile_generator or SWAGProfileGenerator()

    def ingest_file(self, path: Path, generate_swag: bool = True) -> IngestResult:
        """Ingest a single file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info("Ingesting: %s", path.name)
        pages = load_document(path)
        source_filename = path.name

        # Chunk document
        if pages and pages[0][0] is not None:
            chunks = chunk_pages(
                [(p[0], p[1]) for p in pages if p[0] is not None],
                source_filename,
                self.config.chunking,
            )
        else:
            full_text = "\n\n".join(p[1] for p in pages)
            chunks = chunk_markdown_or_text(full_text, source_filename, self.config.chunking)

        if not chunks:
            logger.warning("No chunks produced from %s", path.name)
            return IngestResult(source_filename=source_filename, num_chunks=0, chunk_ids=[])

        # Generate SWAG profiles (stored separately from raw_text)
        if generate_swag:
            for chunk in chunks:
                chunk.swag_profile = self.profile_generator.generate_chunk_profile(
                    chunk_text=chunk.raw_text,
                    source_filename=chunk.source_filename,
                    page_number=chunk.page_number,
                )
                kind = chunk.metadata.get("chunk_kind")
                if kind and chunk.swag_profile:
                    chunk.swag_profile.chunk_kind = str(kind)

        # Generate embeddings from ORIGINAL chunk text only
        texts = [c.raw_text for c in chunks]
        embeddings = self.embedder.embed(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.vector_store.add(chunks, embeddings)

        return IngestResult(
            source_filename=source_filename,
            num_chunks=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
        )

    def ingest_fixture(
        self,
        path: Path,
        *,
        seed_key: str,
        topic: str | None = None,
        generate_swag: bool = True,
    ) -> IngestResult:
        """Ingest a validation fixture with stable seed_key for qrel eval."""
        path = Path(path)
        pages = load_document(path)
        source_filename = f"validation://{seed_key}"

        if pages and pages[0][0] is not None:
            chunks = chunk_pages(
                [(p[0], p[1]) for p in pages if p[0] is not None],
                source_filename,
                self.config.chunking,
            )
        else:
            full_text = "\n\n".join(p[1] for p in pages)
            chunks = chunk_markdown_or_text(full_text, source_filename, self.config.chunking)

        for chunk in chunks:
            chunk.metadata["seed_key"] = seed_key
            if topic:
                chunk.metadata["topic"] = topic

        if not chunks:
            return IngestResult(source_filename=source_filename, num_chunks=0, chunk_ids=[])

        if generate_swag:
            for chunk in chunks:
                chunk.swag_profile = self.profile_generator.generate_chunk_profile(
                    chunk_text=chunk.raw_text,
                    source_filename=seed_key.split("/")[-1],
                    page_number=chunk.page_number,
                )
                kind = chunk.metadata.get("chunk_kind")
                if kind and chunk.swag_profile:
                    chunk.swag_profile.chunk_kind = str(kind)

        texts = [c.raw_text for c in chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.vector_store.add(chunks, embeddings)
        return IngestResult(
            source_filename=source_filename,
            num_chunks=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
        )

    def ingest_directory(
        self,
        directory: Optional[Path] = None,
        generate_swag: bool = True,
    ) -> list[IngestResult]:
        """Ingest all supported files from a directory."""
        data_dir = directory or self.config.data_dir
        data_dir = Path(data_dir)
        results: list[IngestResult] = []

        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in supported_extensions():
                continue
            try:
                result = self.ingest_file(path, generate_swag=generate_swag)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", path.name, exc)

        return results

    def save_index(self) -> None:
        """Persist vector index and SWAG metadata."""
        self.vector_store.save()

    def load_index(self) -> bool:
        """Load existing index."""
        return self.vector_store.load()
