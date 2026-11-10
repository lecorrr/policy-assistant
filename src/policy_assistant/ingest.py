"""Parse, chunk, embed and store the corpus."""

import hashlib
from dataclasses import dataclass
from typing import Protocol

from policy_assistant.adapters.db import StoredChunk, vector_literal
from policy_assistant.contracts import ChunkDraft, Document
from policy_assistant.corpus.chunking import chunk_documents
from policy_assistant.search import Embedder
from policy_assistant.text import Lemmatizer, tsvector_literal


class AdminStore(Protocol):
    def known_embeddings(self, content_hashes: list[str]) -> dict[str, str]: ...

    def replace_corpus(
        self, documents: list[Document], chunks: list[StoredChunk], *, embedding_model: str
    ) -> None: ...


@dataclass(frozen=True)
class IngestReport:
    documents: list[Document]
    chunks: list[ChunkDraft]
    embedded: int  # chunks sent to the embedder
    reused: int  # chunks whose embedding was already stored


def content_hash(embedding_model: str, chunk: ChunkDraft) -> str:
    """Changes when the text or the embedding model changes, so stale vectors are never reused."""
    return hashlib.sha256(f"{embedding_model}\n{chunk.search_text}".encode()).hexdigest()


def ingest(
    documents: list[Document],
    *,
    store: AdminStore,
    embedder: Embedder,
    embedding_model: str,
    lemmatizer: Lemmatizer,
    max_chars: int,
) -> IngestReport:
    chunks = chunk_documents(documents, max_chars)
    hashes = [content_hash(embedding_model, chunk) for chunk in chunks]
    vectors = store.known_embeddings(sorted(set(hashes)))
    reused = sum(1 for h in hashes if h in vectors)

    missing = list(dict.fromkeys(h for h in hashes if h not in vectors))
    texts = {h: chunk.search_text for h, chunk in zip(hashes, chunks, strict=True)}
    for content, vector in zip(missing, embedder.embed([texts[h] for h in missing]), strict=True):
        vectors[content] = vector_literal(vector)

    stored = [
        StoredChunk(
            draft=chunk,
            lexemes=tsvector_literal(lemmatizer.words(chunk.heading), lemmatizer.words(chunk.body)),
            embedding=vectors[content],
            content_hash=content,
        )
        for chunk, content in zip(chunks, hashes, strict=True)
    ]
    store.replace_corpus(documents, stored, embedding_model=embedding_model)
    return IngestReport(documents=documents, chunks=chunks, embedded=len(missing), reused=reused)
