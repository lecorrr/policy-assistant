import re

from policy_assistant.contracts import Block, ChunkDraft, Document, Paragraph
from policy_assistant.corpus.law import law_label

# ~350 tokens of Ukrainian; the reranker reads up to 512.
DEFAULT_MAX_CHARS = 1200

_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


def chunk_documents(
    documents: list[Document], max_chars: int = DEFAULT_MAX_CHARS
) -> list[ChunkDraft]:
    return [chunk for document in documents for chunk in chunk_document(document, max_chars)]


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[ChunkDraft]:
    if max_chars < 200:
        raise ValueError("max_chars below 200 would cut most sentences apart")
    chunks: list[ChunkDraft] = []
    for block in document.blocks:
        pieces = _pack(block.paragraphs, max_chars)
        labels = [_label(document, block, piece, whole=len(pieces) == 1) for piece in pieces]
        if len(set(labels)) < len(labels):
            # Number the pieces so each citation names exactly one.
            labels = [f"{label}, фрагмент {i}/{len(labels)}" for i, label in enumerate(labels, 1)]
        for piece, label in zip(pieces, labels, strict=True):
            chunks.append(
                ChunkDraft(
                    doc_id=document.id,
                    position=len(chunks),
                    section=block.section,
                    label=label,
                    heading=f"{document.title}. {block.title}",
                    body="\n".join(paragraph.text for paragraph in piece),
                    roles=block.roles,
                )
            )
    return chunks


def _label(document: Document, block: Block, piece: list[Paragraph], *, whole: bool) -> str:
    if document.kind == "law":
        parts = [] if whole else list(dict.fromkeys(p.part for p in piece if p.part))
        return law_label(block.section, parts)
    return f"{document.title}, розд. {block.section}"


def _pack(paragraphs: tuple[Paragraph, ...], max_chars: int) -> list[list[Paragraph]]:
    """Pack paragraphs greedily into pieces of at most max_chars."""
    pieces: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    size = 0
    for paragraph in paragraphs:
        for text in _split_long(paragraph.text, max_chars):
            piece = Paragraph(text, paragraph.part)
            if current and size + len(text) + 1 > max_chars:
                pieces.append(current)
                current, size = [], 0
            current.append(piece)
            size += len(text) + 1
    if current:
        # Merge a short tail into the previous piece.
        if pieces and size < max_chars // 4 and _size(pieces[-1]) + size <= max_chars * 1.3:
            pieces[-1].extend(current)
        else:
            pieces.append(current)
    return pieces


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:  # one sentence longer than a chunk: cut at a space
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _size(piece: list[Paragraph]) -> int:
    return sum(len(paragraph.text) + 1 for paragraph in piece)
