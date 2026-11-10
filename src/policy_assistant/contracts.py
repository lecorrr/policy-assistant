"""Data types shared by every module.

Text is Ukrainian. Times are milliseconds, prices are US dollars per million tokens.
"""

from dataclasses import dataclass, field

# Every signed-in user has this role; documents meant for all staff carry it.
STAFF_ROLE = "staff"


@dataclass(frozen=True)
class Paragraph:
    text: str
    # The numbered part of a law article this paragraph belongs to ("1", "4-1"); None elsewhere.
    part: str | None = None


@dataclass(frozen=True)
class Block:
    """One citable unit of a document: a law article, a point of the final provisions, an annex
    or a policy section. A chunk never spans two blocks."""

    section: str  # "15", "IV.3", "Додаток 1", "преамбула" for the law; "2" for a policy section
    title: str  # "Стаття 15. Право споживача на відмову від договору про споживчий кредит"
    paragraphs: tuple[Paragraph, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    kind: str  # "law" or "policy"
    source: str  # where the text came from: a URL or a file path
    version: str
    blocks: tuple[Block, ...]
    url: str | None = None  # a page people can open to read the original

    @property
    def roles(self) -> tuple[str, ...]:
        """Everyone who may read at least one block, in a stable order."""
        return tuple(sorted({role for block in self.blocks for role in block.roles}))


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it has a vector and a database id."""

    doc_id: str
    position: int
    section: str
    label: str  # how a citation names it: "Закон № 1734-VIII, ст. 15, ч. 1–3"
    heading: str  # document and section titles, searched and shown to the model with the body
    body: str
    roles: tuple[str, ...]

    @property
    def search_text(self) -> str:
        return f"{self.heading}\n{self.body}"


@dataclass(frozen=True)
class Hit:
    """A chunk returned by search, readable by the user who searched."""

    chunk_id: int
    doc_id: str
    section: str
    label: str
    heading: str
    body: str
    roles: tuple[str, ...]
    url: str | None
    score: float = 0.0  # the score of the last stage that ranked it

    @property
    def search_text(self) -> str:
        return f"{self.heading}\n{self.body}"


@dataclass(frozen=True)
class User:
    login: str
    name: str
    title: str
    roles: tuple[str, ...]  # always includes STAFF_ROLE


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] the model wrote
    hit: Hit


@dataclass
class Answer:
    question: str
    text: str  # the model's answer with [n] markers, or the refusal sentence
    refused: bool
    citations: list[Citation]
    bad_citations: list[int]  # numbers the model cited that were not among its excerpts
    hits: list[Hit]  # the excerpts the model saw, in the order it saw them
    model: str
    usage: Usage = field(default_factory=Usage)
    timings_ms: dict[str, float] = field(default_factory=dict)
