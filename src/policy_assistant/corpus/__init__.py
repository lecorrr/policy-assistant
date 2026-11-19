import json
from pathlib import Path

from policy_assistant.contracts import Document
from policy_assistant.corpus.law import LawFormatError, parse_law, read_law_bytes
from policy_assistant.corpus.policies import PolicyFormatError, parse_policy

__all__ = ["CorpusError", "LawFormatError", "PolicyFormatError", "load_corpus", "load_law"]


class CorpusError(ValueError):
    pass


def load_policies(directory: Path, known_roles: set[str]) -> list[Document]:
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise CorpusError(f"no policy files (*.md) in {directory}")
    return [parse_policy(path, known_roles) for path in paths]


def load_law(path: Path) -> Document:
    if not path.is_file():
        raise CorpusError(
            f"{path} is missing: run `policy-assistant fetch-law`, or save the law's text there "
            "yourself (README, 'If the download is refused')"
        )
    return parse_law(read_law_bytes(path.read_bytes()), source=str(path), version=law_version(path))


def law_version(path: Path) -> str:
    """Download date of the law text, from the note fetch-law writes."""
    note = path.with_suffix(".json")
    if note.is_file():
        retrieved = json.loads(note.read_text(encoding="utf-8")).get("retrieved_at", "")
        if retrieved:
            return f"чинна редакція станом на {retrieved[:10]}"
    return "редакція невідома: файл збережено вручну"


def load_corpus(policies_dir: Path, law_path: Path, known_roles: set[str]) -> list[Document]:
    documents = [*load_policies(policies_dir, known_roles), load_law(law_path)]
    seen: set[str] = set()
    for document in documents:
        if document.id in seen:
            raise CorpusError(f"two documents have the id {document.id!r}")
        seen.add(document.id)
    return documents
