import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from policy_assistant.config import UsersConfig, describe_validation_error
from policy_assistant.contracts import ChunkDraft, Hit
from policy_assistant.text import normalize

_CITATION = re.compile(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]")


class EvalSetError(ValueError):
    pass


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str
    section: str | None = None
    evidence: str = Field(min_length=5)

    def matches(self, chunk: Hit | ChunkDraft) -> bool:
        """Same document and section, and the chunk contains the evidence (normalised)."""
        if chunk.doc_id != self.doc or (self.section is not None and chunk.section != self.section):
            return False
        return normalize(self.evidence) in normalize(chunk.search_text)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Z]\d{2}$")
    ask_as: str
    question: str = Field(min_length=5)
    answerable: bool = True
    sources: list[Source] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    leak_as: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Question":
        if self.answerable and not (self.sources and self.facts):
            raise ValueError(f"{self.id}: an answerable question needs sources and facts")
        if not self.answerable and (self.sources or self.facts or self.leak_as):
            raise ValueError(
                f"{self.id}: an unanswerable question has no sources, facts or leak_as"
            )
        if self.ask_as in self.leak_as:
            raise ValueError(f"{self.id}: {self.ask_as} both asks and is a leak user")
        _compile(self.id, self.facts)
        return self

    def answer_matches(self, text: str) -> bool:
        return facts_match(self.facts, text)


class Injection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^I\d{2}$")
    ask_as: str
    question: str = Field(min_length=5)
    facts: list[str] = Field(min_length=1)  # any answer matching all of these leaked

    @model_validator(mode="after")
    def _patterns(self) -> "Injection":
        _compile(self.id, self.facts)
        return self


class EvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    law_edition: str
    questions: list[Question] = Field(min_length=1)
    injections: list[Injection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        ids = [q.id for q in self.questions] + [i.id for i in self.injections]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate ids: {', '.join(duplicates)}")
        return self

    @property
    def answerable(self) -> list[Question]:
        return [q for q in self.questions if q.answerable]


def load_eval_set(path: Path) -> EvalSet:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvalSet.model_validate(data)
    except FileNotFoundError:
        raise EvalSetError(f"{path} is missing") from None
    except yaml.YAMLError as error:
        raise EvalSetError(f"{path}: not valid YAML: {error}") from None
    except ValidationError as error:
        raise EvalSetError(f"{path}: {describe_validation_error(error)}") from None


def without_citations(text: str) -> str:
    return _CITATION.sub("", text)


def facts_match(patterns: list[str], text: str) -> bool:
    plain = normalize(without_citations(text))
    return all(re.search(pattern, plain, re.IGNORECASE) for pattern in patterns)


def check_eval_set(eval_set: EvalSet, users: UsersConfig, chunks: list[ChunkDraft]) -> list[str]:
    """Problems that would make the scores meaningless, checked against the chunked corpus."""
    problems = []
    for item in [*eval_set.questions, *eval_set.injections]:
        for login in [item.ask_as, *getattr(item, "leak_as", [])]:
            if users.user(login) is None:
                problems.append(f"{item.id}: unknown user {login!r}")
    for question in eval_set.answerable:
        asker = users.user(question.ask_as)
        if asker is None:
            continue
        evidence_chunks = []
        for source in question.sources:
            found = [chunk for chunk in chunks if source.matches(chunk)]
            if not found:
                where = f"{source.doc}" + (f" section {source.section}" if source.section else "")
                problems.append(f"{question.id}: evidence {source.evidence!r} is not in {where}")
            evidence_chunks.extend(found)
        if not evidence_chunks:
            continue
        if not any(facts_match(question.facts, chunk.search_text) for chunk in evidence_chunks):
            problems.append(f"{question.id}: the facts cannot be read from the evidence chunks")
        if not any(set(chunk.roles) & set(asker.roles) for chunk in evidence_chunks):
            problems.append(f"{question.id}: {question.ask_as} cannot read any evidence chunk")
        for login in question.leak_as:
            leak_user = users.user(login)
            if leak_user is None:
                continue
            readable = [c for c in chunks if set(c.roles) & set(leak_user.roles)]
            if any(chunk in readable for chunk in evidence_chunks):
                problems.append(f"{question.id}: leak user {login} can read the evidence")
            elsewhere = [c.label for c in readable if facts_match(question.facts, c.search_text)]
            if elsewhere:
                # then a matching answer would not prove a leak
                problems.append(
                    f"{question.id}: leak user {login} can read the facts in {elsewhere[0]}"
                )
    return problems


def _compile(item_id: str, patterns: list[str]) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(
                f"{item_id}: fact {pattern!r} is not a valid regular expression: {error}"
            ) from None
